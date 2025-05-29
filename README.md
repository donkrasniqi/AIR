# AIR

## Neural Re-Ranking Pipeline

### Introduction
We start with BM25 to get a rough shortlist of papers for each tweet. Then, we fine-tune a small transformer (cross-encoder) to re-score that shortlist and pick the top 5 most relevant papers.

### Data Preparation
- **Index creation**  
  - Combine each paper's title and abstract into one text field.  
  - Tokenize with NLTK and build a BM25 index over these tokens.  
- **Train/Test Split**  
  - Merge and remove duplicates of the train/dev queries.  
  - Do an 80/20 random split to get `df_train_split` and `df_test_split`.

### Training Example Construction
- **Sampling positives and negatives**  
  - For each tweet in the training set, we pull 5 BM25 hits where 1 of them is a **positive** example and 4 are **negative**.  
  - Make sure the true paper ID is in that list. If it isn’t, swap it in at the last spot.  
- **Labeling**  
  - Create one **positive** example (label `1.0`) for the true paper and four **negative** examples (label `0.0`) for the other hits.  
  - Pack each (tweet, paper text, label) into an `InputExample`.

### Cross-Encoder Fine-Tuning
- **Model**  
  - Use `cross-encoder/ms-marco-MiniLM-L-6-v2`.  
- **Training**  
  - Batch size 16, train for 2 epochs.  
  - Save the best checkpoint by loss.  
- **Goal**  
  - Teach the model to give higher scores to true matches than to the four hard negatives.

### Inference & Reranking
- **Candidate retrieval**  
  - For each test tweet, get 20 BM25 candidates.  
- **Scoring**  
  - Run the transformer on each (tweet, paper) pair to get a score.  
- **Selection**  
  - Sort by score and take the top 5 papers.

### Evaluation
- **MRR@5**  
  - For each tweet, find where the true paper appears in the top 5 (rank `r`) and use `1/r`.  
  - Average these values over all tweets to get the final score.

---

## Our Contribution in the re-ranking appraoch

We designed and implemented the **1 positive : 4 hard negatives** sampling strategy, ensuring that each training query includes exactly one gold paper and four BM25-derived distractors. This focused setup teaches the cross-encoder to spot subtle semantic cues missing from pure lexical matches. We also integrated the BM25 candidate generator with the cross-encoder fine-tuning, resulting in a pipeline that trains in under two hours on a single GPU.

---

### Results
- **BM25-only baseline (MRR@5):** 0.5460  
- **Neural Re-Ranking (MRR@5):** 0.6499  

This +~0.10 MRR improvement demonstrates the effectiveness of our hard-negative approach without adding heavy computational overhead.
# AIR

Don Krasniqi – 121410189

Fisnik Berisha - 12124486

Rina Kastrati - 12129481 

Ardit Ahmeti - 12127030

Rita Selimi - 12332281 

---
# 1. Traditional Information Retrieval Approach

## Method Overview

For the traditional Information Retrieval (IR) model, we implemented a retrieval pipeline based on **BM25**, a classic bag-of-words ranking function that scores documents based on the frequency and rarity of query terms, while accounting for term saturation and document length normalization. BM25 remains a strong baseline in many IR tasks.

We used the `rank_bm25` library, a pure Python implementation of the BM25Okapi algorithm. BM25 ranks documents higher if they contain more of the query terms (especially rare ones) and penalizes very long documents.

### Our implementation performs the following steps:

1. **Document Preprocessing:**
   - Concatenated the document title and abstract from the provided CORD-19 collection.
   - Tokenized text using NLTK's `word_tokenize`.
   - Built a corpus of tokenized documents in memory.

2. **Index Building:**
   - Constructed a BM25 index using the preprocessed corpus.

3. **Claim Processing and Retrieval:**
   - Each tweet (claim) is tokenized similarly.
   - The BM25 index is queried using the processed claim.
   - The top-k relevant documents (default: `k=5`) are returned based on their BM25 scores.

> The full implementation is located in our Jupyter notebook under the **Task 1** section and supports integration with the full claim-source retrieval pipeline.

---

## Why It Qualifies as a Traditional IR Approach

The BM25 model is a canonical example of traditional information retrieval because:

- It does **not rely on neural networks** or learned representations.
- It uses **term frequency (TF)** and **inverse document frequency (IDF)** statistics computed directly from the corpus.
- The retrieval process is based entirely on **lexical overlap** between the query and the documents — no embeddings or semantic models are used.
- The approach is **unsupervised** and **parameter-free**, apart from optional hyperparameters like `k1` and `b`.

> This makes it a textbook example of a traditional IR system in the context of this project.

---

## Evaluation Setup & Performance

Since we missed the official CLEF deadline and could not submit results to Codalab, we implemented our own **local evaluation setup**:

- Merged the official train and dev splits from the dataset.
- Removed any **duplicate queries** to avoid data leakage and overfitting.
- Performed a **random 80/20 split** to create new train and test sets.
- The BM25 model was evaluated **only on the new test set**.
- The main metric used for evaluation was **MRR@5**, consistent with the CLEF task guidelines.

---

## Results

On our local test split, the **BM25-based system achieved an MRR@5 of 0.5460**, demonstrating solid baseline performance. This provides a strong foundation for comparison against the team’s **neural IR approaches**, which aim to improve over this traditional method.

# Neural Information Retrieval: Bi-Encoder Evaluation

## Overview

In this project, we also evaluated the performance of various **Bi-Encoder models** for document retrieval within a Neural Information Retrieval (Neural IR) pipeline. The models were assessed using **Mean Reciprocal Rank (MRR)** at different cutoff levels — specifically **MRR@1**, **MRR@5**, and **MRR@10** — on a shared development dataset of queries and scientific documents.

---

## Why Use Bi-Encoders?

We chose **Bi-Encoders** due to their **computational efficiency and scalability**. In contrast to Cross-Encoders, which require evaluating each query-document pair jointly, Bi-Encoders **independently encode queries and documents** into dense vector embeddings. These embeddings can then be compared using fast similarity metrics (e.g., cosine similarity), enabling **real-time retrieval** from large-scale document collections.

This makes Bi-Encoders particularly well-suited for use as **first-stage retrievers** in multi-stage IR pipelines.

---

## Models Evaluated and Results

The table below summarizes the retrieval performance of each model using MRR scores:

| Model Name                                               | MRR@1   | MRR@5   | MRR@10  |
|----------------------------------------------------------|---------|---------|---------|
| `intfloat/e5-large-v2`                                   | 0.5771  | 0.6394  | 0.6461  |
| `intfloat/e5-base-v2`                                    | 0.5400  | 0.5994  | 0.6071  |
| `sentence-transformers/all-MiniLM-L6-v2`                 | 0.4157  | 0.4897  | 0.4994  |
| `sentence-transformers/msmarco-distilbert-base-v4`       | 0.3650  | 0.4280  | 0.4344  |
| `sentence-transformers/msmarco-MiniLM-L-12-v3`           | 0.3479  | 0.4126  | 0.4218  |

---

## Observations

- The **`intfloat/e5-large-v2`** model achieved the highest scores across all MRR levels, demonstrating its strong capability as a dense retriever. This model benefits from **larger parameter capacity** and training on **diverse retrieval tasks**.
- The **`intfloat/e5-base-v2`** model also performed very well, making it a viable alternative when **computational efficiency** is a concern.
- **MiniLM-based models** such as `msmarco-MiniLM-L-12-v3` and `all-MiniLM-L6-v2` were **significantly faster** but consistently underperformed the E5 models in retrieval accuracy.
- Models trained specifically for **retrieval tasks** (e.g., MSMARCO, BEIR) yielded **noticeably better performance** than those trained for general sentence similarity.

---

## Conclusion

**Bi-Encoder models** offer an effective balance between **retrieval performance** and **inference speed**, making them a practical choice for **scalable document retrieval systems**. Among the models tested, the **E5 family** stood out as the most reliable for achieving **high retrieval accuracy**, particularly in **top-5 and top-10 rankings**.



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

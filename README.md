# AIR

Don Krasniqi – 121410189

Fisnik Berisha - 12124486

Rina Kastrati - 12129481 

Ardit Ahmeti - 12127030

Rita Selimi - 12332281 

#### Github Repository - `https://github.com/donkrasniqi/AIR`

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
To explore the most effective way to fine-tune our neural re-ranker, we implemented and tested two different training strategies. Both methods produced similar results in terms of MRR@5.

### First Approach
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

### Our Contribution in the re-ranking approach

We designed and implemented the **1 positive: 4 hard negatives** sampling strategy, ensuring that each training query includes exactly one gold paper and four BM25-derived distractors. This focused setup teaches the cross-encoder to spot subtle semantic cues missing from pure lexical matches. We also integrated the BM25 candidate generator with the cross-encoder fine-tuning, resulting in a pipeline that trains in under two hours on a single GPU.

### Results
- **BM25-only baseline (MRR@5):** 0.5460  
- **Neural Re-Ranking (MRR@5):** 0.6499  

This +~0.10 MRR improvement demonstrates the effectiveness of our hard-negative approach without adding heavy computational overhead.

## Second Approach
### Method Overview  
To boost the accuracy of our retrieval results, we built a neural re-ranking system using a fine-tuned CrossEncoder. Unlike traditional methods, CrossEncoders take both the query and document together and look at how they relate, which helps them better understand the context and relevance between them.  
We started with the `cross-encoder/ms-marco-MiniLM-L-12-v2`, a lightweight but surprisingly strong model built for semantic relevance tasks. We then fine-tuned it, tailoring the model more specifically to our domain.

### Training Data Generation  
To train our CrossEncoder, we needed a dataset of query-document pairs labeled as either relevant or not. We created this using the tweet-document pairs from the training split:

- For each tweet, we added one positive pair by matching it with its correct document and labeling it with 1.0.  
- For the negative pair, we used our BM25 index to retrieve the top 20 candidate documents.  
- We skipped the gold document and picked the highest-ranked non-matching one, labeling it 0.0.  

This gave us a balanced training set where each tweet had exactly one positive and one negative example. All examples were wrapped in `InputExample` format to prepare for training.

### Fine-Tuning  
To train the model, we used the `CrossEncoder.fit()` function provided by the SentenceTransformers library with the following setup:

- Batch size of 16 to ensure stable gradient updates without overloading memory.  
- Only 1 training epoch, since our dataset was relatively small and we wanted to avoid overfitting.  
- Warmup steps set to 100 to let the model adjust gradually before full training kicks in.  
- The model was saved to `./finetuned_crossencoder` for later use.  

We reloaded the saved model from disk for inference during evaluation.

### Re-Ranking Process  
To test how well our fine-tuned model works, we used it to re-rank the documents retrieved by BM25. Here’s how it went:

- For each tweet in the test set, we pulled the top 50 documents using our BM25 index.  
- Then, for each of those 50 documents, we paired the tweet with the document (title + abstract) and passed them together into the CrossEncoder.  
- The model gave us a relevance score for each pair.  
- We used these scores to sort the documents from most to least relevant.  
- Finally, we kept only the top 5 documents per tweet to evaluate how good the model was at ranking the true document near the top.

### Why It Qualifies  
This approach qualifies as a neural re-ranking method because it doesn’t just look at word overlap, it actually learns what makes a document relevant to a tweet. Instead of relying on basic keyword matching, it uses a deep language model to understand the meaning and context of both the tweet and the document. Since we fine-tuned it on our own dataset, the model learned how tweets and biomedical abstracts usually relate, which helped a lot with ranking the most relevant documents higher.

### Evaluation Setup & Performance  
We used the same local evaluation setup described earlier (80/20 train-test split, deduplicated queries). The main metric was again MRR@5.

The CrossEncoder re-ranking significantly improved performance:

- **BM25-only MRR@5**: 0.5460  
- **Fine-tuned CrossEncoder MRR@5**: 0.6350  

This shows the clear benefit of leveraging supervised neural modeling over lexical BM25 scoring.

### Model Selection and Experiments  
We tried out two different CrossEncoder models:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`: it is smaller and runs faster, but we saw lower performance, around 0.58 MRR@5.  
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: this one performed better overall, so we stuck with it as our final choice.

We also experimented with how many BM25 candidates to re-rank (`topk`). We tested 20, 50, and 100. `topk = 50` gave us the best trade-off, it was deep enough to include relevant documents but not too expensive to compute.

### Challenges Faced  
- We initially encountered environment compatibility issues with `tensorflow` and `keras`, due to version conflicts with Transformers. This was solved by enforcing `TRANSFORMERS_NO_TF=1` and avoiding TensorFlow dependencies.  
- Pushing the fine-tuned model to GitHub failed due to large file limits (>100MB). We addressed this by excluding the model folder using `.gitignore` and documenting it locally instead.

### Results  
Our neural re-ranking pipeline successfully enhanced retrieval effectiveness by integrating the previous models for recall with a fine-tuned CrossEncoder for precision. Fine-tuning yielded a measurable and significant improvement in MRR@5 over the baseline, with a result of **0.6352**, justifying the added complexity.


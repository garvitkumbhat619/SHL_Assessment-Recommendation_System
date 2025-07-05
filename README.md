# SHL-Assessment Recommendation System

## Project Overview

This project builds an intelligent recommendation system to assist hiring managers in selecting relevant SHL assessments based on a natural language job description or query. The system uses semantic search over embedded assessment data to return the top recommendations.

> The goal is to simplify and improve the efficiency of finding SHL assessments using modern GenAI techniques.

---

## What This Does

- Takes **a natural language job description or query** as input.
- Returns **1 to 10 relevant SHL assessments** based on semantic similarity.
- Each recommendation includes:
  - Assessment Name + Link
  - Duration
  - Test Type(s)
  - Remote Testing Support
  - Adaptive/IRT Support

---

### 📄 Preview of the PDF

![PDF Preview](https://github.com/garvitkumbhat619/SHL_Assessment-Recommendation_System/blob/e5e8d6bc0b7d6c78558439ed3fa4283aec5ef21f/SHL%20Assessment%20Recommender-page-001_new.jpg)

---

## Directory Structure

```
Embeddings_shl/
├── outputs/
│   ├── assessment_embeddings.npy
│   ├── assessment_texts.json
│   ├── faiss_index.idx
│   ├── eval_results.xlsx
│   └── ...
├── benchmark_eval.py           # Evaluation script (MAP@K, Recall@K)
├── clean_metadata.py           # Clean raw SHL metadata
├── generate_embeddings.py      # Generate and save embeddings
├── handle_query.py             # Load query and get top N results
├── query_eval_set.json         # Sample test queries
├── requirements.txt
├── scraper.py                  # Crawl data from SHL catalog
├── shl_metadata_index.csv/json
```
---

## Tech Stack

| Task | Tool |
|------|------|
| Embeddings | [INSTRUCTOR-XL](https://huggingface.co/hkunlp/instructor-xl) |
| Semantic Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Language | Python 3.10 |
| Evaluation Metrics | MAP@K, Recall@K |

---

## Your Approach

### 1. Data Collection & Preprocessing
- Scraped SHL assessments using `scraper.py`.
- Cleaned and structured metadata with `clean_metadata.py`.

### 2. Embedding Generation
- Generated sentence embeddings using `hkunlp/instructor-xl` model.
- Stored them in `assessment_embeddings.npy` and indexed with FAISS.

### 3. Query Processing
- Input query is embedded using the same model.
- FAISS returns top relevant assessment vectors.
- Final results are parsed with key metadata.

---

## How to Run (Step-by-Step)

### 1. Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

---

### 2. Generate Embeddings (if not already present)

```bash
python generate_embeddings.py
```

---

### 3. Run a Test Query from Terminal

```bash
python handle_query.py
```

You’ll be prompted to enter your query, e.g.:

```
Looking for Python and SQL developer assessments within 45 minutes.
```

The terminal will output the top recommended assessments in a tabular format.

---

### 4. Evaluate Model (MAP@3, Recall@3)

```bash
python benchmark_eval.py
```

This script uses `query_eval_set.json` and prints out benchmark metrics.

---

## Evaluation Metrics

- **Mean Recall@3** – Measures how many relevant items were retrieved.
- **MAP@3 (Mean Average Precision)** – Measures quality and ranking of recommendations.

> These are computed using `benchmark_eval.py`.

---

## Sample Queries to Try

- “I’m hiring for Java developers who can also collaborate with business teams. The test should be within 40 minutes.”
- “Looking to hire mid-level professionals skilled in Python, SQL, and JavaScript. Max duration: 60 mins.”
- “Need an analyst assessment with both Cognitive and Personality tests, within 45 mins.”

---

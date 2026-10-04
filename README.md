# LegalRAG : Legal Question Answering System

An advanced Retrieval-Augmented Generation (RAG) system for Indian Legal Acts (including the Information Technology Act 2000 and Consumer Protection Act 2019). The system compares **Dense**, **Hybrid (BM25 + Dense with Reciprocal Rank Fusion)**, and **Hybrid + Cross-Encoder Reranking** retrieval pipelines, generating strictly grounded answers with section citations via the Google Gemini API.

> **Disclaimer**: This system is designed solely for educational, academic, and research purposes. It does not constitute formal legal advice.

---

## Architecture Overview

```
                      User Question
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
  Dense Retrieval                         Sparse Retrieval
 (FAISS + all-MiniLM-L6-v2)                 (BM25Okapi)
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
              Reciprocal Rank Fusion (RRF)
                  [Top-N Candidates]
                            │
                            ▼
                  Cross-Encoder Reranker
              (ms-marco-MiniLM-L-6-v2)
                            │
                            ▼
                   Top-K Legal Chunks
                            │
                            ▼
                   Grounded Generation
               (Google Gemini API with
              Document/Section Citations)
                            │
                            ▼
                       Final Answer
```

### Retrieval Pipelines Evaluated

1. **Dense Retrieval**:
   - Encodes text with `sentence-transformers/all-MiniLM-L6-v2`.
   - Uses `faiss.IndexFlatIP` with normalized vectors for exact cosine similarity search.
2. **Hybrid Retrieval**:
   - Combines semantic dense retrieval with keyword-based `BM25Okapi`.
   - Fuses rankings using Reciprocal Rank Fusion:
     $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k = 60$.
3. **Hybrid + Cross-Encoder Reranking**:
   - Takes top candidate chunks from Hybrid RRF fusion.
   - Evaluates full query-document interaction using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Re-ranks candidates by relevance probability before feeding into the LLM context.

---

## Project Structure

```
LegalRAG/
├── app.py              # Streamlit interactive web interface
├── ingest.py           # PDF ingestion & section-aware chunking pipeline
├── retrieve.py         # Dense, Hybrid (RRF), and Cross-Encoder retrieval methods
├── generate.py         # Grounded prompt synthesis & Gemini API integration
├── evaluate.py         # Benchmark suite (P@K, R@K, Hit@K, MRR, latency)
├── config.yaml         # Centralized configuration (models, chunking, top-k)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git exclusion rules (protects API keys)
├── data/               # Source legal PDFs
│   ├── ITAct2000.pdf
│   └── The Consumer Protection Act, 2019.pdf
└── chunks.json         # Processed section-aware chunk repository
```

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and provide your Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Ingest Documents
Process the PDFs in `data/` into section-aware chunks:
```bash
python ingest.py
```
This parses legal section headers (e.g., `Section 43A`, `Section 66`) and creates `chunks.json`.

### 4. Run the Streamlit Web Demo
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. You can select your desired retrieval method (`Dense`, `Hybrid`, or `Hybrid + Reranking`), enter any legal query, and inspect the response time, grounded answer, and expandable source citations.
# RAG-Bench: Retrieval Evaluation Framework for RAG Pipelines

RAG-Bench is a modular evaluation framework for benchmarking the **retrieval performance** of components in Retrieval-Augmented Generation (RAG) systems.  
It computes standard IR metrics and supports plug-and-play evaluation for multiple retrievers and rerankers.

[📄 Planning and Design Doc →](https://docs.google.com/document/d/1vuv3pliy8DV-ipau8KcpQVdp-q1hpTLvozZq0eNrvfA/edit?usp=sharing)

---

## 🎯 Project Objective

- Evaluate retrievers (currently: BM25) on a fixed query set and document store.
- Compute standard retrieval metrics: Precision@K, Recall@K, and NDCG@K.
- Determine if each retriever or reranked pipeline meets configurable performance thresholds.
- Support modular extension to dense retrievers, hybrid methods, and LLM-grounded scoring.
- Visualize query-level behavior and retrieval quality via a Streamlit dashboard.

---

## 🖥️ Dashboard Preview

![Dashboard Demo](dashboard_preview.gif)

---

## 📁 Project Structure

```
rag-bench/
├── data/                     # Corpus, metadata, and ground truth
├── queries/                  # Query sets (JSON or CSV)
├── retrievers/               # Retrieval modules (new retrieval scripts go here)
│   ├── base_retriever.py
│   └── bm25_retriever.py
├── rerankers/                # Reranking modules (new reranker scripts go here)
│   ├── base_reranker.py
│   └── bge_reranker.py       # HuggingFace cross-encoder reranker
├── evaluation/               # Metric computation and threshold evaluation
│   └── evaluator.py
├── reports/                  # Output reports (JSON/CSV)
├── utils/                    # Shared utilities
│   └── data_loader.py
├── dashboard/                # Streamlit dashboard
│   └── app.py
├── main.py                   # Orchestration script (CLI)
├── requirements.txt
└── README.md
```

---

## 🛠 Core Components

| Component        | Responsibility |
|------------------|----------------|
| **Retriever Interface**   | Unified API for plug-in retrievers (BM25 complete) |
| **Reranker Interface**    | Cross-encoder reranking support via `BGE-Reranker` |
| **Evaluator**             | Computes metrics and checks thresholds |
| **Report Generator**      | Outputs results to structured JSON/CSV |
| **Orchestrator (`main.py`)** | Runs retrievers/rerankers via registry + CLI |
| **Test Suite**            | Unit tests for retrievers, rerankers, and evaluation logic |

---

## 📊 Current Retrieval Metrics

- **Precision@K**
- **Recall@K**
- **NDCG@K**

Framework is extensible to:
- MRR, MAP
- LLM-based grounding and hallucination detection (future)

---

## 🚀 How To Run

```bash
python main.py \
  --corpus data/corpus.json \
  --queries data/queries.json \
  --gt data/qrels.json \
  --retrievers bm25 \
  --rerankers bge \
  --report reports/eval_reranked.json \
  --topk 5
```

✅ CLI supports multiple retrievers and rerankers using a clean registry pattern.

---

## 📈 Phase 2 Features (Completed)

- ✅ Integrated `BGE-Reranker` (cross-encoder) for document re-scoring
- ✅ Refactored orchestration logic with `run_pipeline()` abstraction
- ✅ Unit tested reranker integration with edge cases
- ✅ Built an interactive Streamlit dashboard for:
  - 📊 Metric comparison across retrieval strategies
  - 🔍 Per-query exploration of retrieved document content
  - ⚠️ Failure mode filtering and performance debugging

---

## 📋 Current Status

✅ Phase 1 complete: retriever implementation, evaluator, CLI  
✅ Phase 2 complete: reranker integration + Streamlit dashboard  
✅ Project fully functional and demo-ready

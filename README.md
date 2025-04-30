Here’s your updated `README.md` that reflects the **current progress**, including:

- Completion of BM25 retriever  
- Completed evaluator with metrics and threshold checks  
- Full CLI integration  
- Reranker + Streamlit dashboard planned as Phase 2

---

```markdown
# RAG-Bench: Retrieval Evaluation Framework for RAG Pipelines

RAG-Bench is a modular evaluation framework for benchmarking the **retrieval performance** of different components in Retrieval-Augmented Generation (RAG) systems.  
It computes standard information retrieval metrics and supports plug-and-play evaluation for various retrievers and rerankers.

[📄 Planning and Design Doc →](https://docs.google.com/document/d/1vuv3pliy8DV-ipau8KcpQVdp-q1hpTLvozZq0eNrvfA/edit?usp=sharing)

---

## 🎯 Project Objective

- Evaluate retrievers (currently: BM25) on a fixed query set and document store.
- Compute standard retrieval metrics: Precision@K, Recall@K, and NDCG@K.
- Determine if each retriever meets predefined performance thresholds.
- Enable modular extension with rerankers, hybrid retrievers, and future QA scoring.
- Visualize retrieval behavior and failure cases through an interactive dashboard (Phase 2).

---

## 📁 Project Structure

```
rag-bench/
├── data/                     # Indexed document embeddings, metadata, and ground truth
├── queries/                  # Query sets (JSON or CSV)
├── retrievers/               # Retrieval implementations
│   ├── base_retriever.py
│   └── bm25_retriever.py
├── rerankers/                # (Planned) Reranker modules
│   ├── base_reranker.py
│   └── bge_reranker.py       # (To be added in Phase 2)
├── evaluation/               # Metric computation and performance checking
│   └── evaluator.py
├── reports/                  # Evaluation result outputs (JSON/CSV)
├── utils/                    # Helper utilities
│   └── data_loader.py
├── dashboard/                # (Planned) Streamlit dashboard for Phase 2
│   └── app.py
├── main.py                   # CLI orchestration script
├── README.md
├── requirements.txt
└── setup.py                  # (Optional, for pip packaging)
```

---

## 🛠 Core Components

| Component        | Responsibility |
|------------------|----------------|
| **Data Loader**        | Loads queries, documents, and optional ground truth labels |
| **BM25 Retriever**     | Sparse bag-of-words retriever using `rank_bm25` |
| **Evaluator**          | Computes Precision@K, Recall@K, NDCG@K per query and per retriever |
| **Threshold Checker**  | Flags whether each retriever passes/fails based on configured metrics |
| **Report Generator**   | Saves results in structured CSV/JSON formats |
| **Main Orchestrator**  | Runs the full pipeline from CLI |

---

## 📊 Current Retrieval Metrics

- **Precision@K**
- **Recall@K**
- **NDCG@K**

(Framework designed to support MRR, MAP, and LLM-based grounding checks in future phases.)

---

## 🚀 How To Run

```bash
python main.py \
  --retrievers bm25 \
  --query_file queries/queries.json \
  --ground_truth_file data/ground_truth.json \
  --output_dir reports/ \
  --k 5
```

✅ CLI includes arguments for retriever name(s), top-K, input/output paths, and will support reranking in Phase 2.

---

## 🧪 Phase 2 Roadmap (In Progress)

- ✅ Integrate `BGE-Reranker` (cross-encoder) to improve document ordering
- ✅ Add `base_reranker.py` to standardize reranker interface
- 🔜 Update `main.py` to support reranking via CLI flag
- 🔜 Build **Streamlit dashboard** for visual exploration:
  - Metric comparison across retrievers
  - Per-query inspection of retrieved docs
  - Failure mode analysis (low recall, missing ground truth, etc.)

---

## 📋 Current Status

✅ System design, planning, and Phase 1 MVP completed  
✅ BM25 retriever, evaluation engine, and CLI interface implemented and tested  
🚧 Phase 2: Reranker integration and dashboard development **in progress**
```

---

Would you like me to package this up as a downloadable file or just let you copy it into your repo?

# RAG-Bench: Retrieval Evaluation Framework for RAG Pipelines

RAG-Bench is a modular framework to evaluate the retrieval component performance in Retrieval-Augmented Generation (RAG) systems.  
It measures how well different retrieval methods surface relevant documents for a given set of queries, using standard information retrieval metrics.

[Planning and Design doc ->](https://docs.google.com/document/d/1vuv3pliy8DV-ipau8KcpQVdp-q1hpTLvozZq0eNrvfA/edit?usp=sharing)
---

## 🎯 Project Objective

- Evaluate multiple retrievers (e.g., BM25, DenseRetriever) on a fixed query set and document store.
- Compute standard retrieval quality metrics like Precision@K, Recall@K, and NDCG.
- Determine whether retrievers meet predefined performance thresholds.
- Provide a modular, extensible evaluation pipeline.

---

## 📁 Project Structure

```
rag-bench/
├── data/                  # Indexed document embeddings and metadata
├── queries/                # Query sets (JSON or CSV)
├── retrievers/             # Retrieval implementations
│   ├── base_retriever.py
│   ├── bm25_retriever.py
│   └── dense_retriever.py
├── evaluation/             # Metric computation and threshold evaluation
│   └── evaluator.py
├── reports/                # Generated evaluation results (JSON/CSV)
├── utils/                  # Data loaders and helper functions
│   └── data_loader.py
├── main.py                 # Main orchestration script
├── README.md
├── requirements.txt
└── setup.py (optional)
```

---

## 🛠 Core Components

| Component | Responsibility |
|-----------|-----------------|
| Data Loader | Load document embeddings, metadata, queries |
| Retriever Interface | Abstract retriever class for different retrieval methods |
| Evaluator | Compute retrieval metrics and check performance thresholds |
| Report Generator | Save evaluation results in structured format |
| Main Orchestrator | Tie everything together and run evaluation |

---

## ⚙️ Planned Retrieval Metrics

- **Precision@K**
- **Recall@K**
- **NDCG@K**
- (Extensible to MRR, MAP in future phases)

---

## 🚀 How To Run (Planned MVP Flow)

```bash
python main.py \
  --retrievers bm25 dense \
  --query_file queries/queries.json \
  --embedding_store data/embeddings.faiss \
  --ground_truth data/ground_truth.json \
  --output_dir reports/
```

✅ (Arguments and options will be finalized during Phase 1 development.)

---

## 📈 Future Enhancements

- Add hybrid retrieval strategies (combining dense and sparse retrievers).
- Extend evaluation to include generation quality (full end-to-end RAG evaluation).
- Support for real-time retrieval monitoring and dashboards.
- Integration with production-grade vector databases (e.g., FAISS, Qdrant, Milvus).

---

## 📋 Current Status

✅ Project planning and system design completed.  
🚀 Phase 1 MVP (retrieval evaluation and reporting) development underway.

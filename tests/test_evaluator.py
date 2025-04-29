import pytest
from evaluation.evaluator import Evaluator

retrieved_docs = [
    {"id": "doc1", "text": "Doc 1", "score": 0.9},
    {"id": "doc2", "text": "Doc 2", "score": 0.8},
    {"id": "doc3", "text": "Doc 3", "score": 0.5},
]

ground_truth = ["doc1", "doc3"]  # 2 of top 3 are relevant

def test_performance_check_with_ground_truth():
    evaluator = Evaluator(k=3)
    results = evaluator.performance_check(retrieved_docs, ground_truth)

    assert "precision@5" in results
    assert "recall@5" in results
    assert "ndcg@5" in results

    assert results["precision@5"]["value"] == pytest.approx(2/3, rel=1e-2)
    assert results["recall@5"]["value"] == pytest.approx(1.0, rel=1e-2)
    assert results["ndcg@5"]["value"] > 0.0

    assert results["precision@5"]["status"] in {"pass", "fail"}

def test_performance_check_without_ground_truth():
    evaluator = Evaluator(k=3)
    results = evaluator.performance_check(retrieved_docs, ground_truth=None)

    assert "precision@5" in results
    assert "recall@5" not in results
    assert "ndcg@5" not in results

def test_performance_check_empty_retrieval():
    evaluator = Evaluator(k=3)
    results = evaluator.performance_check([], ground_truth)
    assert results == {}

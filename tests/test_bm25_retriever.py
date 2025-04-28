import pytest
from retrievers.bm25_retriever import BM25Retriever
from typing import List, Dict, Any

dummy_corpus = [
    {"id": "doc1", "text": "The Milky Way galaxy is a barred spiral galaxy that contains our Solar System."},
    {"id": "doc2", "text": "Black holes are regions of spacetime where gravity is so strong that nothing can escape from it."},
    {"id": "doc3", "text": "Mars is the fourth planet from the Sun and is often referred to as the Red Planet."},
    {"id": "doc4", "text": "The James Webb Space Telescope is the most powerful telescope ever launched into space."},
    {"id": "doc5", "text": "A supernova is the explosion of a star, the largest explosion that takes place in space."},
    {"id": "doc6", "text": "Saturn is the sixth planet from the Sun and is famous for its beautiful ring system."}
]

def check_keys_in_results(results: List[Dict[str, Any]], keys: List[str]) -> bool:
    for result in results:
        for key in keys:
            if key not in result:
                return False
    return True

def check_correct_score_order(results: List[Dict[str, Any]]) -> bool:
    scores = [doc['score'] for doc in results]
    return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def test_bm25_unindexed_retrieve():
    retriever = BM25Retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("What is the Milky Way?", 3)

def test_bm25_empty_query():
    retriever = BM25Retriever()
    retriever.index(dummy_corpus)
    results = retriever.retrieve("", 3)

    # Should still return top-k documents but all scores should be 0
    assert len(results) == 3
    assert all(doc["score"] == 0 for doc in results), "Expected all scores to be zero for empty query"


def test_bm25_valid_retrieve():
    retriever = BM25Retriever()
    retriever.index(dummy_corpus)
    query = "galaxy and Solar System"
    k = 3
    results = retriever.retrieve(query, k)

    print("Results from valid retrieval test:", results)

    assert len(results) == k, f"Expected {k} documents, got {len(results)}"
    required_keys = ["id", "text", "score"]
    assert check_keys_in_results(results, required_keys), "Missing required keys in result"
    assert check_correct_score_order(results), "Scores are not sorted descending"

def test_bm25_k_greater_than_corpus_size():
    retriever = BM25Retriever()
    retriever.index(dummy_corpus)
    query = "planet"
    k = 10  # Requesting more than corpus size

    results = retriever.retrieve(query, k)

    # Should not return more than corpus size
    assert len(results) <= len(dummy_corpus), "Should not return more than available documents"

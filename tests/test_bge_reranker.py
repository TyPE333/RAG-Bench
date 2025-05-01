from rerankers.bge_reranker import BGEReranker
from typing import List, Dict, Any

def check_correct_score_order(results: List[Dict[str, Any]]) -> bool:
    scores = [doc['reranker_score'] for doc in results]
    return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def check_keys_in_results(results: List[Dict[str, Any]], keys: List[str]) -> bool:
    return all(all(key in doc for key in keys) for doc in results)

def test_bge_empty_query():
    query = ""
    documents = [
        {"id": "1", "text": "Climate change is caused by CO2."},
        {"id": "2", "text": "Electric vehicles reduce emissions."}
    ]

    reranker = BGEReranker()
    reranked_docs = reranker.rerank(query, documents)

    assert len(reranked_docs) == len(documents), "Expected all documents returned"
    for doc in reranked_docs:
        assert "reranker_score" in doc, "Missing score key"
        assert isinstance(doc["reranker_score"], float), "Score should be float"
        assert doc["reranker_score"] < 1.0, "Unexpectedly high score for empty query"

def test_bge_valid_rerank():
    query = "galaxy and Solar System"
    documents = [
        {"id": "doc3", "text": "Mars is the fourth planet from the Sun."},
        {"id": "doc4", "text": "The James Webb Space Telescope is the most powerful telescope ever launched."},
        {"id": "doc1", "text": "The Milky Way galaxy contains our Solar System."},
        {"id": "doc2", "text": "Black holes are regions of spacetime with extreme gravity."},
    ]

    reranked_docs = BGEReranker().rerank(query, documents)

    assert len(reranked_docs) == len(documents), "Mismatch in document count"
    required_keys = ["id", "text", "reranker_score"]
    assert check_keys_in_results(reranked_docs, required_keys), "Missing keys"
    assert check_correct_score_order(reranked_docs), "Scores not sorted in descending order"

    

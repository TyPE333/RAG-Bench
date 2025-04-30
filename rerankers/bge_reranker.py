#bge reranker logic
from rerankers.base_reranker import Reranker
from typing import List, Dict, Any

class BGEReranker(Reranker):
    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank a list of documents for a given query using the BGE reranker.

        Args:
            query (str): The query string.
            docs (List[Dict]): List of documents, each with at least {'id': ..., 'text': ...}

        Returns:
            List[Dict]: Reranked list of documents with added 'score' field, sorted by descending score.
        """

        return documents
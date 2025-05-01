#abstraction for reranker module
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Re-rank a list of documents for a given query.

        Args:
            query (str): The query string.
            docs (List[Dict]): List of documents, each with at least {'id': ..., 'text': ...}

        Returns:
            List[Dict]: Reranked list of documents with added 'score' field, sorted by descending score.
        """
        pass

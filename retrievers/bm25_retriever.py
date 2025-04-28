from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from retrievers.base_retriever import BaseRetriever
from utils.tokenizer import simple_tokenize

class BM25Retriever(BaseRetriever):
    """
    BM25 retriever for document retrieval.
    """
    def __init__(self) -> None:
        self.bm25 = None
        self.corpus = None

    def index(self, corpus: List[Dict[str, str]]) -> None:
        tokenized_corpus = [simple_tokenize(doc['text']) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError("The index has not been built. Please call index() first.")
        
        # Tokenize the query
        tokenized_query = simple_tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank documents by scores (descending order)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Retrieve the top-k documents
        top_k = ranked_indices[:k]
        
        # Prepare the result as a list of dictionaries with document id, text and score
        results = [
            {
                "id": self.corpus[idx]["id"],
                "text":self.corpus[idx]["text"],
                "score": scores[idx]
            } 
            for idx in top_k
        ]
        
        return results
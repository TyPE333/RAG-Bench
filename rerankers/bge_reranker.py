from rerankers.base_reranker import Reranker
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prepare (query, doc) pairs
        pairs = [[query, doc["text"]] for doc in documents]

        # Tokenize in batch
        encodings = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(**encodings).logits.squeeze(-1)

        scores = logits.cpu().tolist()

        for doc, score in zip(documents, scores):
            doc["reranker_score"] = score  # Consistent with evaluator naming

        # Return sorted by descending score
        return sorted(documents, key=lambda d: d["reranker_score"], reverse=True)

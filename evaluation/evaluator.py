# evaluator script for retrievers
from constants.performance import THRESHOLDS
from typing import Optional, List, Dict, Any
import math
import os
import csv
import json
class Evaluator:
    def __init__(self, k: int = 5):
        """
        Initialize evaluator.
        
        Args:
            k: Top-K value for Precision@K, Recall@K, NDCG@K
        """
        self.k = k
        self.thresholds = THRESHOLDS
    
    def generate_report(self, result: Dict[str, Dict[str, Any]], output_dir: str) -> None:
        """
        Generate report for a single query.
        """
        #Validate output_dir
        if not output_dir.endswith("/"):
            output_dir += "/"
        
        #Create output_dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(f"{output_dir}report.txt", "a") as f:
            f.write(f"Query: {result['query']}\n")
            f.write(f"Retriever: {result['retriever']}\n")
            f.write(f"Precision@{self.k}: {result['precision@k']}\n")
            f.write(f"Recall@{self.k}: {result['recall@k']}\n")
            f.write(f"NDCG@{self.k}: {result['ndcg@k']}\n")
            f.write(f"Thresholds: {result['thresholds']}\n")
            f.write("\n")

    def performance_check(
        self,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute metrics for a single query and check against thresholds.

        Args:
            retrieved_docs: List of retrieved document dicts with "id".
            ground_truth: List of ground truth document IDs (optional).

        Returns:
            Dictionary with metric names mapped to value and pass/fail status.
        """

        if not retrieved_docs:
            return {}

        # Extract just the retrieved doc IDs
        retrieved_ids = [doc["id"] for doc in retrieved_docs[:self.k]]

        # Initialize output
        results = {}

        # --- Precision@K ---
        relevant_retrieved = 0
        if ground_truth:
            relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in ground_truth)
        precision = relevant_retrieved / self.k

        threshold = self.thresholds["precision@5"]
        results["precision@5"] = {
            "value": precision,
            "threshold": threshold,
            "status": "pass" if precision >= threshold else "fail"
        }

        # --- Recall@K ---
        if ground_truth:
            recall = relevant_retrieved / len(ground_truth)
            threshold = self.thresholds["recall@5"]
            results["recall@5"] = {
                "value": recall,
                "threshold": threshold,
                "status": "pass" if recall >= threshold else "fail"
            }

        # --- NDCG@K ---
        if ground_truth:
            dcg = 0.0
            for idx, doc_id in enumerate(retrieved_ids):
                if doc_id in ground_truth:
                    dcg += 1.0 / math.log2(idx + 2)  # Rank positions start at 1

            # Compute Ideal DCG
            ideal_dcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(ground_truth), self.k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

            threshold = self.thresholds["ndcg@5"]
            results["ndcg@5"] = {
                "value": ndcg,
                "threshold": threshold,
                "status": "pass" if ndcg >= threshold else "fail"
            }

        return results


    def evaluate(
        self,
        retrievers_outputs: Dict[str, Dict[str, List[Dict[str, Any]]]],
        ground_truth: Optional[Dict[str, List[str]]] = None,
        output_dir: str = "reports/"
    ) -> None:
        """
        Evaluate all retrievers across all queries and write reports.

        Args:
            retrievers_outputs: Dict[retriever_name -> Dict[query_id -> list of retrieved docs]]
            ground_truth: Dict[query_id -> list of relevant doc IDs]
            output_dir: Folder to save evaluation reports
        """

        os.makedirs(output_dir, exist_ok=True)

        for retriever_name, retriever_queries in retrievers_outputs.items():
            print(f"Evaluating {retriever_name}...")
            all_results = []

            for query_id, retrieved_docs in retriever_queries.items():
                gt_ids = ground_truth.get(query_id, []) if ground_truth else None
                metrics = self.performance_check(retrieved_docs, gt_ids)
                
                # Record one row per query
                row = {
                    "query_id": query_id,
                    "retriever": retriever_name
                }
                for metric_name, metric_result in metrics.items():
                    row[f"{metric_name}_value"] = metric_result["value"]
                    row[f"{metric_name}_status"] = metric_result["status"]
                
                # Add the ids of retrieved docs to the row
                row["retrieved_doc_ids"] = [doc["id"] for doc in retrieved_docs]
                
                all_results.append(row)

            # Save CSV file
            output_csv = os.path.join(output_dir, f"{retriever_name}_eval_report.csv")
            with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

            # Save JSON version
            output_json = os.path.join(output_dir, f"{retriever_name}_eval_report.json")
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

            print(f"Saved evaluation report for {retriever_name} at {output_csv}")

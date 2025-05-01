from argparse import ArgumentParser
import logging
import os
import json

from retrievers.registry import RETRIEVER_REGISTRY
from rerankers.registry import RERANKER_REGISTRY
from evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(description="RAG-Bench: Retrieval Evaluation Framework")
    parser.add_argument("--corpus", type=str, default="data/corpus.json")
    parser.add_argument("--queries", type=str, default="data/queries.json")
    parser.add_argument("--gt", type=str, default="data/qrels.json")
    parser.add_argument("--retrievers", type=str, default="bm25")
    parser.add_argument("--rerankers", type=str, default="")
    parser.add_argument("--report_dir", type=str, default="reports/")
    parser.add_argument("--topk", type=int, default=1000)
    return parser.parse_args()

def run_pipeline(query: dict, retriever, retriever_name: str, reranker_names: list, topk: int) -> dict:
    """
    Run retrieval and optional reranking for a single query.
    Returns a dict mapping strategy names to ranked document lists.
    """
    query_id = query["query_id"]
    query_text = query["text"]

    retrieved_docs = retriever.retrieve(query_text, topk)
    outputs = {retriever_name: retrieved_docs}

    for reranker_name in reranker_names:
        if reranker_name not in RERANKER_REGISTRY:
            logger.warning(f"Skipping unknown reranker: {reranker_name}")
            continue

        reranker = RERANKER_REGISTRY[reranker_name]()
        reranked_docs = reranker.rerank(query_text, retrieved_docs)
        strategy_name = f"{retriever_name}+{reranker_name}"
        outputs[strategy_name] = reranked_docs

    return outputs

if __name__ == "__main__":
    args = parse_args()

    with open(args.corpus, "r") as f:
        corpus = json.load(f)
    with open(args.queries, "r") as f:
        queries = json.load(f)

    gt = None
    if args.gt and os.path.exists(args.gt):
        with open(args.gt, "r") as f:
            gt = json.load(f)

    retrievers = args.retrievers.split(",")
    rerankers = args.rerankers.split(",") if args.rerankers else []

    results = {}

    for retriever_name in retrievers:
        if retriever_name not in RETRIEVER_REGISTRY:
            logger.warning(f"Skipping unknown retriever: {retriever_name}")
            continue

        logger.info(f"Indexing with retriever: {retriever_name}")
        retriever = RETRIEVER_REGISTRY[retriever_name]()
        retriever.index(corpus)

        for query in queries:
            query_id = query["query_id"]
            output_by_strategy = run_pipeline(query, retriever, retriever_name, rerankers, args.topk)

            for strategy, docs in output_by_strategy.items():
                if strategy not in results:
                    results[strategy] = {}
                results[strategy][query_id] = docs
    
    
    try:
        #check if args.report is a file and not a directory
        if os.path.exists(args.report) and not os.path.isdir(args.report):
            raise ValueError("Report directory is either not valid or does not exist")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    #if report directory is not valid, create it
    if not os.path.exists(args.report):
        os.makedirs(args.report)
        
    evaluator = Evaluator(k=args.topk)
    evaluator.evaluate(results, gt, output_dir=args.report)

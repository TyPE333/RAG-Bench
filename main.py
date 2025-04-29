from argparse import ArgumentParser
import logging
import os
import json

from retrievers.registry import RETRIEVER_REGISTRY
from evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/corpus.json")
    parser.add_argument("--queries", type=str, default="data/queries.json")
    parser.add_argument("--gt", type=str, default="data/qrels.json")
    parser.add_argument("--retrievers", type=str, default="bm25,tfidf")
    parser.add_argument("--report", type=str, default="report.json")
    parser.add_argument("--topk", type=int, default=1000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    corpus = json.load(open(args.corpus, "r"))
    queries = json.load(open(args.queries, "r"))

    gt = None
    if args.gt and os.path.exists(args.gt):
        gt = json.load(open(args.gt, "r"))

    retrievers = args.retrievers.split(",")
    results = {}

    for retriever_name in retrievers:
        if retriever_name not in RETRIEVER_REGISTRY:
            logger.error(f"Retriever {retriever_name} not found in registry.")
            continue

        retriever_class = RETRIEVER_REGISTRY[retriever_name]
        retriever_instance = retriever_class()

        retriever_instance.index(corpus)

        if retriever_name not in results:
            results[retriever_name] = {}

        for query in queries:
            query_id = query["query_id"]
            query_text = query["text"]
            print(f"Query: {query_text}")
            retriever_output = retriever_instance.retrieve(query_text, args.topk)
            print(retriever_output)
            results[retriever_name][query_id] = retriever_output

    evaluator = Evaluator(k=args.topk)
    evaluator.evaluate(results, gt)

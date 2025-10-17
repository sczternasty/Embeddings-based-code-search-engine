from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple

import httpx
from datasets import load_dataset


def recall_at_k(relevant: List[str], ranked_ids: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked_ids[:k])
    rel_set = set(relevant)
    hits = len(top_k & rel_set)
    return float(hits) / float(len(rel_set))


def mrr_at_k(relevant: List[str], ranked_ids: List[str], k: int = 10) -> float:
    rel_set = set(relevant)
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(gains: List[int], k: int = 10) -> float:
    import math

    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        if i == 1:
            dcg += float(g)
        else:
            dcg += float(g) / math.log2(i)
    return dcg


def ndcg_at_k(relevant: List[str], ranked_ids: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    gains_ranked = [1 if doc_id in rel_set else 0 for doc_id in ranked_ids]
    dcg = dcg_at_k(gains_ranked, k)
    ideal_gains = [1] * min(len(rel_set), k)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(qrels: Dict[str, List[str]], run: Dict[str, List[str]], k: int = 10) -> Dict[str, float]:
    queries = list(qrels.keys())
    if not queries:
        return {f"Recall@{k}": 0.0, f"MRR@{k}": 0.0, f"NDCG@{k}": 0.0}
    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    for qid in queries:
        relevant = qrels.get(qid, [])
        ranked = run.get(qid, [])
        recalls.append(recall_at_k(relevant, ranked, k))
        mrrs.append(mrr_at_k(relevant, ranked, k))
        ndcgs.append(ndcg_at_k(relevant, ranked, k))
    return {f"Recall@{k}": sum(recalls) / len(recalls), f"MRR@{k}": sum(mrrs) / len(mrrs), f"NDCG@{k}": sum(ndcgs) / len(ndcgs)}


def load_cosqa_hf(split: str = "test") -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:

    corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split='corpus')
    queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split='queries')
    qrels_ds = load_dataset("CoIR-Retrieval/cosqa", "default", split=split)

    docs: List[dict] = []
    for item in corpus_ds:
        doc_id = str(item.get("_id"))
        text = str(item.get("text"))
        metadata = {"language":str(item.get("language")), "meta": str(item.get("meta_information"))}
        if doc_id and text:
            docs.append({"id": doc_id, "text": text, "metadata": None})

    queries: Dict[str, str] = {}
    for item in queries_ds:
        qid = str(item.get("_id"))
        text = item.get("text")
        if qid and text:
            queries[qid] = text

    qrels: Dict[str, List[str]] = {}
    for item in qrels_ds:
        qid = str(item.get("query-id"))
        doc_id = str(item.get("corpus-id"))
        label = int(item.get("score"))
        if label != 0 and qid and doc_id:
            qrels.setdefault(qid, []).append(doc_id)

    return docs, queries, qrels


def post_index(api_base: str, documents: List[dict], batch_size: int = 1000) -> None:
    api = api_base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            resp = client.post(f"{api}/index", json={"documents": batch})
            resp.raise_for_status()


def run_search(api_base: str, queries: Dict[str, str], k: int) -> Dict[str, List[str]]:
    api = api_base.rstrip("/")
    run: Dict[str, List[str]] = {}
    with httpx.Client(timeout=60.0) as client:
        for qid, text in queries.items():
            sresp = client.post(f"{api}/search", json={"query": text, "k": k})
            sresp.raise_for_status()
            payload = sresp.json()
            ranked = [item["id"] for item in payload.get("results", [])]
            run[qid] = ranked
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the API-based engine on CoSQA (Hugging Face)")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of the running API")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Dataset split")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for metrics")
    parser.add_argument("--batch_size", type=int, default=1000, help="Indexing batch size")
    args = parser.parse_args()

    documents, queries, qrels = load_cosqa_hf(split=args.split)

    post_index(args.api, documents, batch_size=args.batch_size)

    run = run_search(args.api, queries, k=args.k)

    metrics = aggregate_metrics(qrels=qrels, run=run, k=args.k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



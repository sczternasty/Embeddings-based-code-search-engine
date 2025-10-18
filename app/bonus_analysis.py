from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Any
import httpx
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


def extract_function_names(code_text: str) -> List[str]:
    """
    Extract function names from code text using regex patterns.
    
    Args:
        code_text: Source code text to analyze
        
    Returns:
        List of unique function names found in the code
    """
    function_patterns = [
        r'def\s+(\w+)\s*\(',
        r'function\s+(\w+)\s*\(',
        r'(\w+)\s*:\s*function',
        r'(\w+)\s*=\s*function',
        r'(\w+)\s*=\s*\([^)]*\)\s*=>',
        r'(\w+)\s*\([^)]*\)\s*{',
        r'public\s+\w+\s+(\w+)\s*\(',
        r'private\s+\w+\s+(\w+)\s*\(',
        r'protected\s+\w+\s+(\w+)\s*\(',
        r'static\s+\w+\s+(\w+)\s*\(',
        r'(\w+)\s*\([^)]*\)\s*;',
    ]
    
    function_names = []
    for pattern in function_patterns:
        matches = re.findall(pattern, code_text, re.IGNORECASE | re.MULTILINE)
        function_names.extend(matches)
    
    return list(set(function_names))


def load_cosqa_with_function_names(split: str = "test") -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split='corpus')
    queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split='queries')
    qrels_ds = load_dataset("CoIR-Retrieval/cosqa", "default", split=split)

    docs: List[dict] = []
    for item in corpus_ds:
        doc_id = str(item.get("_id"))
        text = str(item.get("text"))
        if doc_id and text:
            function_names = extract_function_names(text)
            if function_names:
                function_text = " ".join(function_names)
                docs.append({"id": doc_id, "text": function_text, "metadata": {"original_text": text, "function_names": function_names}})
            else:
                docs.append({"id": doc_id, "text": text, "metadata": {"original_text": text, "function_names": []}})

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


def load_cosqa_hf(split: str = "test") -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split='corpus')
    queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split='queries')
    qrels_ds = load_dataset("CoIR-Retrieval/cosqa", "default", split=split)

    docs: List[dict] = []
    for item in corpus_ds:
        doc_id = str(item.get("_id"))
        text = str(item.get("text"))
        metadata = {"language": str(item.get("language")), "meta": str(item.get("meta_information"))}
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


def start_server(model_path: str = None, port: int = 8000) -> subprocess.Popen:
    env = {}
    if model_path:
        env["FINETUNED_MODEL_PATH"] = model_path
    
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]
    return subprocess.Popen(cmd, env={**os.environ, **env})


def evaluate_experiment(experiment_name: str, documents: List[dict], queries: Dict[str, str], qrels: Dict[str, List[str]], model_path: str = None, port: int = 8000, k: int = 10) -> Dict[str, float]:
    """
    Evaluate an experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        documents: List of documents to index
        queries: Dictionary of queries to test
        qrels: Relevance judgments
        model_path: Optional path to fine-tuned model
        port: Port for the server
        k: Number of results to retrieve
        
    Returns:
        Dictionary of evaluation metrics
    """
    server = start_server(model_path, port)
    time.sleep(10)
    
    try:
        api_url = f"http://127.0.0.1:{port}"
        post_index(api_url, documents, batch_size=1000)
        run = run_search(api_url, queries, k=k)
        metrics = aggregate_metrics(qrels=qrels, run=run, k=k)
        return metrics
        
    finally:
        server.terminate()
        server.wait()


def analyze_function_names_vs_full_code():
    """
    Analyze performance differences between function names and full code.
    
    Returns:
        Dictionary containing results for all configurations
    """
    documents_full, queries, qrels = load_cosqa_hf(split="test")
    documents_functions, _, _ = load_cosqa_with_function_names(split="test")
    
    results = {}
    results["baseline_full"] = evaluate_experiment("Baseline + Full Code", documents_full, queries, qrels, None, 8000)
    results["baseline_functions"] = evaluate_experiment("Baseline + Function Names", documents_functions, queries, qrels, None, 8001)
    results["finetuned_full"] = evaluate_experiment("Fine-tuned + Full Code", documents_full, queries, qrels, "./finetuned_model", 8002)
    results["finetuned_functions"] = evaluate_experiment("Fine-tuned + Function Names", documents_functions, queries, qrels, "./finetuned_model", 8003)
    
    return results


def create_hyperparameter_engine(metric: str = "cos", dtype: str = "f32", connectivity: int = 16):
    from app.configurable_engine import ConfigurableEmbeddingSearchEngine
    return ConfigurableEmbeddingSearchEngine


def analyze_vector_storage_hyperparameters():
    """
    Analyze different vector storage hyperparameter configurations.
    
    Returns:
        Dictionary containing results for all hyperparameter configurations
    """
    documents, queries, qrels = load_cosqa_hf(split="test")
    
    hyperparameter_configs = [
        {"metric": "cos", "dtype": "f32", "connectivity": 16, "name": "Default"},
        {"metric": "cos", "dtype": "f16", "connectivity": 16, "name": "FP16"},
        {"metric": "cos", "dtype": "f32", "connectivity": 32, "name": "High Connectivity"},
        {"metric": "cos", "dtype": "f32", "connectivity": 8, "name": "Low Connectivity"},
        {"metric": "l2", "dtype": "f32", "connectivity": 16, "name": "L2 Distance"},
        {"metric": "ip", "dtype": "f32", "connectivity": 16, "name": "Inner Product"},
    ]
    
    results = {}
    
    for config in hyperparameter_configs:
        engine_class = create_hyperparameter_engine()
        
        engine = engine_class(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            metric=config["metric"],
            dtype=config["dtype"],
            connectivity=config["connectivity"]
        )
        
        for doc in documents:
            engine.add_documents([doc["id"]], [doc["text"]], [doc["metadata"]])
        
        run = {}
        for qid, query_text in queries.items():
            results_raw = engine.search(query_text, k=10)
            run[qid] = [doc_id for doc_id, _, _, _ in results_raw]
        
        metrics = aggregate_metrics(qrels=qrels, run=run, k=10)
        results[config["name"]] = metrics
    
    return results


def create_comparison_plots(function_results: Dict[str, Dict[str, float]], hyperparameter_results: Dict[str, Dict[str, float]]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ["Recall@10", "MRR@10", "NDCG@10"]
    
    baseline_full = function_results["baseline_full"]
    baseline_functions = function_results["baseline_functions"]
    finetuned_full = function_results["finetuned_full"]
    finetuned_functions = function_results["finetuned_functions"]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    axes[0, 0].bar(x - 1.5*width, [baseline_full[m] for m in metrics], width, label='Baseline + Full', alpha=0.8)
    axes[0, 0].bar(x - 0.5*width, [baseline_functions[m] for m in metrics], width, label='Baseline + Functions', alpha=0.8)
    axes[0, 0].bar(x + 0.5*width, [finetuned_full[m] for m in metrics], width, label='Fine-tuned + Full', alpha=0.8)
    axes[0, 0].bar(x + 1.5*width, [finetuned_functions[m] for m in metrics], width, label='Fine-tuned + Functions', alpha=0.8)
    
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Function Names vs Full Code Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    config_names = list(hyperparameter_results.keys())
    recall_scores = [hyperparameter_results[config]["Recall@10"] for config in config_names]
    mrr_scores = [hyperparameter_results[config]["MRR@10"] for config in config_names]
    ndcg_scores = [hyperparameter_results[config]["NDCG@10"] for config in config_names]
    
    x_config = np.arange(len(config_names))
    axes[0, 1].bar(x_config - width, recall_scores, width, label='Recall@10', alpha=0.8)
    axes[0, 1].bar(x_config, mrr_scores, width, label='MRR@10', alpha=0.8)
    axes[0, 1].bar(x_config + width, ndcg_scores, width, label='NDCG@10', alpha=0.8)
    
    axes[0, 1].set_xlabel('Configuration')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Vector Storage Hyperparameter Comparison')
    axes[0, 1].set_xticks(x_config)
    axes[0, 1].set_xticklabels(config_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    baseline_improvements = []
    finetuned_improvements = []
    for metric in metrics:
        baseline_improvement = ((baseline_functions[metric] - baseline_full[metric]) / baseline_full[metric]) * 100
        finetuned_improvement = ((finetuned_functions[metric] - finetuned_full[metric]) / finetuned_full[metric]) * 100
        baseline_improvements.append(baseline_improvement)
        finetuned_improvements.append(finetuned_improvement)
    
    x_metric = np.arange(len(metrics))
    axes[1, 0].bar(x_metric - width/2, baseline_improvements, width, label='Baseline Model', alpha=0.8)
    axes[1, 0].bar(x_metric + width/2, finetuned_improvements, width, label='Fine-tuned Model', alpha=0.8)
    
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_title('Function Names vs Full Code Improvement')
    axes[1, 0].set_xticks(x_metric)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    best_config = max(hyperparameter_results.keys(), key=lambda k: hyperparameter_results[k]["Recall@10"])
    best_scores = hyperparameter_results[best_config]
    default_scores = hyperparameter_results["Default"]
    
    improvements = []
    for metric in metrics:
        improvement = ((best_scores[metric] - default_scores[metric]) / default_scores[metric]) * 100
        improvements.append(improvement)
    
    axes[1, 1].bar(metrics, improvements, alpha=0.8, color='green')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title(f'Best Config ({best_config}) vs Default')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bonus_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function for bonus analysis.
    
    Performs comprehensive analysis of function names vs full code
    and vector storage hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Bonus analysis: function names and hyperparameters")
    parser.add_argument("--finetuned_path", default="./finetuned_model", help="Path to fine-tuned model")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for metrics")
    args = parser.parse_args()

    function_results = analyze_function_names_vs_full_code()
    hyperparameter_results = analyze_vector_storage_hyperparameters()
    
    create_comparison_plots(function_results, hyperparameter_results)


if __name__ == "__main__":
    main()

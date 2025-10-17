from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


class CodeSearchDataset(Dataset):
    def __init__(self, queries: Dict[str, str], qrels: Dict[str, List[str]], corpus: Dict[str, str]):
        self.examples = []
        for qid, query_text in queries.items():
            relevant_docs = qrels.get(qid, [])
            for doc_id in relevant_docs:
                if doc_id in corpus:
                    self.examples.append(InputExample(
                        texts=[query_text, corpus[doc_id]],
                        label=1.0
                    ))
            
            negative_docs = [doc_id for doc_id in corpus.keys() if doc_id not in relevant_docs]
            if negative_docs:
                neg_doc_id = np.random.choice(negative_docs)
                self.examples.append(InputExample(
                    texts=[query_text, corpus[neg_doc_id]],
                    label=0.0
                ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_cosqa_data(split: str = "train") -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    corpus_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split='corpus')
    queries_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split='queries')
    qrels_ds = load_dataset("CoIR-Retrieval/cosqa", "default", split=split)

    corpus = {}
    for item in corpus_ds:
        doc_id = str(item.get("_id"))
        text = str(item.get("text"))
        if doc_id and text:
            corpus[doc_id] = text

    queries = {}
    for item in queries_ds:
        qid = str(item.get("_id"))
        text = item.get("text")
        if qid and text:
            queries[qid] = text

    qrels = {}
    for item in qrels_ds:
        qid = str(item.get("query-id"))
        doc_id = str(item.get("corpus-id"))
        label = int(item.get("score"))
        if label != 0 and qid and doc_id:
            qrels.setdefault(qid, []).append(doc_id)

    return queries, corpus, qrels


def create_evaluation_data(queries: Dict[str, str], corpus: Dict[str, str], qrels: Dict[str, List[str]]) -> List[Tuple[str, str, float]]:
    eval_data = []
    for qid, query_text in queries.items():
        relevant_docs = qrels.get(qid, [])
        for doc_id in relevant_docs:
            if doc_id in corpus:
                eval_data.append((query_text, corpus[doc_id], 1.0))
        
        negative_docs = [doc_id for doc_id in corpus.keys() if doc_id not in relevant_docs]
        if negative_docs:
            neg_doc_id = np.random.choice(negative_docs)
            eval_data.append((query_text, corpus[neg_doc_id], 0.0))
    
    return eval_data


def train_model(
    model_name: str,
    train_queries: Dict[str, str],
    train_corpus: Dict[str, str],
    train_qrels: Dict[str, List[str]],
    val_queries: Dict[str, str],
    val_corpus: Dict[str, str],
    val_qrels: Dict[str, List[str]],
    output_path: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> List[float]:
    
    model = SentenceTransformer(model_name)
    
    train_dataset = CodeSearchDataset(train_queries, train_qrels, train_corpus)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = losses.CosineSimilarityLoss(model)
    
    eval_data = create_evaluation_data(val_queries, val_corpus, val_qrels)
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[item[0] for item in eval_data],
        sentences2=[item[1] for item in eval_data],
        scores=[item[2] for item in eval_data],
        name='cosqa_eval'
    )
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            optimizer_params={'lr': learning_rate},
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=None,
            save_best_model=False,
            use_amp=True
        )
        
        epoch_losses = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Evaluation"):
                embeddings1 = model.encode(batch[0], convert_to_tensor=True)
                embeddings2 = model.encode(batch[1], convert_to_tensor=True)
                labels = torch.tensor(batch[2], dtype=torch.float32)
                
                cosine_sim = torch.cosine_similarity(embeddings1, embeddings2)
                loss = train_loss(cosine_sim, labels)
                epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    model.save(output_path)
    return loss_history


def plot_training_loss(loss_history: List[float], output_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model for code search")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Base model name")
    parser.add_argument("--output_dir", default="./finetuned_model", help="Output directory for fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CoSQA training data...")
    train_queries, train_corpus, train_qrels = load_cosqa_data("train")
    
    print("Loading CoSQA validation data...")
    val_queries, val_corpus, val_qrels = load_cosqa_data("valid")

    print(f"Training samples: {len(train_queries)} queries, {len(train_corpus)} documents")
    print(f"Validation samples: {len(val_queries)} queries, {len(val_corpus)} documents")

    print("Starting fine-tuning...")
    loss_history = train_model(
        model_name=args.model_name,
        train_queries=train_queries,
        train_corpus=train_corpus,
        train_qrels=train_qrels,
        val_queries=val_queries,
        val_corpus=val_corpus,
        val_qrels=val_qrels,
        output_path=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    plot_training_loss(loss_history, os.path.join(args.output_dir, "training_loss.png"))
    
    print(f"Fine-tuning completed. Model saved to {args.output_dir}")
    print(f"Final training loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()

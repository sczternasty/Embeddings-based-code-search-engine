from __future__ import annotations
import json
import os
from typing import List
from fastapi import FastAPI
from .engine import EmbeddingSearchEngine
from .models import Document, QueryRequest, ScoredDocument, SearchResponse


def load_documents_from_path(path: str) -> List[Document]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs: List[Document] = []
    if isinstance(data, dict):
        for key, value in data.items():
            docs.append(Document(id=str(key), text=str(value)))
    elif isinstance(data, list):
        for item in data:
            docs.append(Document(**item))
    return docs


def create_app() -> FastAPI:
    app = FastAPI(title="Embeddings Code Search", version="0.1.0")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    docs_path = os.getenv("DOCS_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "documents.json"))
    docs_path = os.path.abspath(docs_path)

    engine = EmbeddingSearchEngine(model_name=model_name)

    def load_initial_docs() -> None:
        docs = load_documents_from_path(docs_path)

        engine.add_documents(
            ids=[d.id for d in docs],
            texts=[d.text for d in docs],
            metadata=[d.metadata for d in docs],
        )

    load_initial_docs()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(body: QueryRequest) -> SearchResponse:
        results_raw = engine.search(body.query, k=body.k)
        results = [
            ScoredDocument(id=doc_id, text=text, score=score, metadata=metadata)
            for doc_id, text, score, metadata in results_raw
        ]
        return SearchResponse(query=body.query, results=results)

    @app.post("/reload")
    def reload() -> dict:
        nonlocal engine
        engine = EmbeddingSearchEngine(model_name=model_name)
        load_initial_docs()
        return {"status": "reloaded", "count": len(engine._doc_ids)}

    return app


app = create_app()

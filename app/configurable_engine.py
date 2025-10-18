from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from usearch.index import Index as USearchIndex


class ConfigurableEmbeddingSearchEngine:
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        model_path: str | None = None,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.metric = metric
        self.dtype = dtype
        self.connectivity = connectivity
        self._model: SentenceTransformer | None = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metadata: List[dict | None] = []
        self._usearch_index: USearchIndex | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            if self.model_path and os.path.exists(self.model_path):
                self._model = SentenceTransformer(self.model_path, device="cpu")
            else:
                name = self.model_name
                if "/" not in name:
                    name = f"sentence-transformers/{name}"
                self._model = SentenceTransformer(name, device="cpu")
        return self._model

    def _ensure_usearch_index(self) -> None:
        if self._usearch_index is None:
            self._usearch_index = USearchIndex(
                ndim=self.dimension,
                metric=self.metric,
                dtype=self.dtype,
                connectivity=self.connectivity,
            )

    def add_documents(self, ids: List[str], texts: List[str], metadata: List[dict | None] | None = None) -> None:
        if metadata is None:
            metadata = [None] * len(ids)
        if not (len(ids) == len(texts) == len(metadata)):
            raise ValueError("ids, texts, metadata must have the same length")

        embeddings = self._encode(texts)
        embeddings = self._normalize(embeddings)

        self._ensure_usearch_index()
        assert self._usearch_index is not None

        start = len(self._doc_ids)
        keys = (np.arange(start, start + len(ids))).astype(np.uint64)
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._usearch_index.add(keys, vecs)

        self._doc_ids.extend(ids)
        self._doc_texts.extend(texts)
        self._doc_metadata.extend(metadata)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        _ = self.model
        return int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float, dict | None]]:
        if len(self._doc_ids) == 0:
            return []

        query_vec = self._encode([query])
        query_vec = self._normalize(query_vec)[0]

        neighbors = self._usearch_index.search(query_vec, k)

        keys = np.asarray(neighbors.keys)
        dists = np.asarray(neighbors.distances, dtype=np.float32)

        if self.metric == "cos":
            sims = 1.0 - dists
        elif self.metric == "l2":
            sims = 1.0 / (1.0 + dists)
        elif self.metric == "ip":
            sims = dists
        else:
            sims = 1.0 - dists

        order = np.argsort(-sims)
        keys = keys[order]
        sims = sims[order]
        results: List[Tuple[str, str, float, dict | None]] = []
        for key, sim in zip(keys, sims):
            idx = int(key)
            if 0 <= idx < len(self._doc_ids):
                results.append((self._doc_ids[idx], self._doc_texts[idx], float(sim), self._doc_metadata[idx]))
        return results[:k]

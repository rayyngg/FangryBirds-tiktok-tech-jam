from __future__ import annotations

import os
from collections.abc import Iterable


class SentenceTransformerReranker:
    """Optional dense reranker backed by sentence-transformers."""

    def __init__(self, model_name: str | None = None, batch_size: int = 64) -> None:
        self.model_name = model_name or os.getenv("AGENT_SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = batch_size
        self.model = None
        self.available = False
        self._cache: dict[str, object] = {}
        if os.getenv("AGENT_USE_SENTENCE_TRANSFORMER", "1").lower() in {"0", "false", "no"}:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def score(self, query_text: str, products: Iterable[tuple[str, str]]) -> dict[str, float]:
        if not self.available or self.model is None:
            return {}

        product_rows = list(products)
        missing = [(asin, text) for asin, text in product_rows if asin not in self._cache]
        if missing:
            embeddings = self.model.encode(
                [text for _asin, text in missing],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for (asin, _text), embedding in zip(missing, embeddings):
                self._cache[asin] = embedding

        query_embedding = self.model.encode(
            [query_text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores: dict[str, float] = {}
        for asin, _text in product_rows:
            embedding = self._cache.get(asin)
            if embedding is not None:
                scores[asin] = float(query_embedding @ embedding)
        return scores

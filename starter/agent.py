from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

try:
    from .semantic_index import SentenceTransformerReranker
    from .vector_index import HashedVectorizer, cosine_similarity
except ImportError:  # Allows copying starter/agent.py to a flat submission folder.
    from semantic_index import SentenceTransformerReranker
    from vector_index import HashedVectorizer, cosine_similarity


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str: #turns every input into string
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _product_text(product: dict) -> str:
    weighted_parts = [
        _text(product.get("title")),
        _text(product.get("title")),
        _text(product.get("categories")),
        _text(product.get("categories")),
        _text(product.get("features")),
        _text(product.get("details")),
        _text(product.get("store")),
        _text(product.get("description")),
    ]
    if product.get("price") not in (None, ""):
        weighted_parts.append(f"budget price {product['price']}")
    return " ".join(part for part in weighted_parts if part)


def _terms(text: str) -> list[str]:  #removes useless words
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Hybrid BM25, local vector, and optional sentence-transformer retrieval."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.vectorizer = HashedVectorizer(dimensions=1024)
        self.semantic_reranker = SentenceTransformerReranker()
        self._sessions: dict[str, dict] = {} #create dict to remember prev inputs
        self._product_vectors: dict[str, dict[int, float]] = {}
        self._product_texts: dict[str, str] = {}
        self._build_index() #loads all Amazon products into database

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                product_text = _product_text(product)
                self._product_texts[parent_asin] = product_text
                self._product_vectors[parent_asin] = self.vectorizer.embed(product_text)
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None: #starts new shopper conversation
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "messages":[],
            "profile": user_profile,
            "asked": set(),
        }

    def _query_text(self, state: dict, user_message: str) -> str:
        messages: list[str] = state["messages"]
        lowered = user_message.lower()
        if "actually" in lowered and ("ignore" in lowered or "instead" in lowered):
            messages.clear()
        messages.append(user_message)
        profile = state.get("profile") or {}
        profile_text = " ".join([
            _text(profile.get("summary")),
            _text(profile.get("preference_tags")),
        ])
        return " ".join([*messages[-4:], profile_text]).strip()

    def _bm25_candidates(self, query_text: str, limit: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:60]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(parent_asin), float(score)) for parent_asin, score in rows]

    def _rank(self, query_text: str, top_k: int) -> list[dict]:
        candidate_limit = max(80, min(300, top_k * 30))
        candidates = self._bm25_candidates(query_text, candidate_limit)
        if not candidates:
            return []

        query_vector = self.vectorizer.embed(query_text)
        semantic_scores = self.semantic_reranker.score(
            query_text,
            ((asin, self._product_texts.get(asin, "")) for asin, _score in candidates),
        )
        count = max(1, len(candidates) - 1)
        scored: list[tuple[float, str]] = []
        for rank, (parent_asin, _bm25_score) in enumerate(candidates):
            bm25_rank_score = 1.0 - (rank / count)
            vector_score = (cosine_similarity(query_vector, self._product_vectors.get(parent_asin, {})) + 1.0) / 2.0
            semantic_score = (semantic_scores.get(parent_asin, 0.0) + 1.0) / 2.0
            if semantic_scores:
                final_score = 0.45 * bm25_rank_score + 0.15 * vector_score + 0.40 * semantic_score
            else:
                final_score = 0.62 * bm25_rank_score + 0.38 * vector_score
            scored.append((final_score, parent_asin))
        scored.sort(reverse=True)
        return [{"parent_asin": parent_asin, "score": score} for score, parent_asin in scored[:top_k]]

    def _ask_attribute(self, state: dict, turn: int, user_message: str) -> str | None:
        lowered = user_message.lower()
        if turn >= 7:
            return None
        if "don't have a preference" in lowered or "not quite right" in lowered:
            priority = ["feature", "style", "material", "color", "use_case", "brand", "size", "budget"]
        elif turn <= 2:
            priority = ["feature", "material", "style", "color", "use_case", "brand", "size", "budget"]
        else:
            priority = ["material", "style", "feature", "color", "use_case", "brand", "size", "budget"]

        asked: set[str] = state["asked"]
        for attribute in priority:
            if attribute not in asked:
                asked.add(attribute)
                return attribute
        return None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        query_text = self._query_text(state, user_message)
        recommendations = self._rank(query_text, top_k)
        ask_attribute = self._ask_attribute(state, turn, user_message)
        return {
            "message": "Here are the closest matches I found. I can narrow them further with one more detail.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}, #0 because not using LLM right now
        }

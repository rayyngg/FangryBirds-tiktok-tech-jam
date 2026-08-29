from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except ImportError:  # keeps the agent importable in an offline / minimal environment
    OpenAI = None  # type: ignore[assignment]

load_dotenv()  # reads OPENAI_API_KEY (and optional overrides) from a local .env if present

DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore|disregard|change of mind|new requirement|what i need is)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str: ##turns every input into string
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
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._sessions: dict[str, dict] = {} #create dict to remember prev inputs
        self._product_embeddings: np.ndarray | None = None
        self._asin_to_embedding_index: dict[str, int] = {}
        self._asin_to_product_text: dict[str, str] = {}
        self.llm = self._init_llm()
        self._build_index()

    def _init_llm(self):
        """Return an OpenAI client, or None to run fully offline.

        The organizer may disable network access for final scoring, so every
        LLM call must be optional: check `if self.llm:` before using it.
        """
        if os.getenv("AGENT_USE_LLM", "1").lower() in {"0", "false", "no"}:
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        return OpenAI(api_key=api_key)

    def _llm(self, state: dict, system: str, user: str, max_tokens: int = 200) -> str | None:
        """Single chat completion; accumulates token usage into the session state.

        Returns None (never raises) when the LLM is unavailable or the call fails,
        so callers can fall back to the local heuristic path.
        """
        if self.llm is None:
            return None
        try:
            response = self.llm.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
        except Exception:
            return None
        usage = response.usage
        if usage is not None:
            state["usage"]["prompt_tokens"] += int(usage.prompt_tokens or 0)
            state["usage"]["completion_tokens"] += int(usage.completion_tokens or 0)
        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else None

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        product_asins: list[str] = []
        product_texts: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                product_asins.append(parent_asin)
                product_texts.append(_product_text(product))
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
        embedding_cache = self.catalog_path.with_name("catalog.embeddings.npy")
        if embedding_cache.exists():
            cached_embeddings = np.load(embedding_cache, mmap_mode="r")
            if cached_embeddings.shape[0] == len(product_texts):
                self._product_embeddings = cached_embeddings
            else:
                self._product_embeddings = None
        if self._product_embeddings is None:
            self._product_embeddings = self.embedding_model.encode(
                product_texts,
                batch_size=int(os.getenv("AGENT_EMBED_BATCH_SIZE", "512")),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(embedding_cache, self._product_embeddings)
        self._asin_to_embedding_index = {
            parent_asin: index
            for index, parent_asin in enumerate(product_asins)
        }
        self._asin_to_product_text = dict(zip(product_asins, product_texts))

    def reset(self, session_id: str, user_profile: dict) -> None: ##starts new shopper conversation
        # The profile is anonymized and may be used for personalization.
        profile_text = " ".join(
            [
                _text(user_profile.get("summary")),
                _text(user_profile.get("preference_tags")),
            ]
        )
        self._sessions[session_id] = {
            "profile_text": profile_text,
            "messages": [],
            "base_context": "",
            "override_context": "",
            "override_requirement": "",
            "override_active": False,
            "focus_text": "",
            "asked": set(),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _is_override(user_message: str) -> bool:
        """Detect a changed requirement without depending on one exact phrase."""
        return bool(OVERRIDE_RE.search(user_message))

    @staticmethod
    def _override_query(user_message: str) -> str:
        """Remove reset wording while preserving the new requirement."""
        cleaned = re.sub(
            r"^\s*(?:actually|instead)?\s*,?\s*"
            r"(?:ignore|disregard)\s+(?:my|the)\s+(?:earlier|previous)\s+"
            r"(?:preference|preferences|request|requirements?)\s*[,.:;-]*",
            "",
            user_message,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^\s*actually\s*[,.:;-]*\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^\s*what\s+i\s+need\s+is\s*[:,-]*\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _query_text(self, state: dict, user_message: str) -> str:
        messages: list[str] = state["messages"]
        lowered = user_message.lower()

        if not state["base_context"]:
            state["base_context"] = user_message.split(".", 1)[0].strip()

        if self._is_override(user_message):
            # Retain the product category, but discard old preferences and
            # old conversation text after a hard intent change.
            state["override_active"] = True
            state["override_context"] = state["base_context"]
            state["override_requirement"] = self._override_query(user_message)
            messages.clear()

        no_preference = (
            "don't have a preference" in lowered
            or "no preference" in lowered
            or "use your judgment" in lowered
        )

        if not no_preference:
            messages.append(user_message)
            state["focus_text"] = user_message

        #messages.append(user_message)
        recent_messages = messages[-4:]

        query_parts = []
        if state["override_context"]:
            query_parts.append(state["override_context"])
        query_parts.extend(recent_messages)
        query_parts.append(state["profile_text"])
        return " ".join(query_parts).strip()

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

    def _rank(
        self,
        query_text: str,
        top_k: int,
        override_active: bool = False,
        override_context: str = "",
        override_requirement: str = "",
        focus_text: str = "",
    ) -> list[dict]:
        candidate_limit = max(80, min(300, top_k * 30))
        candidates = self._bm25_candidates(query_text, candidate_limit)
        if not candidates or self._product_embeddings is None:
            return []

        query_embedding = self.embedding_model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        count = max(1, len(candidates) - 1)
        scored: list[tuple[float, str]] = []
        override_terms = set(_terms(override_requirement)) if override_active else set()
        override_phrase = " ".join(_terms(override_requirement)) if override_active else ""
        focus_terms = set(_terms(focus_text))
        focus_phrase = " ".join(_terms(focus_text))

        for rank, (parent_asin, _bm25_score) in enumerate(candidates):
            bm25_rank_score = 1.0 - (rank / count)

            embedding_index = self._asin_to_embedding_index.get(parent_asin)
            if embedding_index is None:
                semantic_score = 0.0
            else:
                semantic_score = float(
                    np.dot(
                        query_embedding,
                        self._product_embeddings[embedding_index],
                    )
                )
                semantic_score = (semantic_score + 1.0) / 2.0

            final_score = 0.80 * bm25_rank_score + 0.20 * semantic_score
            if override_terms:
                product_text = self._asin_to_product_text.get(parent_asin, "")
                product_terms = set(_terms(product_text))
                overlap = len(override_terms & product_terms) / len(override_terms)
                # The new requirement is a tie-breaker among recalled
                # candidates, never a replacement for lexical/semantic rank.
                final_score += 0.12 * overlap
                if len(override_terms) > 1 and override_phrase in " ".join(_terms(product_text)):
                    final_score += 0.10
            elif focus_terms:
                product_text = self._asin_to_product_text.get(parent_asin, "")
                product_terms = set(_terms(product_text))
                overlap = len(focus_terms & product_terms) / len(focus_terms)
                final_score += 0.24 * overlap
                if len(focus_terms) > 1 and focus_phrase in " ".join(_terms(product_text)):
                    final_score += 0.18
            scored.append((final_score, parent_asin))

        scored.sort(reverse=True)
        return [
            {"parent_asin": parent_asin, "score": score}
            for score, parent_asin in scored[:top_k]
        ]


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
        # Report only the tokens spent on this turn, not the session running total.
        usage_before = dict(state["usage"])

        query_text = self._query_text(state, user_message)
        recommendations = self._rank(
            query_text,
            top_k,
            state["override_active"],
            state["override_context"],
            state["override_requirement"],
            state["focus_text"],
        )
        ask_attribute = self._ask_attribute(state, turn, user_message)

        # Example LLM use: a natural customer-facing message. Falls back to a
        # fixed string when no key is configured or the call fails.
        message = self._llm(
            state,
            system="You are a concise shopping assistant. Reply in one sentence.",
            user=(
                f"The customer said: {user_message!r}. "
                f"You are showing them {len(recommendations)} products"
                + (f" and want to ask about their {ask_attribute}." if ask_attribute else ".")
            ),
            max_tokens=60,
        ) or "Here are the closest matches I found. I can narrow them further with one more detail."

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": state["usage"]["prompt_tokens"] - usage_before["prompt_tokens"],
                "completion_tokens": state["usage"]["completion_tokens"] - usage_before["completion_tokens"],
            },
        }

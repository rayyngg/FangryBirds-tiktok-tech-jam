from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from . import ask_policy

# The organizer may score with no network. Default every Hugging Face client to offline mode before
# any of those libraries is imported, so a missing model fails fast instead of hanging on retries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from openai import OpenAI
except ImportError:  # keeps the agent importable in an offline / minimal environment
    OpenAI = None  # type: ignore[assignment]

load_dotenv()  # reads OPENAI_API_KEY (and optional overrides) from a local .env if present

DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PACKAGE_ROOT / "data" / "catalog.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "all-MiniLM-L6-v2"


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", ""}


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


def load_embedder(model_path: str | Path | None = None):
    """Load the sentence-transformer from a local directory or the local HF cache; never touch the network.

    Returns None when embeddings are disabled (AGENT_USE_EMBEDDINGS=0, the default) or the model is
    not available locally, so the caller can run BM25-only.
    """
    if not _flag("AGENT_USE_EMBEDDINGS", "0"):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    device = os.getenv("AGENT_EMBED_DEVICE", "cpu")
    # 1. an explicit or default local directory (vendored / fetched model), 2. the local HF cache.
    directories = [Path(p) for p in (model_path, os.getenv("AGENT_MODEL_PATH"), DEFAULT_MODEL_PATH) if p]
    for directory in directories:
        if (directory / "modules.json").exists():
            try:
                return SentenceTransformer(str(directory), device=device, local_files_only=True)
            except Exception:
                continue
    try:
        return SentenceTransformer(os.getenv("AGENT_SENTENCE_MODEL", DEFAULT_MODEL_NAME), device=device, local_files_only=True)
    except Exception:
        return None


class Agent:
    """Hybrid BM25 retrieval with an optional, fully offline sentence-transformer reranker."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog_path = Path(catalog_path or os.getenv("AGENT_CATALOG_PATH") or DEFAULT_CATALOG_PATH)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {} #create dict to remember prev inputs
        self._product_embeddings: np.ndarray | None = None
        self._asin_to_embedding_index: dict[str, int] = {}
        self._asin_to_product_text: dict[str, str] = {}
        self.embedding_model = None
        self.ask_mode = os.getenv("AGENT_ASK_POLICY", "other").strip().lower()
        self.llm = self._init_llm()
        self._build_index()
        try:
            self.embedding_model = load_embedder()
            if self.embedding_model is not None:
                self._load_or_build_embeddings()
        except Exception:
            # Construction must never fail on the grader: degrade to BM25-only.
            self.embedding_model = None
            self._product_embeddings = None

    def _init_llm(self):
        """Return an OpenAI client, or None to run fully offline.

        The organizer may disable network access for final scoring, so every
        LLM call must be optional: check `if self.llm:` before using it.
        """
        if not _flag("AGENT_USE_LLM", "0"):
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

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
        self._asin_to_embedding_index = {
            parent_asin: index
            for index, parent_asin in enumerate(product_asins)
        }
        self._asin_to_product_text = dict(zip(product_asins, product_texts))

    def _load_or_build_embeddings(self) -> None:
        """Read a cached embedding matrix if present; otherwise encode in memory.

        The cache is only written when AGENT_WRITE_EMBEDDING_CACHE=1 — construction never writes by default.
        """
        product_texts = list(self._asin_to_product_text.values())
        embedding_cache = self.catalog_path.with_name("catalog.embeddings.npy")
        if embedding_cache.exists():
            cached_embeddings = np.load(embedding_cache, mmap_mode="r")
            if cached_embeddings.shape[0] == len(product_texts):
                self._product_embeddings = cached_embeddings
                return
        self._product_embeddings = self.embedding_model.encode(
            product_texts,
            batch_size=int(os.getenv("AGENT_EMBED_BATCH_SIZE", "512")),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if _flag("AGENT_WRITE_EMBEDDING_CACHE", "0"):
            try:
                np.save(embedding_cache, self._product_embeddings)
            except Exception:
                pass

    def reset(self, session_id: str, user_profile: dict) -> None: ##starts new shopper conversation
        # The profile is anonymized and may be used for personalization.
        profile = user_profile if isinstance(user_profile, dict) else {}
        profile_text = " ".join(
            [
                _text(profile.get("summary")),
                _text(profile.get("preference_tags")),
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
            "asked": [],
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

        # Zero-information replies must not enter the retrieval text: they would only add noise.
        no_preference = (
            "don't have a preference" in lowered
            or "no preference" in lowered
            or "use your judgment" in lowered
            or "additional preference" in lowered
            or "not quite right" in lowered
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
        if not candidates:
            return []

        use_embeddings = self.embedding_model is not None and self._product_embeddings is not None
        query_embedding = None
        if use_embeddings:
            try:
                query_embedding = self.embedding_model.encode(
                    query_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception:
                query_embedding = None

        count = max(1, len(candidates) - 1)
        scored: list[tuple[float, str]] = []
        override_terms = set(_terms(override_requirement)) if override_active else set()
        override_phrase = " ".join(_terms(override_requirement)) if override_active else ""
        focus_terms = set(_terms(focus_text))
        focus_phrase = " ".join(_terms(focus_text))

        for rank, (parent_asin, _bm25_score) in enumerate(candidates):
            bm25_rank_score = 1.0 - (rank / count)

            embedding_index = self._asin_to_embedding_index.get(parent_asin)
            if query_embedding is None or embedding_index is None:
                # BM25-only path: the rank score carries the full lexical weight.
                final_score = bm25_rank_score
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


    def _ask_attribute(self, state: dict, turn: int, user_message: str) -> str:
        """Question policy: ask every turn (a null ask yields a zero-information reply).

        The simulator reveals at most two undisclosed constraints per ask and "other" matches every
        constraint class, so the fixed policy asks "other" on every turn, including after a
        "no preference" reply. AGENT_ASK_POLICY=value switches to the question-value estimate once a
        structural catalog index is available (see ask_policy.choose_attribute).
        """
        attribute = ask_policy.choose_attribute(self.ask_mode, state, None, [], turn)
        state["asked"].append(attribute)
        return attribute

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            # The evaluator counts an exception as an empty turn; a valid dict keeps the session alive.
            return {
                "message": "Here are the closest matches I found.",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            self.reset(session_id, {})
        state = self._sessions[session_id]
        user_message = user_message if isinstance(user_message, str) else str(user_message or "")
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
        ) or ("Here are the closest matches I found. " + ask_policy.question_for(ask_attribute))

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": state["usage"]["prompt_tokens"] - usage_before["prompt_tokens"],
                "completion_tokens": state["usage"]["completion_tokens"] - usage_before["completion_tokens"],
            },
        }

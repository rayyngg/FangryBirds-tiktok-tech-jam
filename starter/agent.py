"""FangryBirds conversational shopping agent.

Per turn: parse the customer message into structured state -> rank a bounded candidate pool ->
decide how many items to show -> choose the next question -> phrase the reply. All retrieval is
local (SQLite FTS5 plus in-memory structural indexes built from the frozen catalog); the dense
sentence-transformer tier and the LLM phrasing are optional, default off, and every failure path
degrades to a valid response instead of raising.

Environment switches (all optional):
  AGENT_CATALOG_PATH       catalog location (default <repo>/data/catalog.jsonl)
  AGENT_USE_BUCKET=1       coarse-category tier (bucket members rank first)
  AGENT_PARSE_REPLIES=1    structured constraint parsing + exact/substring tiers
  AGENT_DEMOTE_SHOWN=1     items already shown this session are demoted to the tail
  AGENT_ASK_POLICY=other   other | value (question-value estimate; ablation only)
  AGENT_CONFIDENCE_GATE=1  show fewer items while the next reply is expected to pin the target (0 = always 10)
  AGENT_USE_EMBEDDINGS=0   dense tier from a local model + cached catalog matrix
  AGENT_W_POP/W_FTS/W_EMBED  blend weights for the popularity / lexical / dense scores
  AGENT_USE_LLM=0          optional OpenAI phrasing of the customer-facing message
"""
from __future__ import annotations

import os
from pathlib import Path

from . import ask_policy, embedder, parsing, retrieval
from .state import SessionState

try:  # optional: reads OPENAI_API_KEY (and overrides) from a local .env; the default path needs neither
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv not installed
    pass

try:
    from openai import OpenAI
except ImportError:  # keeps the agent importable in a minimal environment
    OpenAI = None  # type: ignore[assignment]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = PACKAGE_ROOT / "data" / "catalog.jsonl"
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RANK_WEIGHT = 0.30      # MRR weight in the technical score
TURN_WEIGHT = 0.02      # efficiency weight per turn (0.20 / 10)
GATE_MAX_TURN = 4


def _float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


class Agent:
    """Structured-state conversational search over the frozen catalog."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog_path = Path(catalog_path or os.getenv("AGENT_CATALOG_PATH") or DEFAULT_CATALOG_PATH)
        self.use_bucket = embedder.flag("AGENT_USE_BUCKET", "1")
        self.parse_replies = embedder.flag("AGENT_PARSE_REPLIES", "1")
        self.demote_shown = embedder.flag("AGENT_DEMOTE_SHOWN", "1")
        # Default decided by scripts/perturb_eval.py: the gate scored >= the full-list variant under
        # every simulator perturbation (results/*-perturb.md). AGENT_CONFIDENCE_GATE=0 restores full lists.
        self.confidence_gate = embedder.flag("AGENT_CONFIDENCE_GATE", "1")
        self.ask_mode = os.getenv("AGENT_ASK_POLICY", "other").strip().lower()
        self.weights = (_float("AGENT_W_POP", "1.0"), _float("AGENT_W_FTS", "0.0"), _float("AGENT_W_EMBED", "0.0"))
        self.index = retrieval.CatalogIndex(self.catalog_path)
        self.embedding_model = None
        self.embeddings = None
        try:
            self.embedding_model = embedder.load_embedder()
            if self.embedding_model is not None:
                cache = self.catalog_path.with_name("catalog.embeddings.npy")
                self.embeddings = embedder.load_cached_embeddings(cache, len(self.index.asins))
                if self.embeddings is None:
                    self.embedding_model = None  # never encode the catalog at construction
        except Exception:
            self.embedding_model = None
            self.embeddings = None
        self._row_of = {asin: row for row, asin in enumerate(self.index.asins)}
        self.llm = self._init_llm()
        self._sessions: dict[str, SessionState] = {}

    # ----------------------------------------------------------------- interface
    def reset(self, session_id: str, user_profile: dict) -> None:
        # The anonymised profile carries no information about the target product, so retrieval
        # ignores it; it is kept only for message phrasing.
        state = SessionState(str(session_id))
        state.log.append({"profile": user_profile if isinstance(user_profile, dict) else {}})
        self._sessions[str(session_id)] = state

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(str(session_id), user_message, int(turn), int(top_k))
        except Exception:
            return self._fallback(str(session_id), top_k)

    def session_debug(self, session_id: str) -> dict | None:
        state = self._sessions.get(str(session_id))
        return state.debug() if state else None

    def bucket_stats(self) -> dict:
        return self.index.bucket_stats()

    # ---------------------------------------------------------------- pipeline
    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self._sessions[session_id]
        state.turn = turn
        usage_before = dict(state.usage)
        top_k = max(1, top_k)

        parsed = parsing.parse_message(user_message, state, self.index)
        if not self.use_bucket:
            parsed.bucket = None
        if self.parse_replies or parsed.kind.startswith("opener"):
            parsing.apply_parsed(state, parsed, self.index)
        else:
            state.last_kind, state.last_template_matched, state.last_new_exact_info = parsed.kind, False, False
            state.free_text.append(str(user_message))

        query_text = self._query_text(state)
        ranked = retrieval.rank_candidates(
            self.index, state, query_text, embed_fn=self._embed_fn(query_text), weights=self.weights
        )
        candidates = [item.parent_asin for item in ranked]
        count = self._list_length(state, ranked, turn, top_k)
        recommendations = candidates[:count]

        attribute = ask_policy.choose_attribute(self.ask_mode, state, self.index, candidates, turn)
        state.pending_ask = attribute
        state.asked.append(attribute)
        if self.demote_shown and state.hits_count:
            for asin in recommendations:
                state.shown.setdefault(asin, turn)

        message = self._message(state, attribute, recommendations, user_message)
        state.log.append({
            "turn": turn, "kind": parsed.kind, "template": parsed.template_matched, "ask": attribute,
            "shown": len(recommendations), "top": recommendations[:3],
        })
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": {
                "prompt_tokens": max(0, state.usage["prompt_tokens"] - usage_before["prompt_tokens"]),
                "completion_tokens": max(0, state.usage["completion_tokens"] - usage_before["completion_tokens"]),
            },
        }

    def _query_text(self, state: SessionState) -> str:
        parts = [state.bucket or state.category or ""]
        parts.extend(state.known)
        parts.extend(state.free_text[-4:])
        return " ".join(part for part in parts if part).strip()

    def _embed_fn(self, query_text: str):
        if self.embedding_model is None or self.embeddings is None or self.weights[2] <= 0:
            return None
        query = embedder.encode_query(self.embedding_model, query_text)
        if query is None:
            return None
        matrix = self.embeddings
        row_of = self._row_of

        def score(asins: list[str]) -> dict[str, float]:
            import numpy as np

            rows = [row_of[asin] for asin in asins if asin in row_of]
            if not rows:
                return {}
            sims = np.asarray(matrix[rows]) @ query
            return {asin: float((sim + 1.0) / 2.0) for asin, sim in zip([a for a in asins if a in row_of], sims)}

        return score

    def _list_length(self, state: SessionState, ranked: list[retrieval.Scored], turn: int, top_k: int) -> int:
        """Confidence gate: show only the items whose hit-now value beats deferring one turn.

        A hit at rank r on turn t contributes 0.30/r - 0.02*t (plus the hit constant); deferring one
        turn and asking "other" is worth 0.30*E[rr_next] - 0.02*(t+1). Rank r is shown when the
        former is at least the latter. Guarded so that a paraphrased or unparsed message, an
        exhausted card, a late turn, or a pre-override turn always yields the full list.
        """
        if not self.confidence_gate or not ranked:
            return top_k
        if (
            not state.hits_count
            or state.card_exhausted
            or turn > GATE_MAX_TURN
            or not state.last_template_matched
            or not (turn == 1 or state.last_new_exact_info or state.last_kind == "boundary")
        ):
            return top_k
        candidates = [item.parent_asin for item in ranked if not item.shown]
        expected = ask_policy.expected_next_reciprocal_rank(self.index, candidates, state.sim_disclosed)
        deferral = RANK_WEIGHT * expected - TURN_WEIGHT * (turn + 1)
        count = 0
        for rank in range(1, top_k + 1):
            if RANK_WEIGHT / rank - TURN_WEIGHT * turn >= deferral:
                count = rank
            else:
                break
        return max(1, count)

    def _message(self, state: SessionState, attribute: str, recommendations: list[str], user_message: str) -> str:
        category = state.category or state.bucket or "options"
        if not recommendations:
            lead = f"I could not find matching {category} yet."
        elif len(recommendations) == 1:
            lead = f"My best {category} pick so far is {self.index.titles.get(recommendations[0], 'this item')[:80]}."
        else:
            lead = f"Here are {len(recommendations)} {category} options matching what you told me."
        question = ask_policy.question_for(attribute)
        llm_text = self._llm(
            state,
            system="You are a concise shopping assistant. Reply in one or two short sentences.",
            user=(
                f"The customer said: {user_message!r}. You are showing {len(recommendations)} products "
                f"for {category!r} and want to ask about their {attribute}. Constraints so far: {state.known!r}."
            ),
            max_tokens=80,
        )
        return llm_text or f"{lead} {question}"

    def _fallback(self, session_id: str, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        try:
            pool: list[str] = []
            if state is not None and state.bucket in self.index.bucket_members:
                pool = [asin for asin in self.index.bucket_members[state.bucket] if asin not in state.shown]
            pool.extend(asin for asin in self.index.popularity_head if asin not in pool)
            recommendations = pool[: max(1, int(top_k))]
        except Exception:
            recommendations = []
        return {
            "message": "Here are some options. " + ask_policy.question_for("other"),
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    # --------------------------------------------------------------------- llm
    def _init_llm(self):
        """Optional OpenAI client (AGENT_USE_LLM=1 and OPENAI_API_KEY set); None means fully offline."""
        if not embedder.flag("AGENT_USE_LLM", "0"):
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _llm(self, state: SessionState, system: str, user: str, max_tokens: int = 120) -> str | None:
        """Single chat completion; accumulates token usage into the session. Never raises."""
        if self.llm is None:
            return None
        try:
            response = self.llm.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0,
            )
        except Exception:
            return None
        usage = getattr(response, "usage", None)
        if usage is not None:
            state.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            state.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        content = response.choices[0].message.content if getattr(response, "choices", None) else None
        return content.strip() if content else None

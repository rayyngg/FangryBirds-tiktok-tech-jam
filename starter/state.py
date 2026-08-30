"""Structured per-session state.

Everything the agent knows about a shopper is kept here as typed fields rather than as a bag of
past messages: the parsed category, the hard and soft constraints disclosed so far, the items
already shown, which attributes were asked and which came back empty, and the bookkeeping needed to
mirror the simulator's own disclosure set (``sim_disclosed``) so that replies can be interpreted.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    category: str | None = None          # category text as it appeared in the opener
    bucket: str | None = None            # canonical coarse-category bucket, when recognised
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    free_text: list[str] = field(default_factory=list)   # unparsed customer text (paraphrase fallback)
    budget: tuple[float, float] | None = None            # (low, high) price window
    sim_disclosed: set[str] = field(default_factory=set)  # mirror of the simulator's `disclosed`
    events: list[tuple] = field(default_factory=list)     # reveal events for the consistency tie-break
    pending_ask: str | None = None       # attribute asked on the previous turn (interprets the next reply)
    asked: list[str] = field(default_factory=list)
    exhausted: set[str] = field(default_factory=set)      # attributes that returned "no additional preference"
    card_exhausted: bool = False                          # "other" returned nothing: nothing left to reveal
    boundary_seen: bool = False
    override_seen: bool = False
    scenario_guess: str = "unknown"      # buying | browsing | override | unknown, inferred from the opener
    shown: dict[str, int] = field(default_factory=dict)   # parent_asin -> turn it was first shown
    turn: int = 0
    last_kind: str = "none"              # parsed kind of the latest message
    last_template_matched: bool = False
    last_new_exact_info: bool = False    # latest message added a constraint that exists in the exact index
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    log: list[dict] = field(default_factory=list)

    @property
    def known(self) -> list[str]:
        """Ordered, de-duplicated constraints the agent treats as true of the target."""
        return list(dict.fromkeys([*self.hard, *self.soft]))

    @property
    def hits_count(self) -> bool:
        """Whether a hit would be scored now (pre-override turns of an override session are ignored)."""
        return not (self.scenario_guess == "override" and not self.override_seen)

    def add_constraint(self, value: str, kind: str = "hard") -> bool:
        """Add a constraint once; promote soft -> hard when the customer restates it as a requirement."""
        value = value.strip()
        if not value:
            return False
        if kind == "hard":
            if value in self.hard:
                return False
            if value in self.soft:
                self.soft.remove(value)
            self.hard.append(value)
            return True
        if value in self.hard or value in self.soft:
            return False
        self.soft.append(value)
        return True

    def debug(self) -> dict:
        return {
            "category": self.category,
            "bucket": self.bucket,
            "hard": list(self.hard),
            "soft": list(self.soft),
            "free_text": list(self.free_text),
            "budget": self.budget,
            "sim_disclosed": sorted(self.sim_disclosed),
            "asked": list(self.asked),
            "exhausted": sorted(self.exhausted),
            "card_exhausted": self.card_exhausted,
            "boundary_seen": self.boundary_seen,
            "override_seen": self.override_seen,
            "scenario_guess": self.scenario_guess,
            "shown": len(self.shown),
            "turns": list(self.log),
        }

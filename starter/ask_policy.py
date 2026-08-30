"""Clarification policy: which attribute to ask the customer about next.

The simulator answers an ask for attribute ``a`` with the first two not-yet-disclosed card
constraints whose class is ``a`` (``other`` matches every class), so at most two constraints arrive
per turn whatever we ask, and ``other`` is the only ask that can never come back empty while
something remains to be disclosed. The production policy is therefore fixed: ask ``other`` every
turn, never return ``null`` (a null ask yields a zero-information reply), keep asking after a
"no preference" reply and through the last turn.

``attribute_values`` implements the question-value estimate used for the ablation
(``AGENT_ASK_POLICY=value``) and for the confidence gate: for each attribute it simulates the reply
of every candidate product under the reveal rule, partitions the candidates by that reply, and
returns the expected reciprocal rank of the target on the next turn under a popularity-weighted
prior. Brand and category are excluded because the simulator never classifies a constraint as
either, so those asks are always wasted.
"""
from __future__ import annotations

from collections import defaultdict

from . import cardspec

ATTRIBUTES = tuple(cardspec.REVEALABLE_ATTRIBUTES)  # material, color, size, style, budget, feature, use_case, other
DEFAULT_ATTRIBUTE = "other"
CANDIDATE_CAP = 500
VALUE_MARGIN = 0.02


def attribute_values(index, candidates: list[str], disclosed, top_k: int = 10, cap: int = CANDIDATE_CAP) -> dict[str, float]:
    """Expected next-turn reciprocal rank for each attribute under the reveal rule.

    ``candidates`` is the current ranked list (best first); the first ``cap`` entries form the
    hypothesis set. The prior is ``rating_number + 1`` (targets are overwhelmingly popular products).
    """
    top = candidates[:cap]
    if not top:
        return {attribute: 0.0 for attribute in ATTRIBUTES}
    prior = {asin: float(index.rating_number.get(asin, 0)) + 1.0 for asin in top}
    total = sum(prior.values())
    disclosed = set(disclosed)
    values: dict[str, float] = {}
    for attribute in ATTRIBUTES:
        cells: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for asin in top:
            cells[index.reply_key(asin, attribute, disclosed)].append(asin)
        value = 0.0
        for members in cells.values():
            members.sort(key=lambda asin: (-prior[asin], asin))
            for rank, asin in enumerate(members[:top_k], 1):
                value += prior[asin] / rank
        values[attribute] = value / total
    return values


def expected_next_reciprocal_rank(index, candidates: list[str], disclosed) -> float:
    """E[1/rank] on the next turn if we ask ``other`` now (the deferral value used by the gate)."""
    if index is None or not candidates:
        return 0.0
    return attribute_values(index, candidates, disclosed)[DEFAULT_ATTRIBUTE]


def choose_attribute(
    mode: str,
    state,
    index,
    candidates: list[str],
    turn: int,
    margin: float = VALUE_MARGIN,
) -> str:
    """Pick the next ask. Never returns None.

    ``mode="other"`` (production) always asks ``other``. ``mode="value"`` asks a specific attribute
    only when its estimated value beats ``other`` by ``margin`` and every top candidate would still
    reveal something for it (so the turn cannot be wasted under the reveal rule).
    """
    if mode != "value" or index is None or not candidates or state.card_exhausted or turn >= 10:
        return DEFAULT_ATTRIBUTE
    values = attribute_values(index, candidates, state.sim_disclosed)
    best = max(ATTRIBUTES, key=lambda attribute: (values[attribute], attribute == DEFAULT_ATTRIBUTE))
    if best == DEFAULT_ATTRIBUTE or best in state.exhausted:
        return DEFAULT_ATTRIBUTE
    if values[best] < values[DEFAULT_ATTRIBUTE] + margin:
        return DEFAULT_ATTRIBUTE
    if any(not index.reply_key(asin, best, state.sim_disclosed) for asin in candidates[:10]):
        return DEFAULT_ATTRIBUTE
    return best


QUESTIONS = {
    "other": "Is there anything else that matters to you — a feature, material, fit, colour or budget?",
    "feature": "Is there a particular feature you need?",
    "material": "Do you have a material preference?",
    "color": "Do you have a colour in mind?",
    "size": "What size or fit are you looking for?",
    "style": "Is there a style or cut you prefer?",
    "budget": "Do you have a budget in mind?",
    "use_case": "What will you mainly use it for?",
}


def question_for(attribute: str) -> str:
    return QUESTIONS.get(attribute, QUESTIONS[DEFAULT_ATTRIBUTE])

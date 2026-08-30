"""Agent-side replica of the public simulator's deterministic rules.

The public evaluator derives the hidden intent card from product metadata with a handful of pure
functions and answers clarification questions with a fixed reveal rule. The agent re-implements those
rules here so it can reason about the customer's messages: which constraint strings a product can
produce, which coarse category label appears in the opener, and which constraints a given ask can
reveal. ``tests/test_cardspec.py`` asserts parity with ``evaluator.local_evaluator`` on every catalog
product; the agent itself never imports the evaluator package at runtime.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

CONSTRAINT_LIMIT = 180
CARD_SIZE = 4  # two hard constraints + two soft preferences
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other",
)
# classify_constraint never yields "brand" or "category", so those asks can never reveal anything.
REVEALABLE_ATTRIBUTES = ("material", "color", "size", "style", "budget", "feature", "use_case", "other")
EXCLUDED_CATEGORY_PARTS = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
DEFAULT_CATEGORY = "clothing item"

_COLOR_WORDS = ("color", "black", "white", "blue", "red", "pink", "green")
_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
_STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
_USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
_BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*\d")
_WHITESPACE_RE = re.compile(r"\s+")


def searchable_text(product: dict) -> str:
    """Concatenate the searchable fields the way the simulator does before regex scans."""
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def flatten_values(value: object) -> list[str]:
    """Render a features list or details dict into candidate constraint strings."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def clean_constraint(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    """Whitespace-collapse, strip punctuation at both ends, then truncate (in that order)."""
    return _WHITESPACE_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def cleaned_candidates(product: dict, limit: int = CONSTRAINT_LIMIT) -> list[str]:
    """Every distinct cleaned constraint string a product can produce, in simulator order."""
    candidates = [*flatten_values(product.get("features")), *flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(clean_constraint(item, limit) for item in candidates if clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [clean_constraint(str(product.get("title") or "product"), limit)]
    return cleaned


def intent_card(product: dict, limit: int = CONSTRAINT_LIMIT) -> dict:
    """The hidden card the simulator builds for a target product."""
    cleaned = cleaned_candidates(product, limit)
    return {
        "target_category": clean_constraint(str(product.get("title") or "product"), limit),
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def card_constraints(product: dict, limit: int = CONSTRAINT_LIMIT) -> list[str]:
    """``hard + soft`` in reveal order, duplicates included (a two-candidate product repeats hard[0])."""
    card = intent_card(product, limit)
    return [*card["hard_constraints"], *card["soft_preferences"]]


def coarse_category(values: Iterable[object]) -> str:
    """The category label used in the opener: last two comma-split parts minus the top-level labels."""
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORY_PARTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else DEFAULT_CATEGORY


def classify_constraint(value: str) -> str:
    """Bucket a constraint string the way the simulator does (substring tests, fixed priority)."""
    lowered = value.lower()
    if "budget" in lowered or _BUDGET_RE.search(lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in _COLOR_WORDS):
        return "color"
    if any(word in lowered for word in _SIZE_WORDS):
        return "size"
    if any(word in lowered for word in _STYLE_WORDS):
        return "style"
    if any(word in lowered for word in _USE_CASE_WORDS):
        return "use_case"
    return "feature"


def normalize_attribute(value: object) -> str | None:
    """How the simulator interprets an ``ask_attribute``: unknown strings become ``other``."""
    if not isinstance(value, str) or not value:
        return None
    return value if value in ALLOWED_ATTRIBUTES else "other"


def reply_key(
    constraints: list[str],
    classes: list[str],
    attribute: str,
    disclosed: Iterable[str],
) -> tuple[str, ...]:
    """The constraints an ask for ``attribute`` reveals for a product with this card.

    Mirrors the simulator's reveal rule: the first two not-yet-disclosed card constraints whose class
    equals the attribute (``other`` matches every class). An empty tuple means the reply would be
    "I don't have an additional preference for ...".
    """
    disclosed_set = set(disclosed)
    matches = [
        constraint
        for constraint, cls in zip(constraints, classes)
        if constraint not in disclosed_set and (attribute == "other" or cls == attribute)
    ]
    return tuple(matches[:2])

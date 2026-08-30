"""Customer-message parser.

The public simulator speaks in a handful of fixed templates (opener, reveal, override, exhausted,
boundary, null-ask). Each is matched by its anchored prefix and the verbatim constraint strings are
recovered — never by "text after the first colon", because constraints themselves contain colons,
periods and semicolons. Anything that matches no template falls through to a tolerant path: an
override is still detected by keywords, a category is still recognised anywhere in the sentence, and
the remaining text is kept as free text for the substring / lexical tiers of the ranker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import cardspec

OPENER = "I'm looking for "
KEY_REQUIREMENT = ". A key requirement is: "
EXPLORING = ", but I'm still exploring."
REVEAL = "For that, what matters is: "
OVERRIDE = "Actually, ignore my earlier preference. What I need is: "
EXHAUSTED = "I don't have an additional preference for "
BOUNDARY = "I don't have a preference for "
NULL_ASK = "Those options are not quite right"

OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|ignore|disregard|change of mind|new requirement|what i need is|forget)\b",
    re.IGNORECASE,
)
LOOKING_RE = re.compile(r"looking\s+for\s+(.+)", re.IGNORECASE | re.DOTALL)
LEAD_IN_RE = re.compile(
    r"^\s*(?:(?:actually|well|hmm|ok|okay|so|also|and)\b[\s,.:;-]*)*"
    r"(?:(?:for that|what matters(?: most)?(?: to me)?|what i (?:need|want|care about(?: most)?)|"
    r"i(?:'d| would)? (?:need|want|prefer|like)|it (?:has to|must|should) (?:be|have)|"
    r"the (?:key )?requirement is|a key requirement is|my (?:preference|requirement) is|"
    r"i(?:'m| am) looking for)\b[\s,.:;-]*)?",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(r"\b(?:no|don't have a(?:n additional)?|do not have a(?:n additional)?)\s+preference\b|use your judgment", re.IGNORECASE)


@dataclass
class ParsedMessage:
    kind: str                       # opener_buying | opener_browsing | opener_override | reveal | override
    template_matched: bool          # | exhausted | boundary | null_ask | freeform
    category: str | None = None
    bucket: str | None = None
    constraints: list[str] = field(default_factory=list)
    attribute: str | None = None
    free_text: list[str] = field(default_factory=list)


def strip_one_period(text: str) -> str:
    """Templates append exactly one period; constraints may legitimately end with '..' after that."""
    return text[:-1] if text.endswith(".") else text


def category_from_text(text: str) -> str | None:
    """Text after 'looking for' up to the first comma or period (tolerant fallback)."""
    match = LOOKING_RE.search(text)
    if not match:
        return None
    candidate = re.split(r"[,.]", match.group(1), maxsplit=1)[0].strip()
    return candidate or None


def split_reveal_payload(payload: str, attribute: str, disclosed, candidates, index) -> tuple[str, ...]:
    """Recover the (at most two) constraints in a reveal payload that may itself contain '; '.

    Every no-split / one-split option is scored by (number of candidate products whose simulated
    reply for ``attribute`` equals the option, number of pieces present in the exact index); the best
    option wins and ties prefer fewer pieces.
    """
    options: list[tuple[str, ...]] = [(payload,)]
    position = payload.find("; ")
    while position != -1:
        options.append((payload[:position], payload[position + 2:]))
        position = payload.find("; ", position + 1)
    if len(options) == 1 or index is None:
        return options[0] if len(options) == 1 else (payload[: payload.find("; ")], payload[payload.find("; ") + 2:])
    disclosed = frozenset(disclosed)
    known = index.constraint_index
    best: tuple[str, ...] = options[0]
    best_score = (-1, -1, 0)
    candidate_list = list(candidates)
    for option in options:
        support = sum(1 for asin in candidate_list if index.reply_key(asin, attribute, disclosed) == option)
        present = sum(1 for piece in option if piece in known)
        score = (support, present, -len(option))
        if score > best_score:
            best, best_score = option, score
    return best


def _opener(message: str, index) -> ParsedMessage:
    rest = message[len(OPENER):]
    bucket = None
    remainder = rest
    if index is not None:
        bucket, remainder = index.match_bucket_prefix(rest)
    if bucket is None:
        category = category_from_text(message)
        if index is not None:
            bucket = index.find_bucket_anywhere(message)
        parsed = ParsedMessage("opener_browsing", False, category=category, bucket=bucket)
        lowered = rest.lower()
        if "key requirement is:" in lowered:
            constraint = strip_one_period(rest[lowered.index("key requirement is:") + len("key requirement is:"):].strip())
            parsed.kind = "opener_buying"
            parsed.constraints = [constraint] if constraint else []
        elif "still exploring" in lowered:
            parsed.kind = "opener_browsing"
        elif ". " in rest:
            parsed.kind = "opener_override"
            parsed.constraints = [rest.split(". ", 1)[1].strip()]
        return parsed
    parsed = ParsedMessage("opener_browsing", True, category=bucket, bucket=bucket)
    if remainder == EXPLORING or remainder == "":
        return parsed
    if remainder.startswith(KEY_REQUIREMENT):
        parsed.kind = "opener_buying"
        constraint = strip_one_period(remainder[len(KEY_REQUIREMENT):])
        parsed.constraints = [constraint] if constraint else []
        return parsed
    if remainder.startswith(". "):
        parsed.kind = "opener_override"
        parsed.constraints = [remainder[2:]]   # verbatim: the override opener appends no period
        return parsed
    parsed.template_matched = False
    parsed.free_text = [remainder.strip(" ,.")]
    return parsed


def parse_message(message: str, state, index) -> ParsedMessage:
    message = message if isinstance(message, str) else str(message or "")
    text = message.strip()
    if text.startswith(OPENER):
        return _opener(text, index)
    if text.startswith(REVEAL):
        payload = strip_one_period(text[len(REVEAL):])
        attribute = state.pending_ask or "other"
        candidates = index.bucket_members.get(state.bucket, []) if (index is not None and state.bucket) else []
        pieces = split_reveal_payload(payload, attribute, state.sim_disclosed, candidates, index)
        return ParsedMessage("reveal", True, constraints=[piece for piece in pieces if piece], attribute=attribute)
    if text.startswith(OVERRIDE):
        constraint = strip_one_period(text[len(OVERRIDE):])
        return ParsedMessage("override", True, constraints=[constraint] if constraint else [])
    if text.startswith(EXHAUSTED):
        attribute = strip_one_period(text[len(EXHAUSTED):]).strip()
        return ParsedMessage("exhausted", True, attribute=cardspec.normalize_attribute(attribute) or "other")
    if text.startswith(BOUNDARY):
        attribute = text[len(BOUNDARY):].split(";", 1)[0].strip()
        return ParsedMessage("boundary", True, attribute=attribute)
    if text.startswith(NULL_ASK):
        return ParsedMessage("null_ask", True)
    return _freeform(text, index)


def _freeform(text: str, index) -> ParsedMessage:
    """Tolerant path for paraphrased messages."""
    if NO_PREFERENCE_RE.search(text) and len(text) < 120:
        return ParsedMessage("boundary", False)
    if OVERRIDE_RE.search(text):
        body = text
        if ": " in body:
            body = body.rsplit(": ", 1)[1]
        elif ". " in body:
            body = body.rsplit(". ", 1)[1]
        body = strip_one_period(LEAD_IN_RE.sub("", body, count=1).strip())
        return ParsedMessage("override", False, constraints=[body] if body else [])
    category = category_from_text(text)
    bucket = index.find_bucket_anywhere(text) if index is not None else None
    if category or bucket:
        parsed = ParsedMessage("opener_browsing", False, category=category, bucket=bucket)
        lowered = text.lower()
        if "key requirement is:" in lowered:
            parsed.kind = "opener_buying"
            remainder = text[lowered.index("key requirement is:") + len("key requirement is:"):]
            parsed.constraints = [strip_one_period(remainder.strip())]
        elif ": " in text:
            # "Hi! I want to buy X. It must have: Y." — a colon after the category introduces a requirement
            constraint = strip_one_period(text.rsplit(": ", 1)[1].strip())
            if constraint:
                parsed.kind = "opener_buying"
                parsed.constraints = [constraint]
        return parsed
    body = strip_one_period(LEAD_IN_RE.sub("", text, count=1).strip())
    pieces = [piece.strip() for piece in body.split("; ") if piece.strip()] if "; " in body else [body]
    return ParsedMessage("freeform", False, free_text=[piece for piece in pieces if piece])


def apply_parsed(state, parsed: ParsedMessage, index) -> None:
    """Fold a parsed message into the session state (constraints, disclosure mirror, events, flags)."""
    state.last_kind = parsed.kind
    state.last_template_matched = parsed.template_matched
    state.last_new_exact_info = False
    known_index = index.constraint_index if index is not None else {}

    def note_exact(value: str) -> None:
        if value in known_index:
            state.last_new_exact_info = True

    if parsed.kind.startswith("opener"):
        if parsed.category and state.category is None:
            state.category = parsed.category
        if parsed.bucket and state.bucket is None:
            state.bucket = parsed.bucket
        if parsed.kind == "opener_buying":
            state.scenario_guess = "buying"
            for value in parsed.constraints:
                if state.add_constraint(value, "hard"):
                    note_exact(value)
                state.sim_disclosed.add(value)
                state.events.append(("card0", value))
                _maybe_budget(state, value)
        elif parsed.kind == "opener_override":
            state.scenario_guess = "override"
            for value in parsed.constraints:
                if state.add_constraint(value, "soft"):
                    note_exact(value)
                state.events.append(("cardlast", value))
                _maybe_budget(state, value)
        else:
            state.scenario_guess = "browsing"
        state.free_text.extend(parsed.free_text)
        return
    if parsed.kind == "reveal":
        before = frozenset(state.sim_disclosed)
        for value in parsed.constraints:
            if state.add_constraint(value, "hard"):
                note_exact(value)
            state.sim_disclosed.add(value)
            _maybe_budget(state, value)
        if parsed.template_matched and parsed.attribute:
            state.events.append(("reply", parsed.attribute, before, tuple(parsed.constraints)))
        return
    if parsed.kind == "override":
        state.override_seen = True
        state.shown.clear()   # pre-override pages are not rejections: the evaluator ignored those turns
        for value in parsed.constraints:
            if state.add_constraint(value, "hard"):
                note_exact(value)
            state.sim_disclosed.add(value)
            if parsed.template_matched:
                state.events.append(("card0", value))
            _maybe_budget(state, value)
        return
    if parsed.kind == "exhausted":
        attribute = parsed.attribute or "other"
        state.exhausted.add(attribute)
        if attribute == "other":
            state.card_exhausted = True
        if parsed.template_matched:
            state.events.append(("noadd", attribute, frozenset(state.sim_disclosed)))
        return
    if parsed.kind == "boundary":
        state.boundary_seen = True
        return
    if parsed.kind == "freeform":
        # Paraphrased replies still carry verbatim constraint strings: pieces that exist in the
        # exact index are treated as disclosed constraints, the rest feeds the substring / lexical tiers.
        for piece in parsed.free_text:
            if piece in known_index or (piece and piece[0].isupper() and piece.lower() in _lowercase_index(index)):
                canonical = piece if piece in known_index else _lowercase_index(index)[piece.lower()]
                if state.add_constraint(canonical, "hard"):
                    state.last_new_exact_info = True
                state.sim_disclosed.add(canonical)
                _maybe_budget(state, canonical)
            else:
                state.free_text.append(piece)
        return
    # null_ask: nothing to learn


def _lowercase_index(index) -> dict:
    """Lower-cased view of the exact constraint index (built lazily, cached on the index)."""
    if index is None:
        return {}
    cached = getattr(index, "_constraint_lower", None)
    if cached is None:
        cached = {}
        for value in index.constraint_index:
            cached.setdefault(value.lower(), value)
        index._constraint_lower = cached
    return cached


def _maybe_budget(state, value: str) -> None:
    lowered = value.lower()
    if "budget" in lowered or "$" in lowered:
        from .retrieval import budget_window

        window = budget_window(value)
        if window is not None:
            state.budget = window

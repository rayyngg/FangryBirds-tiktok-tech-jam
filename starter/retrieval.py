"""Catalog index and candidate ranking.

``CatalogIndex`` is built once at construction, with no network access and no writes:

* an SQLite FTS5 table for lexical (BM25) search, with a token-index fallback when FTS5 is missing,
* per-product structure the agent reasons with: the card constraints a product can produce (see
  ``cardspec``), their classes, the coarse-category bucket, a normalised text for substring checks,
  price and a popularity prior (``log1p(rating_number)``),
* inverted indexes: bucket -> members and exact constraint string -> products, both popularity-sorted.

``rank_candidates`` scores a bounded candidate pool (bucket members, exact-index postings, FTS hits
and the popularity head — never the whole catalog) with a lexicographic key: bucket first, then
exact card matches, reply consistency, substring matches, budget window, then *unseen before seen*
(so paging happens among equally supported candidates and a product from another bucket can never
outrank an in-bucket match just because the latter was shown), and finally a blended popularity /
lexical / semantic score. Tiers, never filters: the list is never empty and a mis-parsed bucket or
constraint only demotes the target instead of hiding it.

Set ``AGENT_DISABLE_FTS=1`` (or run on an SQLite without FTS5) to use the pure-Python token index
instead of FTS5; ranking quality is unchanged because the lexical tier only feeds the candidate pool.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import cardspec

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "still", "exploring", "key", "requirement", "matters", "need",
    "actually", "ignore", "earlier", "preference", "what",
}
EXACT_POSTING_CAP = 1500   # popularity-sorted prefix of an exact-index posting list added to the pool
FTS_POOL_LIMIT = 300       # lexical hits added to the pool each turn
POPULARITY_HEAD = 300      # most popular products, always in the pool
PHRASE_HIT_CAP = 2000
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str, limit: int = 60) -> list[str]:
    seen: dict[str, None] = {}
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) > 1 and lowered not in STOPWORDS:
            seen.setdefault(lowered, None)
        if len(seen) >= limit:
            break
    return list(seen)


def parse_price(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMBER_RE.search(value.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def normalise(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


@dataclass
class Scored:
    parent_asin: str
    key: tuple
    exact: int
    in_bucket: bool
    shown: bool


class CatalogIndex:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.asins: list[str] = []
        self.titles: dict[str, str] = {}
        self.price: dict[str, float | None] = {}
        self.rating_number: dict[str, int] = {}
        self.popularity: dict[str, float] = {}
        self.cards: dict[str, list[str]] = {}
        self.classes: dict[str, list[str]] = {}
        self.bucket_of: dict[str, str] = {}
        self.bucket_members: dict[str, list[str]] = {}
        self.bucket_names_by_length: list[str] = []
        self.bucket_by_lower: dict[str, str] = {}
        self.constraint_index: dict[str, list[str]] = {}
        self.norm_text: dict[str, str] = {}
        self.popularity_head: list[str] = []
        self.max_popularity = 1.0
        self.connection = sqlite3.connect(":memory:")
        self.fts_available = False
        self._token_index: dict[str, set[str]] | None = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        cursor = self.connection.cursor()
        self.fts_available = False
        if os.getenv("AGENT_DISABLE_FTS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            try:
                cursor.execute(
                    "CREATE VIRTUAL TABLE products USING fts5("
                    "parent_asin UNINDEXED, title, categories, features, details, store, description, "
                    "tokenize='unicode61 remove_diacritics 2')"
                )
                self.fts_available = True
            except sqlite3.OperationalError:
                self.fts_available = False
        if not self.fts_available:
            self._token_index = defaultdict(set)

        bucket_members: dict[str, list[str]] = defaultdict(list)
        constraint_index: dict[str, list[str]] = defaultdict(list)
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.asins.append(asin)
                self.titles[asin] = str(product.get("title") or "")
                self.price[asin] = parse_price(product.get("price"))
                rating_number = product.get("rating_number")
                rating_number = int(rating_number) if isinstance(rating_number, (int, float)) else 0
                self.rating_number[asin] = rating_number
                self.popularity[asin] = math.log1p(max(0, rating_number))
                card = cardspec.card_constraints(product)
                self.cards[asin] = card
                self.classes[asin] = [cardspec.classify_constraint(value) for value in card]
                bucket = cardspec.coarse_category(product.get("categories") or [])
                self.bucket_of[asin] = bucket
                bucket_members[bucket].append(asin)
                for value in dict.fromkeys(card):
                    constraint_index[value].append(asin)
                self.norm_text[asin] = normalise(
                    " | ".join(
                        [
                            *cardspec.flatten_values(product.get("features")),
                            *cardspec.flatten_values(product.get("details")),
                            self.titles[asin],
                        ]
                    )
                )
                row = (
                    asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                )
                if self.fts_available:
                    batch.append(row)
                    if len(batch) >= 1000:
                        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                        batch.clear()
                else:
                    # token fallback: title, categories, features, details, store (descriptions are
                    # skipped to keep the index small; they carry the least weight in the FTS tier)
                    for token in set(TOKEN_RE.findall(" ".join(row[1:6]).lower())):
                        self._token_index[token].add(asin)  # type: ignore[index]
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

        popularity = self.popularity
        self.max_popularity = max(popularity.values(), default=1.0) or 1.0
        by_popularity = lambda asin: (-popularity[asin], asin)  # noqa: E731
        self.bucket_members = {name: sorted(members, key=by_popularity) for name, members in bucket_members.items()}
        self.constraint_index = {value: sorted(members, key=by_popularity) for value, members in constraint_index.items()}
        self.bucket_names_by_length = sorted(self.bucket_members, key=lambda name: (-len(name), name))
        self.bucket_by_lower = {name.lower(): name for name in self.bucket_members}
        self.popularity_head = sorted(self.asins, key=by_popularity)[:POPULARITY_HEAD]

    # --------------------------------------------------------------- queries
    def bucket_stats(self) -> dict:
        sizes = sorted(len(members) for members in self.bucket_members.values())
        if not sizes:
            return {"buckets": 0}

        def quantile(fraction: float) -> int:
            return sizes[min(len(sizes) - 1, int(fraction * (len(sizes) - 1)))]

        largest = sorted(self.bucket_members.items(), key=lambda item: -len(item[1]))[:5]
        return {
            "buckets": len(sizes),
            "products": sum(sizes),
            "size_min": sizes[0],
            "size_p25": quantile(0.25),
            "size_median": quantile(0.5),
            "size_p75": quantile(0.75),
            "size_p90": quantile(0.9),
            "size_max": sizes[-1],
            "largest": [(name, len(members)) for name, members in largest],
        }

    def match_bucket_prefix(self, text: str) -> tuple[str | None, str]:
        """Longest known bucket at the start of ``text``; returns (bucket, remainder)."""
        for name in self.bucket_names_by_length:
            if text.startswith(name):
                remainder = text[len(name):]
                if remainder == "" or remainder[:1] in ".,;:!?\n" or remainder[:1].isspace():
                    return name, remainder
        return None, text

    def find_bucket_anywhere(self, text: str) -> str | None:
        """Longest known bucket mentioned anywhere in ``text`` (case-insensitive, word boundaries)."""
        lowered = " " + normalise(text) + " "
        for name in self.bucket_names_by_length:
            needle = name.lower()
            position = lowered.find(needle)
            while position != -1:
                before = lowered[position - 1]
                after = lowered[position + len(needle)]
                if not before.isalnum() and not after.isalnum():
                    return name
                position = lowered.find(needle, position + 1)
        return None

    def reply_key(self, asin: str, attribute: str, disclosed) -> tuple[str, ...]:
        return cardspec.reply_key(self.cards[asin], self.classes[asin], attribute, disclosed)

    def fts_search(self, query_text: str, limit: int = FTS_POOL_LIMIT) -> list[tuple[str, float]]:
        """Best-first BM25 hits for an OR-query over the query terms."""
        unique_terms = terms(query_text)
        if not unique_terms:
            return []
        if not self.fts_available:
            # idf-weighted match over the rarest query terms (bounded work: at most 12 posting lists)
            postings = [(term, self._token_index.get(term)) for term in unique_terms]  # type: ignore[union-attr]
            postings = sorted(((term, members) for term, members in postings if members), key=lambda item: len(item[1]))
            total = max(1, len(self.asins))
            counts: dict[str, float] = defaultdict(float)
            for _term, members in postings[:12]:
                idf = math.log(total / len(members)) + 1.0
                for asin in members:
                    counts[asin] += idf
            ranked = sorted(counts.items(), key=lambda item: (-item[1], -self.popularity[item[0]], item[0]))[:limit]
            return [(asin, -score) for asin, score in ranked]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        weights = ", ".join(str(weight) for weight in BM25_WEIGHTS)
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin, bm25(products, {weights}) AS score "
                "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(asin), float(score)) for asin, score in rows]

    def phrase_hits(self, phrase: str, limit: int = PHRASE_HIT_CAP) -> list[str]:
        """Products whose title/features/details contain the phrase (token sequence)."""
        tokens = [token.lower() for token in TOKEN_RE.findall(phrase)]
        if not tokens:
            return []
        if not self.fts_available:
            hits = None
            for token in tokens:
                members = self._token_index.get(token, set())  # type: ignore[union-attr]
                hits = members if hits is None else hits & members
                if not hits:
                    return []
            return sorted(hits, key=lambda asin: (-self.popularity[asin], asin))[:limit]
        expression = "{title features details} : \"" + " ".join(tokens) + "\""
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT ?", (expression, limit)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(asin) for asin, in rows]


def budget_window(value: str, tolerance: float = 0.05) -> tuple[float, float] | None:
    """Parse ``budget around $N`` (or any price-like number) into an N +/- tolerance window."""
    price = parse_price(value.replace("$", " "))
    if price is None or price <= 0:
        return None
    return (price * (1.0 - tolerance), price * (1.0 + tolerance))


def consistency(index: CatalogIndex, asin: str, events: list[tuple]) -> int:
    """How many observed simulator events this product would have produced (tie-break only)."""
    card = index.cards[asin]
    score = 0
    for event in events:
        kind = event[0]
        if kind == "card0":
            score += card[0] == event[1]
        elif kind == "cardlast":
            score += card[-1] == event[1]
        elif kind == "reply":
            score += index.reply_key(asin, event[1], event[2]) == event[3]
        elif kind == "noadd":
            score += index.reply_key(asin, event[1], event[2]) == ()
    return score


def rank_candidates(
    index: CatalogIndex,
    state,
    query_text: str,
    embed_fn=None,
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    limit: int | None = None,
) -> list[Scored]:
    """Rank a bounded candidate pool for the session; see the module docstring for the key order.

    ``embed_fn(asins) -> {asin: score in [0, 1]}`` is the optional dense tier, called once on the pool.
    """
    known = state.known
    lowered_known = [value.lower() for value in known if value]
    lowered_free = [normalise(value) for value in state.free_text if value.strip()]
    bucket = state.bucket
    shown = state.shown
    budget = state.budget

    pool: set[str] = set()
    if bucket in index.bucket_members:
        pool.update(index.bucket_members[bucket])
    for value in known:
        pool.update(index.constraint_index.get(value, ())[:EXACT_POSTING_CAP])
    for value in lowered_free:
        pool.update(index.phrase_hits(value))
    fts_hits = index.fts_search(query_text) if query_text else []
    pool.update(asin for asin, _score in fts_hits)
    pool.update(index.popularity_head)
    if len(pool) < 10 * 5:
        pool.update(index.asins[: 10 * 5])

    fts_rank: dict[str, float] = {}
    if fts_hits:
        denominator = max(1, len(fts_hits) - 1)
        for position, (asin, _score) in enumerate(fts_hits):
            fts_rank[asin] = 1.0 - position / denominator
    w_pop, w_fts, w_embed = weights
    embed_scores: dict[str, float] = {}
    if embed_fn is not None and w_embed > 0:
        try:
            embed_scores = embed_fn(list(pool)) or {}
        except Exception:
            embed_scores = {}
    max_popularity = index.max_popularity
    events = state.events
    cards = index.cards
    norm_text = index.norm_text
    bucket_of = index.bucket_of
    price = index.price
    popularity = index.popularity

    scored: list[Scored] = []
    for asin in pool:
        card = cards[asin]
        exact = sum(1 for value in known if value in card)
        text = norm_text.get(asin, "")
        substring = sum(1 for value in lowered_known if value in text) + sum(
            1 for value in lowered_free if value in text
        )
        if budget is None:
            budget_ok = 1
        else:
            product_price = price.get(asin)
            budget_ok = 1 if product_price is not None and budget[0] <= product_price <= budget[1] else 0
        blended = (
            w_pop * popularity[asin] / max_popularity
            + w_fts * fts_rank.get(asin, 0.0)
            + w_embed * embed_scores.get(asin, 0.0)
        )
        is_shown = asin in shown
        in_bucket = bucket is not None and bucket_of[asin] == bucket
        key = (
            1 if in_bucket else 0,
            exact,
            consistency(index, asin, events) if events else 0,
            substring,
            budget_ok,
            0 if is_shown else 1,   # paging: among equally supported candidates, unseen ones come first
            blended,
        )
        scored.append(Scored(asin, key, exact, in_bucket, is_shown))
    # Best key first; ties broken by ascending asin for determinism (two stable sorts).
    scored.sort(key=lambda item: item.parent_asin)
    scored.sort(key=lambda item: item.key, reverse=True)
    return scored[:limit] if limit else scored

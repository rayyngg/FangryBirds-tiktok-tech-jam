from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class HashedVectorizer:
    """Small dependency-free sparse embedding for lexical semantic reranking."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> dict[int, float]:
        tokens = [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]
        weights: dict[int, float] = {}
        for feature, weight in self._features(tokens):
            index, sign = self._hash_feature(feature)
            weights[index] = weights.get(index, 0.0) + sign * weight
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm <= 0.0:
            return {}
        return {index: value / norm for index, value in weights.items()}

    def _features(self, tokens: list[str]) -> Iterable[tuple[str, float]]:
        for token in tokens:
            yield f"w:{token}", 1.0
            if len(token) >= 5:
                yield f"p:{token[:5]}", 0.35
                yield f"s:{token[-5:]}", 0.35
            if len(token) >= 6:
                for start in range(len(token) - 2):
                    yield f"c:{token[start:start + 3]}", 0.08
        for first, second in zip(tokens, tokens[1:]):
            yield f"b:{first}_{second}", 1.35

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if (value >> 63) == 0 else -1.0
        return value % self.dimensions, sign


def cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())

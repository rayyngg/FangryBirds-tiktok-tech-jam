"""Parity of the agent-side simulator replica with the organizer's evaluator."""
from __future__ import annotations

import json
import os
import random
import unittest
from pathlib import Path

from evaluator import local_evaluator as ev
from starter import cardspec

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(os.getenv("AGENT_CATALOG_PATH", ROOT / "data" / "catalog.jsonl"))


class CardspecUnitTest(unittest.TestCase):
    def test_clean_constraint_strips_then_truncates(self) -> None:
        self.assertEqual(cardspec.clean_constraint("  - 100% Cotton.  "), "100% Cotton")
        long = "A" * 179 + "." + "BBBB"
        self.assertEqual(cardspec.clean_constraint(long), "A" * 179 + ".")  # truncation keeps a trailing period

    def test_coarse_category(self) -> None:
        self.assertEqual(cardspec.coarse_category(["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings", "Hoop"]), "Earrings Hoop")
        self.assertEqual(cardspec.coarse_category(["Clothing, Shoes & Jewelry", "Women"]), "Shoes & Jewelry Women")
        self.assertEqual(cardspec.coarse_category([]), "clothing item")

    def test_classify_is_substring_based(self) -> None:
        self.assertEqual(cardspec.classify_constraint("covered buttons"), "color")   # 'red' in 'covered'
        self.assertEqual(cardspec.classify_constraint("outfit"), "style")            # 'fit' in 'outfit'
        self.assertEqual(cardspec.classify_constraint("under 5 dollars"), "budget")
        self.assertEqual(cardspec.classify_constraint("Rubber sole"), "feature")

    def test_reply_key_reveal_rule(self) -> None:
        card = ["cotton", "color: blue", "Imported", "Zipper closure"]
        classes = [cardspec.classify_constraint(value) for value in card]
        self.assertEqual(cardspec.reply_key(card, classes, "other", set()), ("cotton", "color: blue"))
        self.assertEqual(cardspec.reply_key(card, classes, "other", {"cotton"}), ("color: blue", "Imported"))
        self.assertEqual(cardspec.reply_key(card, classes, "feature", set()), ("Imported", "Zipper closure"))
        self.assertEqual(cardspec.reply_key(card, classes, "size", set()), ())

    def test_normalize_attribute(self) -> None:
        self.assertEqual(cardspec.normalize_attribute("zzz"), "other")
        self.assertEqual(cardspec.normalize_attribute("material"), "material")
        self.assertIsNone(cardspec.normalize_attribute(None))
        self.assertIsNone(cardspec.normalize_attribute(""))


@unittest.skipUnless(CATALOG.exists(), "catalog not available")
class CardspecParityTest(unittest.TestCase):
    """Every product in the frozen catalog must produce identical cards, buckets and classes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.products = []
        with CATALOG.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cls.products.append(json.loads(line))

    def test_intent_card_and_bucket_parity(self) -> None:
        for product in self.products:
            categories = [str(value) for value in product.get("categories") or []]
            self.assertEqual(cardspec.intent_card(product), ev.intent_card(product), product["parent_asin"])
            self.assertEqual(cardspec.coarse_category(categories), ev.coarse_category(categories), product["parent_asin"])
            for value in cardspec.card_constraints(product):
                self.assertEqual(cardspec.classify_constraint(value), ev.classify_constraint(value), value)

    def test_reply_key_matches_customer_reply(self) -> None:
        rng = random.Random(7)
        for product in rng.sample(self.products, 300):
            card = ev.intent_card(product)
            constraints = cardspec.card_constraints(product)
            classes = [cardspec.classify_constraint(value) for value in constraints]
            sample = {"scenario_type": "buying", "intent_card": card}
            for attribute in cardspec.ALLOWED_ATTRIBUTES:
                disclosed: set[str] = set()
                for _ in range(3):
                    expected = cardspec.reply_key(constraints, classes, cardspec.normalize_attribute(attribute), disclosed)
                    message, _ = ev.customer_reply(sample, attribute, disclosed, False)
                    if expected:
                        self.assertEqual(message, "For that, what matters is: " + "; ".join(expected) + ".")
                    else:
                        self.assertEqual(message, f"I don't have an additional preference for {cardspec.normalize_attribute(attribute)}.")


if __name__ == "__main__":
    unittest.main()

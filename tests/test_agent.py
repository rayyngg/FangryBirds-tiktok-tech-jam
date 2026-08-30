"""Behavioural tests for the agent on a small synthetic catalog (embeddings and LLM off)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from evaluator import local_evaluator as ev
from starter import cardspec, parsing, retrieval
from starter.agent import Agent

OFFLINE_ENV = {"AGENT_USE_EMBEDDINGS": "0", "AGENT_USE_LLM": "0", "AGENT_CONFIDENCE_GATE": "0"}

LONG_FEATURE = "A" * 179 + "." + "BBBB"   # cleans to 179 A's plus a period (truncation keeps it)

CATALOG = [
    {"parent_asin": "D1", "title": "Blue Velvet Wrap Dress", "features": ["95% Polyester, 5% Spandex", "Wrap closure", "Machine Wash"],
     "details": {"Department": "Womens"}, "description": ["party dress"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses", "Casual"],
     "store": "Guberry", "average_rating": 4.4, "rating_number": 1200, "price": 49.99},
    {"parent_asin": "D2", "title": "Red Linen Summer Dress", "features": ["100% Linen", "Pull On closure", "Hand Wash Only"],
     "details": {"Department": "Womens"}, "description": ["summer dress"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses", "Casual"],
     "store": "Sunny", "average_rating": 4.1, "rating_number": 800, "price": 39.0},
    {"parent_asin": "D3", "title": "Floral Maxi Dress", "features": ["Solids: 100% Cotton; Heathers: 50% Cotton, 50% Polyester", "Pull On closure"],
     "details": {"Department": "Womens"}, "description": ["maxi"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses", "Casual"],
     "store": "Sunny", "average_rating": 4.0, "rating_number": 300, "price": 29.0},
    {"parent_asin": "D4", "title": "Plain Shift Dress", "features": [LONG_FEATURE, "Zipper closure"],
     "details": {"Department": "Womens"}, "description": ["shift"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses", "Casual"],
     "store": "Sunny", "average_rating": 3.9, "rating_number": 50, "price": None},
    {"parent_asin": "D5", "title": "Knit Sweater Dress", "features": ["100% Acrylic", "Pull On closure"],
     "details": {"Department": "Womens"}, "description": ["knit"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses", "Casual"],
     "store": "Sunny", "average_rating": 4.2, "rating_number": 20, "price": 35.0},
    {"parent_asin": "B1", "title": "Black Leather Combat Boot", "features": ["100% Leather", "Rubber sole", "Lace-up"],
     "details": {"Department": "Mens"}, "description": ["boot"], "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Boots"],
     "store": "Boots Co", "average_rating": 4.5, "rating_number": 5000, "price": 89.0},
    {"parent_asin": "B2", "title": "Brown Suede Chelsea Boot", "features": ["Suede", "Rubber sole", "Elastic panels"],
     "details": {"Department": "Mens"}, "description": ["boot"], "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Boots"],
     "store": "Boots Co", "average_rating": 4.3, "rating_number": 700, "price": 79.0},
    {"parent_asin": "B3", "title": "Hiking Boot Waterproof", "features": ["Waterproof membrane", "Rubber sole"],
     "details": {"Department": "Mens"}, "description": ["hiking"], "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Boots"],
     "store": "Trail", "average_rating": 4.6, "rating_number": 9000, "price": 120.0},
    {"parent_asin": "N1", "title": "Celtic Knot Pendant Necklace", "features": ["Material:alloy", "Triple Moon Symbol", "Comes in a gift box"],
     "details": {"Department": "Womens"}, "description": ["pendant"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
     "store": "Qian", "average_rating": 4.2, "rating_number": 2500, "price": 12.99},
    {"parent_asin": "N2", "title": "Silver Chain Necklace", "features": ["925 Sterling Silver", "Lobster clasp"],
     "details": {"Department": "Womens"}, "description": ["chain"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
     "store": "Silverline", "average_rating": 4.7, "rating_number": 15000, "price": 25.0},
    {"parent_asin": "N3", "title": "Handmade Bead Necklace", "features": ["Handmade"],
     "details": {}, "description": ["beads"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
     "store": "Craft", "average_rating": 4.0, "rating_number": 10, "price": 19.99},
    {"parent_asin": "N4", "title": "Gold Plated Necklace", "features": ["18K gold plated", "Lobster clasp"],
     "details": {"Department": "Womens"}, "description": ["gold"], "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Necklaces"],
     "store": "Goldie", "average_rating": 4.1, "rating_number": 400, "price": 30.0},
]


class AgentTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls.tmp.name) / "catalog.jsonl"
        cls.catalog_path.write_text("".join(json.dumps(row) + "\n" for row in CATALOG), encoding="utf-8")
        cls.catalog_ids, cls.categories, cls.products = ev.catalog_index(cls.catalog_path)
        with mock.patch.dict(os.environ, OFFLINE_ENV):
            cls.agent = Agent(cls.catalog_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def new_session(self, name: str = "s"):
        session_id = f"{name}-{time.time_ns()}"
        self.agent.reset(session_id, {"summary": "x", "preference_tags": ["fit"]})
        return session_id

    def state(self, session_id: str):
        return self.agent._sessions[session_id]


class ParserTest(AgentTestBase):
    def parse(self, text, session_id=None):
        session_id = session_id or self.new_session()
        return parsing.parse_message(text, self.state(session_id), self.agent.index), session_id

    def test_buying_opener_with_colon_constraint(self) -> None:
        parsed, _ = self.parse("I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.")
        self.assertEqual((parsed.kind, parsed.bucket, parsed.constraints, parsed.template_matched),
                         ("opener_buying", "Jewelry Necklaces", ["Material:alloy"], True))

    def test_browsing_opener(self) -> None:
        parsed, _ = self.parse("I'm looking for Dresses Casual, but I'm still exploring.")
        self.assertEqual((parsed.kind, parsed.bucket, parsed.constraints), ("opener_browsing", "Dresses Casual", []))

    def test_override_opener_is_verbatim(self) -> None:
        parsed, _ = self.parse("I'm looking for Dresses Casual. " + "A" * 179 + ".")
        self.assertEqual(parsed.kind, "opener_override")
        self.assertEqual(parsed.constraints, ["A" * 179 + "."])   # no period stripped: the template appends none

    def test_unknown_category_falls_back(self) -> None:
        parsed, _ = self.parse("I'm looking for something for the beach, but I'm still exploring.")
        self.assertEqual(parsed.kind, "opener_browsing")
        self.assertIsNone(parsed.bucket)
        self.assertEqual(parsed.category, "something for the beach")
        self.assertFalse(parsed.template_matched)

    def test_reveal_strips_exactly_one_period(self) -> None:
        parsed, _ = self.parse('For that, what matters is: Shaft measures approximately 8.37" from arch.')
        self.assertEqual(parsed.constraints, ['Shaft measures approximately 8.37" from arch'])
        parsed, _ = self.parse("For that, what matters is: " + "A" * 179 + "..")
        self.assertEqual(parsed.constraints, ["A" * 179 + "."])

    def test_reveal_split_respects_semicolon_inside_constraint(self) -> None:
        _, session_id = self.parse("I'm looking for Dresses Casual, but I'm still exploring.")
        state = self.state(session_id)
        parsing.apply_parsed(state, parsing.parse_message("I'm looking for Dresses Casual, but I'm still exploring.", state, self.agent.index), self.agent.index)
        state.pending_ask = "other"
        parsed = parsing.parse_message(
            "For that, what matters is: cotton; Solids: 100% Cotton; Heathers: 50% Cotton, 50% Polyester.", state, self.agent.index
        )
        self.assertEqual(parsed.constraints, ["cotton", "Solids: 100% Cotton; Heathers: 50% Cotton, 50% Polyester"])

    def test_override_message(self) -> None:
        parsed, _ = self.parse("Actually, ignore my earlier preference. What I need is: Solids: 100% Cotton; Heathers: 50% Cotton, 50% Polyester.")
        self.assertEqual((parsed.kind, parsed.constraints), ("override", ["Solids: 100% Cotton; Heathers: 50% Cotton, 50% Polyester"]))

    def test_exhausted_boundary_null(self) -> None:
        self.assertEqual(self.parse("I don't have an additional preference for other.")[0].kind, "exhausted")
        self.assertEqual(self.parse("I don't have an additional preference for zzz.")[0].attribute, "other")
        parsed, _ = self.parse("I don't have a preference for other; please use your judgment.")
        self.assertEqual((parsed.kind, parsed.attribute), ("boundary", "other"))
        self.assertEqual(self.parse("Those options are not quite right yet. Ask me about one specific attribute.")[0].kind, "null_ask")

    def test_paraphrased_reply_becomes_free_text_and_keyword_override(self) -> None:
        parsed, _ = self.parse("What I care about most is Rubber sole; Lace-up")
        self.assertEqual(parsed.kind, "freeform")
        self.assertTrue(parsed.free_text)
        parsed, _ = self.parse("Forget what I said before, I really need: Rubber sole.")
        self.assertEqual((parsed.kind, parsed.constraints), ("override", ["Rubber sole"]))

    def test_budget_window(self) -> None:
        self.assertEqual(retrieval.budget_window("budget around $20"), (19.0, 21.0))
        self.assertIsNone(retrieval.budget_window("budget around $—"))


class StateMirrorTest(AgentTestBase):
    """The agent's disclosure mirror must equal the evaluator's `disclosed` set after every message."""

    def drive(self, scenario: str, target: str, turns: int = 6):
        sample = {"sample_id": f"t-{scenario}", "scenario_type": scenario, "user_profile": {}, "ground_truth": {"parent_asin": target}}
        card, behavior = ev.materialize_hidden_fields(sample, self.products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario != "intent_override"
        message = ev.initial_message(effective, ev.coarse_category(self.categories[target]), disclosed)
        session_id = self.new_session(scenario)
        state = self.state(session_id)
        for turn in range(1, turns + 1):
            response = self.agent.respond(session_id, message, turn, 10)
            self.assertEqual(state.sim_disclosed, disclosed, f"{scenario} turn {turn}")
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                disclosed.add(str(override["new_value"]))
                message = str(override["message"])
            else:
                message, boundary_used = ev.customer_reply(effective, response["ask_attribute"], disclosed, boundary_used)
        return state, card

    def test_all_scenarios_mirror_disclosed(self) -> None:
        for scenario, target in (("buying", "N1"), ("browsing", "D3"), ("boundary", "B1"), ("intent_override", "D1")):
            state, card = self.drive(scenario, target)
            expected = set(card["hard_constraints"]) | set(card["soft_preferences"])
            self.assertEqual(set(state.known), expected, scenario)

    def test_override_keeps_old_constraint_and_resets_shown(self) -> None:
        state, card = self.drive("intent_override", "D1")
        self.assertTrue(state.override_seen)
        self.assertIn(card["soft_preferences"][-1], state.known)     # "ignored" preference is still true of the target
        self.assertIn(card["hard_constraints"][0], state.hard)
        override_turn = next(entry for entry in state.log if entry.get("kind") == "override")["turn"]
        self.assertTrue(all(turn >= override_turn for turn in state.shown.values()))

    def test_full_card_disclosed_by_turn_three(self) -> None:
        for scenario, target in (("buying", "D2"), ("browsing", "N4")):
            sample = {"sample_id": "x", "scenario_type": scenario, "user_profile": {}, "ground_truth": {"parent_asin": target}}
            card, behavior = ev.materialize_hidden_fields(sample, self.products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            message = ev.initial_message(effective, ev.coarse_category(self.categories[target]), disclosed)
            session_id = self.new_session()
            messages = [message]
            for turn in (1, 2):
                response = self.agent.respond(session_id, message, turn, 10)
                message, _ = ev.customer_reply(effective, response["ask_attribute"], disclosed, False)
                messages.append(message)
            for value in card["hard_constraints"] + card["soft_preferences"]:
                self.assertTrue(any(value in text for text in messages), f"{scenario}: {value} not disclosed by turn 3")


class RespondContractTest(AgentTestBase):
    def assert_valid(self, response: dict, top_k: int = 10) -> None:
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], cardspec.ALLOWED_ATTRIBUTES)
        asins = [item["parent_asin"] for item in response["recommendations"]]
        self.assertTrue(all(set(item) == {"parent_asin"} for item in response["recommendations"]))
        self.assertEqual(len(asins), len(set(asins)))
        self.assertLessEqual(len(asins), top_k)
        self.assertTrue(all(asin in self.catalog_ids for asin in asins))
        self.assertTrue(all(isinstance(value, int) and value >= 0 for value in response["usage"].values()))

    def test_never_raises(self) -> None:
        session_id = self.new_session()
        self.assert_valid(self.agent.respond("unknown-session", "hello", 1, 10))
        self.assert_valid(self.agent.respond(session_id, None, 2, 10))
        self.assert_valid(self.agent.respond(session_id, "", 99, 10))
        self.assert_valid(self.agent.respond(session_id, "x" * 5000, 3, 0), top_k=1)
        with mock.patch.object(retrieval, "rank_candidates", side_effect=RuntimeError("boom")):
            response = self.agent.respond(session_id, "I'm looking for Shoes Boots, but I'm still exploring.", 4, 10)
        self.assert_valid(response)
        self.assertTrue(response["recommendations"])

    def test_ask_never_null_and_full_lists(self) -> None:
        session_id = self.new_session()
        for turn, message in enumerate([
            "I'm looking for Shoes Boots, but I'm still exploring.",
            "I don't have a preference for other; please use your judgment.",
            "For that, what matters is: Suede; Rubber sole.",
            "I don't have an additional preference for other.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
        ] + ["I don't have an additional preference for other."] * 5, 1):
            response = self.agent.respond(session_id, message, turn, 10)
            self.assert_valid(response)
            self.assertEqual(response["ask_attribute"], "other")
            self.assertEqual(len(response["recommendations"]), 10)

    def test_shown_items_are_demoted_until_override(self) -> None:
        session_id = self.new_session()
        first = [item["parent_asin"] for item in self.agent.respond(session_id, "I'm looking for Shoes Boots, but I'm still exploring.", 1, 2)["recommendations"]]
        second = [item["parent_asin"] for item in self.agent.respond(session_id, "I don't have an additional preference for other.", 2, 2)["recommendations"]]
        self.assertFalse(set(first) & set(second))
        self.agent.respond(session_id, "Actually, ignore my earlier preference. What I need is: Rubber sole.", 3, 2)
        self.assertEqual(set(self.state(session_id).shown), set(self.state(session_id).shown) & set(self.catalog_ids))
        self.assertTrue(all(turn >= 3 for turn in self.state(session_id).shown.values()))

    def test_bucket_and_exact_matching_rank_target_first(self) -> None:
        session_id = self.new_session()
        response = self.agent.respond(session_id, "I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "N1")
        session_id = self.new_session()
        # top_k=1 on turn 1 so the tiny bucket is not entirely "shown" (which would demote it on turn 2)
        first = self.agent.respond(session_id, "I'm looking for Dresses Casual, but I'm still exploring.", 1, 1)
        self.assertEqual(first["recommendations"][0]["parent_asin"], "D1")   # most popular dress
        response = self.agent.respond(session_id, "For that, what matters is: 100% Acrylic; Pull On closure.", 2, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "D5")

    def test_budget_constraint_prefers_price_window(self) -> None:
        session_id = self.new_session()
        self.agent.respond(session_id, "I'm looking for Jewelry Necklaces, but I'm still exploring.", 1, 1)
        response = self.agent.respond(session_id, "For that, what matters is: Handmade; budget around $19.99.", 2, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "N3")
        self.assertEqual(self.state(session_id).budget, (19.99 * 0.95, 19.99 * 1.05))

    def test_latency(self) -> None:
        session_id = self.new_session()
        started = time.perf_counter()
        for turn in range(1, 11):
            self.agent.respond(session_id, "I'm looking for Dresses Casual, but I'm still exploring." if turn == 1 else "For that, what matters is: Pull On closure.", turn, 10)
        self.assertLess((time.perf_counter() - started) / 10, 1.0)


class ConfigurationTest(AgentTestBase):
    def test_confidence_gate_never_empty_and_respects_guards(self) -> None:
        with mock.patch.dict(os.environ, {**OFFLINE_ENV, "AGENT_CONFIDENCE_GATE": "1"}):
            gated = Agent(self.catalog_path)
        gated.reset("g", {})
        first = gated.respond("g", "I'm looking for Dresses Casual, but I'm still exploring.", 1, 10)["recommendations"]
        self.assertGreaterEqual(len(first), 1)
        paraphrased = gated.respond("g", "I like flowy dresses honestly", 2, 10)["recommendations"]
        self.assertEqual(len(paraphrased), 10)    # unparsed message: no truncation
        gated.respond("g", "I don't have an additional preference for other.", 3, 10)
        late = gated.respond("g", "For that, what matters is: Pull On closure.", 10, 10)["recommendations"]
        self.assertEqual(len(late), 10)

    def test_offline_construction_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            env = {**OFFLINE_ENV, "AGENT_USE_EMBEDDINGS": "1", "HF_HOME": empty, "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "AGENT_MODEL_PATH": str(Path(empty) / "missing")}
            with mock.patch.dict(os.environ, env):
                agent = Agent(self.catalog_path)
        self.assertIsNone(agent.embedding_model)
        agent.reset("o", {})
        self.assertEqual(len(agent.respond("o", "I'm looking for Shoes Boots, but I'm still exploring.", 1, 3)["recommendations"]), 3)

    def test_catalog_path_env_override(self) -> None:
        with mock.patch.dict(os.environ, {**OFFLINE_ENV, "AGENT_CATALOG_PATH": str(self.catalog_path)}):
            agent = Agent()
        self.assertEqual(agent.catalog_path, self.catalog_path)


if __name__ == "__main__":
    unittest.main()

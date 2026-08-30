"""Slow regression on the public set (RUN_SLOW=1): guards the headline score."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from evaluator import local_evaluator as ev
from starter.agent import Agent

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(os.getenv("AGENT_CATALOG_PATH", ROOT / "data" / "catalog.jsonl"))
DATASET = ROOT / "data" / "public_set.jsonl"


@unittest.skipUnless(os.getenv("RUN_SLOW") == "1" and CATALOG.exists() and DATASET.exists(), "set RUN_SLOW=1 with data present")
class PublicRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = ev.load_jsonl(DATASET)
        cls.catalog_ids, cls.categories, cls.products = ev.catalog_index(CATALOG)

    def score(self, env: dict) -> float:
        with mock.patch.dict(os.environ, {"AGENT_USE_EMBEDDINGS": "0", "AGENT_USE_LLM": "0", **env}):
            agent = Agent(CATALOG)
        result = ev.evaluate(agent, self.samples, self.catalog_ids, self.categories, self.products)
        return result["recommended_technical_score"]

    def test_full_lists(self) -> None:
        self.assertGreaterEqual(self.score({"AGENT_CONFIDENCE_GATE": "0"}), 0.90)

    def test_confidence_gate(self) -> None:
        self.assertGreaterEqual(self.score({"AGENT_CONFIDENCE_GATE": "1"}), 0.96)


if __name__ == "__main__":
    unittest.main()

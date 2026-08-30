#!/usr/bin/env python3
"""Print one multi-turn session as a transcript (customer message, agent reply, ask, top items).

    AGENT_USE_LLM=0 python3 scripts/demo_session.py --sample public_0006
    python3 scripts/demo_session.py --scenario intent_override --index 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as ev  # noqa: E402


class Transcript:
    def __init__(self, agent, products: dict) -> None:
        self.agent = agent
        self.products = products
        self.turns: list[dict] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.turns = []
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.turns.append({"turn": turn, "customer": user_message, "response": response})
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Show one evaluated session as a transcript")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--sample", default=None, help="sample_id, e.g. public_0006")
    parser.add_argument("--scenario", default="browsing")
    parser.add_argument("--index", type=int, default=0, help="n-th session of the scenario when --sample is not given")
    parser.add_argument("--top", type=int, default=3, help="how many recommended titles to print per turn")
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.sample:
        sample = next(item for item in samples if item["sample_id"] == args.sample)
    else:
        sample = [item for item in samples if item["scenario_type"] == args.scenario][args.index]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    from starter.agent import Agent

    transcript = Transcript(Agent(args.catalog), products)
    result = ev.evaluate(transcript, [sample], catalog_ids, categories, products)
    target = sample["ground_truth"]["parent_asin"]
    session = result["sessions"][0]

    print(f"session {sample['sample_id']} [{sample['scenario_type']}]  target={target} :: {products[target]['title'][:90]!r}")
    print(f"profile: {sample['user_profile'].get('summary')}")
    for entry in transcript.turns:
        response = entry["response"]
        recs = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"\n--- turn {entry['turn']}")
        print(f"customer: {entry['customer']}")
        print(f"agent   : {response.get('message')}")
        print(f"ask     : {response.get('ask_attribute')}   shown: {len(recs)}")
        for rank, asin in enumerate(recs[: args.top], 1):
            marker = "  <-- target" if asin == target else ""
            print(f"   {rank:2d}. {asin} {products[asin]['title'][:80]!r}{marker}")
        if target in recs and recs.index(target) >= args.top:
            print(f"   ... target at rank {recs.index(target) + 1}")
    print(f"\nresult: hit={session['hit']} first_hit_turn={session['first_hit_turn']} best_rank={session['best_rank']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Perturbation study: score the (unchanged) agent against *modified* simulators.

The private set may ship intent cards produced differently, paraphrase the simulator's messages, or
move the override turn. This script monkeypatches the organizer module's functions at runtime (the
file on disk is never touched) so the customer side changes while the agent keeps its replica of the
public rules, and reports how much of the score survives each change. Results are written to
``results/<sha>-perturb.json`` and a markdown table for the README.

Usage:
    AGENT_USE_LLM=0 python3 scripts/perturb_eval.py                # gate off and on
    python3 scripts/perturb_eval.py --configs base,gate,value --only none,lowercased
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as ev  # noqa: E402
from scripts.run_eval import git_meta  # noqa: E402

CONFIGS = {
    "base": {"AGENT_CONFIDENCE_GATE": "0"},
    "gate": {"AGENT_CONFIDENCE_GATE": "1"},
    "value": {"AGENT_CONFIDENCE_GATE": "0", "AGENT_ASK_POLICY": "value"},
    "gate+value": {"AGENT_CONFIDENCE_GATE": "1", "AGENT_ASK_POLICY": "value"},
}

ORIGINAL = {
    name: getattr(ev, name) for name in ("intent_card", "initial_message", "customer_reply", "behavior_for", "coarse_category")
}


# ----------------------------------------------------------------- perturbed simulator pieces
def card_no_material_color(product: dict, limit: int = 180) -> dict:
    title = ev._clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*ev._flatten_values(product.get("features")), *ev._flatten_values(product.get("details"))]
    cleaned = list(dict.fromkeys(ev._clean_constraint(item, limit) for item in candidates if ev._clean_constraint(item, limit))) or [title]
    return {"target_category": title, "hard_constraints": cleaned[:2], "soft_preferences": cleaned[2:4] or cleaned[:1]}


def card_limit_120(product: dict, limit: int = 120) -> dict:
    return ORIGINAL["intent_card"](product, limit)


def card_swapped(product: dict, limit: int = 180) -> dict:
    card = ORIGINAL["intent_card"](product, limit)
    return {"target_category": card["target_category"], "hard_constraints": card["soft_preferences"], "soft_preferences": card["hard_constraints"]}


def card_lowercased(product: dict, limit: int = 180) -> dict:
    card = ORIGINAL["intent_card"](product, limit)
    return {
        "target_category": card["target_category"],
        "hard_constraints": [value.lower() for value in card["hard_constraints"]],
        "soft_preferences": [value.lower() for value in card["soft_preferences"]],
    }


def reply_paraphrased(sample, ask, disclosed, boundary_used):
    message, boundary_used = ORIGINAL["customer_reply"](sample, ask, disclosed, boundary_used)
    prefix = "For that, what matters is: "
    if message.startswith(prefix):
        message = "What I care about most is " + message[len(prefix):]
    elif message.startswith("I don't have an additional preference for "):
        message = "Nothing more comes to mind about " + message[len("I don't have an additional preference for "):]
    return message, boundary_used


def opener_paraphrased(sample, category, disclosed):
    message = ORIGINAL["initial_message"](sample, category, disclosed)
    if message.startswith("I'm looking for ") and ". A key requirement is: " in message:
        constraint = message.split(". A key requirement is: ", 1)[1]
        return f"Hi! I want to buy {category}. It must have: {constraint}"
    if message.endswith(", but I'm still exploring."):
        return f"Hi! Can you help me find {category}? I'm just browsing for now."
    return message.replace("I'm looking for ", "Hi, I need ", 1)


def behavior_override_paraphrased(scenario, card, rng):
    behavior = ORIGINAL["behavior_for"](scenario, card, rng)
    if "override" in behavior:
        behavior["override"]["message"] = f"Forget what I said before, I really need: {behavior['override']['new_value']}."
    return behavior


def behavior_override_turn_2_to_6(scenario, card, rng):
    behavior = ORIGINAL["behavior_for"](scenario, card, rng)
    if "override" in behavior:
        behavior["override"]["turn"] = random.Random(f"turn:{card['target_category']}").choice([2, 3, 4, 5, 6])
    return behavior


PERTURBATIONS = {
    "none": {},
    "no_material_color": {"intent_card": card_no_material_color},
    "limit_120": {"intent_card": card_limit_120},
    "hard_soft_swapped": {"intent_card": card_swapped},
    "lowercased": {"intent_card": card_lowercased},
    "reply_paraphrased": {"customer_reply": reply_paraphrased},
    "opener_paraphrased": {"initial_message": opener_paraphrased},
    "override_paraphrased": {"behavior_for": behavior_override_paraphrased},
    "override_turn_2_to_6": {"behavior_for": behavior_override_turn_2_to_6},
    "all_paraphrased": {
        "customer_reply": reply_paraphrased,
        "initial_message": opener_paraphrased,
        "behavior_for": behavior_override_paraphrased,
    },
}


@contextlib.contextmanager
def patched(overrides: dict):
    for name, value in overrides.items():
        setattr(ev, name, value)
    try:
        yield
    finally:
        for name in overrides:
            setattr(ev, name, ORIGINAL[name])


def build_agent(env: dict, catalog: str):
    from starter.agent import Agent

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update({"AGENT_USE_LLM": "0", **env})
    try:
        return Agent(catalog)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize(result: dict) -> dict:
    return {
        "score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "scenario_metrics": {
            name: {key: metrics[key] for key in ("hit_rate_at_10", "mrr", "mttc")}
            for name, metrics in result["scenario_metrics"].items()
        },
    }


def markdown_table(table: dict, configs: list[str]) -> str:
    lines = ["| perturbation | " + " | ".join(f"{name} score (HR / MRR / MTTC)" for name in configs) + " |",
             "|---|" + "---|" * len(configs)]
    for perturbation, row in table.items():
        cells = []
        for name in configs:
            entry = row.get(name)
            cells.append(f"{entry['score']:.4f} ({entry['hit_rate_at_10']:.3f} / {entry['mrr']:.3f} / {entry['mttc']:.2f})" if entry else "-")
        lines.append(f"| {perturbation} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the agent against perturbed simulators")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--configs", default="base,gate", help="comma-separated subset of " + ",".join(CONFIGS))
    parser.add_argument("--only", default=None, help="comma-separated subset of perturbations")
    parser.add_argument("--label", default="perturb")
    args = parser.parse_args()

    configs = [name for name in args.configs.split(",") if name in CONFIGS]
    perturbations = list(PERTURBATIONS)
    if args.only:
        perturbations = [name for name in args.only.split(",") if name in PERTURBATIONS]
    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agents = {name: build_agent(CONFIGS[name], args.catalog) for name in configs}

    table: dict[str, dict[str, dict]] = {}
    started = time.perf_counter()
    for perturbation in perturbations:
        table[perturbation] = {}
        with patched(PERTURBATIONS[perturbation]):
            for name in configs:
                result = ev.evaluate(agents[name], samples, catalog_ids, categories, products)
                table[perturbation][name] = summarize(result)
                print(f"{perturbation:22s} {name:10s} score={result['recommended_technical_score']:.4f} "
                      f"HR={result['hit_rate_at_10']:.3f} MRR={result['mrr']:.3f} MTTC={result['mttc']:.2f}", flush=True)

    meta = git_meta()
    sha = meta["sha"] + ("-dirty" if meta["dirty"] else "")
    output = {
        "meta": {"git": meta, "configs": {name: CONFIGS[name] for name in configs}, "perturbations": perturbations,
                 "wall_time_s": round(time.perf_counter() - started, 1)},
        "table": table,
    }
    path = ROOT / "results" / f"{sha}-{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    markdown = markdown_table(table, configs)
    path.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()

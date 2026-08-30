#!/usr/bin/env python3
"""Evaluation wrapper around the organizer's local evaluator.

The organizer's module (``evaluator/local_evaluator.py``) is imported unchanged; this script only
adds instrumentation around it:

* ``--limit N`` / ``--scenario`` to run a subset of sessions,
* a recording proxy that captures every turn (customer message, response, latency),
* per-difficulty and scenario x difficulty metrics,
* a miss log (target title / coarse category / disclosed constraints / final top-10 titles),
* git + environment metadata, written to ``results/<short-sha>-<label>.json``.

Usage:
    AGENT_USE_LLM=0 python3 scripts/run_eval.py --label step1
    python3 scripts/run_eval.py --limit 50 --scenario browsing --label quick
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as ev  # noqa: E402  (organizer module, never modified)

SAFE_ENV_PREFIXES = ("AGENT_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "OPENAI_MODEL", "TOKENIZERS_")


class RecordingAgent:
    """Proxy that records every turn without changing the wrapped agent's behaviour."""

    def __init__(self, agent, catalog_ids: set[str]) -> None:
        self.agent = agent
        self.catalog_ids = catalog_ids
        self.records: list[list[dict]] = []  # one list of turn records per reset(), in call order
        self.session_ids: list[str] = []
        self._by_session: dict[str, list[dict]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        turns: list[dict] = []
        self.records.append(turns)
        self.session_ids.append(session_id)
        self._by_session[session_id] = turns
        return self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        try:
            response = self.agent.respond(session_id, user_message, turn, top_k)
        except Exception as exc:  # the evaluator turns this into a miss; record it first
            self._record(session_id, turn, user_message, None, time.perf_counter() - started, repr(exc))
            raise
        self._record(session_id, turn, user_message, response, time.perf_counter() - started, None)
        return response

    def _record(self, session_id, turn, user_message, response, latency, error) -> None:
        recs: list[str] = []
        ask = None
        if isinstance(response, dict):
            recs = ev.normalize_recommendations(response.get("recommendations"), self.catalog_ids)
            ask = response.get("ask_attribute")
        self._by_session.setdefault(session_id, []).append({
            "turn": turn,
            "user_message": user_message,
            "ask_attribute": ask,
            "recommendations": recs,
            "latency_s": round(latency, 6),
            "error": error,
        })


def git_meta() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    porcelain = [line for line in run("status", "--porcelain").splitlines() if line.strip()]
    dirty = [line for line in porcelain if not line[3:].startswith("results/")]
    return {
        "sha": run("rev-parse", "--short", "HEAD") or "nogit",
        "sha_full": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_paths": dirty[:50],
    }


def env_meta() -> dict:
    return {key: value for key, value in sorted(os.environ.items()) if key.startswith(SAFE_ENV_PREFIXES)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_latency(records: list[list[dict]]) -> dict:
    latencies = [turn["latency_s"] for session in records for turn in session]
    if not latencies:
        return {"turns": 0}
    return {
        "turns": len(latencies),
        "mean_s": round(statistics.fmean(latencies), 6),
        "p50_s": round(percentile(latencies, 0.5), 6),
        "p95_s": round(percentile(latencies, 0.95), 6),
        "max_s": round(max(latencies), 6),
        "errors": sum(1 for session in records for turn in session if turn["error"]),
    }


def disclosed_constraints(card: dict, records: list[dict]) -> list[str]:
    constraints = [str(value) for value in card.get("hard_constraints", [])] + [
        str(value) for value in card.get("soft_preferences", [])
    ]
    messages = [turn["user_message"] for turn in records]
    return [value for value in dict.fromkeys(constraints) if any(value in message for message in messages)]


def build_result(
    agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    label: str,
    extra_meta: dict | None = None,
) -> dict:
    proxy = RecordingAgent(agent, catalog_ids)
    started = time.perf_counter()
    result = ev.evaluate(proxy, samples, catalog_ids, categories, products)
    wall = time.perf_counter() - started

    sessions = result["sessions"]
    assert len(sessions) == len(samples) == len(proxy.records)

    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    by_scenario_difficulty: dict[str, list[dict]] = defaultdict(list)
    hit_turns: dict[str, Counter] = defaultdict(Counter)
    misses: list[dict] = []
    session_details: list[dict] = []
    debug = getattr(agent, "session_debug", None)

    for sample, session, records, session_id in zip(samples, sessions, proxy.records, proxy.session_ids):
        difficulty = str(sample.get("difficulty_bucket", "unknown"))
        scenario = str(sample["scenario_type"])
        by_difficulty[difficulty].append(session)
        by_scenario_difficulty[f"{scenario}/{difficulty}"].append(session)
        hit_turns[scenario][str(session["first_hit_turn"])] += 1
        target = str(sample["ground_truth"]["parent_asin"])
        product = products.get(target, {})
        card, _behavior = ev.materialize_hidden_fields(sample, products)
        detail = {
            "sample_id": sample["sample_id"],
            "scenario_type": scenario,
            "difficulty_bucket": difficulty,
            "hit": session["hit"],
            "first_hit_turn": session["first_hit_turn"],
            "best_rank": session["best_rank"],
            "target": target,
            "target_coarse_category": ev.coarse_category(categories.get(target, [])),
            "turns": records,
        }
        if callable(debug):
            try:
                detail["agent_debug"] = debug(session_id)
            except Exception:
                pass
        session_details.append(detail)
        if not session["hit"]:
            final = records[-1]["recommendations"] if records else []
            misses.append({
                "sample_id": sample["sample_id"],
                "scenario_type": scenario,
                "difficulty_bucket": difficulty,
                "target": target,
                "target_title": product.get("title"),
                "target_coarse_category": detail["target_coarse_category"],
                "target_rating_number": product.get("rating_number"),
                "disclosed_constraints": disclosed_constraints(card, records),
                "asked": [turn["ask_attribute"] for turn in records],
                "final_top10": [{"parent_asin": asin, "title": products.get(asin, {}).get("title")} for asin in final],
                "errors": [turn["error"] for turn in records if turn["error"]],
            })

    meta = {
        "label": label,
        "git": git_meta(),
        "env": env_meta(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall, 3),
        "latency": summarize_latency(proxy.records),
        "sample_count": len(samples),
    }
    if extra_meta:
        meta.update(extra_meta)
    bucket_stats = getattr(agent, "bucket_stats", None)
    if callable(bucket_stats):
        try:
            meta["bucket_stats"] = bucket_stats()
        except Exception:
            pass

    output = {key: value for key, value in result.items() if key != "sessions"}
    output["meta"] = meta
    output["difficulty_metrics"] = {name: ev.metric_summary(by_difficulty[name]) for name in sorted(by_difficulty)}
    output["scenario_x_difficulty"] = {
        name: ev.metric_summary(by_scenario_difficulty[name]) for name in sorted(by_scenario_difficulty)
    }
    output["hit_turn_histogram"] = {name: dict(sorted(counter.items())) for name, counter in sorted(hit_turns.items())}
    output["misses"] = misses
    output["sessions"] = sessions
    output["session_details"] = session_details
    return output


def format_table(output: dict) -> str:
    def row(name: str, metrics: dict, score: str = "") -> str:
        mttc = metrics.get("mttc")
        return (
            f"{name:28s} n={metrics['sample_count']:4d}  HR@10={metrics['hit_rate_at_10']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  MTTC={mttc if mttc is None else f'{mttc:.2f}'}  {score}"
        )

    lines = [row("overall", output, f"score={output['recommended_technical_score']:.4f} eff={output['efficiency']:.3f}")]
    for name, metrics in output["scenario_metrics"].items():
        lines.append(row(f"  scenario {name}", metrics))
    for name, metrics in output["difficulty_metrics"].items():
        lines.append(row(f"  difficulty {name}", metrics))
    for name, metrics in output["scenario_x_difficulty"].items():
        lines.append(row(f"  {name}", metrics))
    latency = output["meta"]["latency"]
    lines.append(
        f"latency: turns={latency.get('turns')} mean={latency.get('mean_s')}s p95={latency.get('p95_s')}s "
        f"max={latency.get('max_s')}s errors={latency.get('errors')}  wall={output['meta']['wall_time_s']}s  "
        f"tokens={output['reported_token_usage']['total_tokens']}"
    )
    lines.append(f"misses: {len(output['misses'])}  hit-turn histogram: {output['hit_turn_histogram']}")
    return "\n".join(lines)


def format_misses(misses: list[dict], limit: int) -> str:
    lines = []
    for miss in misses[:limit]:
        lines.append(
            f"- {miss['sample_id']} [{miss['scenario_type']}/{miss['difficulty_bucket']}] "
            f"{miss['target_coarse_category']!r} :: {str(miss['target_title'])[:90]!r}"
        )
        lines.append(f"    disclosed: {[c[:60] for c in miss['disclosed_constraints']]}")
        lines.append(f"    asked: {miss['asked']}")
        for item in miss["final_top10"][:10]:
            lines.append(f"    top: {str(item['title'])[:100]!r}")
    if len(misses) > limit:
        lines.append(f"... {len(misses) - limit} more misses in the JSON")
    return "\n".join(lines)


def load_samples(dataset: str, scenarios: list[str] | None, limit: int | None) -> list[dict]:
    samples = ev.load_jsonl(dataset)
    if scenarios:
        wanted = {name.strip() for name in scenarios if name.strip()}
        samples = [sample for sample in samples if sample["scenario_type"] in wanted]
    if limit is not None:
        samples = samples[:limit]
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Instrumented wrapper around evaluator.local_evaluator")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N (filtered) sessions")
    parser.add_argument("--scenario", action="append", default=None, help="scenario filter; repeatable or comma-separated")
    parser.add_argument("--label", default="run", help="label used in the results file name")
    parser.add_argument("--output", default=None, help="explicit output path (default results/<sha>-<label>.json)")
    parser.add_argument("--miss-log", default=None, help="also write the miss log as JSONL to this path")
    parser.add_argument("--print-misses", type=int, default=15, help="how many misses to print")
    args = parser.parse_args()

    scenarios = None
    if args.scenario:
        scenarios = [name for value in args.scenario for name in value.split(",")]
    samples = load_samples(args.dataset, scenarios, args.limit)
    if not samples:
        raise SystemExit("no sessions selected")

    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    from starter.agent import Agent  # imported late so env flags set by callers apply

    started = time.perf_counter()
    agent = Agent(args.catalog)
    construction = time.perf_counter() - started

    output = build_result(
        agent, samples, catalog_ids, categories, products, args.label,
        extra_meta={"agent_construction_s": round(construction, 3), "scenario_filter": scenarios, "limit": args.limit},
    )

    meta = output["meta"]
    sha = meta["git"]["sha"] + ("-dirty" if meta["git"]["dirty"] else "")
    path = Path(args.output) if args.output else ROOT / "results" / f"{sha}-{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.miss_log:
        Path(args.miss_log).write_text(
            "".join(json.dumps(miss, ensure_ascii=False) + "\n" for miss in output["misses"]), encoding="utf-8"
        )

    print(format_table(output))
    if output["misses"]:
        print(format_misses(output["misses"], args.print_misses))
    print(f"wrote {path}")
    if meta["git"]["dirty"]:
        print("WARNING: working tree dirty; results file is suffixed -dirty", file=sys.stderr)


if __name__ == "__main__":
    main()

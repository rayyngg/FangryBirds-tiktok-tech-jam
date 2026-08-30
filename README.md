# FangryBirds — TechJam 2026 Conversational E-Commerce Search (Track 4)

A multi-turn shopping agent for the organizer's `Clothing_Shoes_and_Jewelry` challenge: it keeps a
structured picture of what the customer has said, retrieves from the frozen 50,000-product catalog
with local indexes only, asks one clarification per turn, and returns a ranked top-10.

**Public-set score: 0.979 (Hit Rate@10 1.000, MRR 0.995, MTTC 1.97)** with the default configuration;
0.914 (HR 1.000, MRR 0.749, MTTC 1.53) with full top-10 lists on every turn (`AGENT_CONFIDENCE_GATE=0`).
The organizer's weak baseline is 0.107 and our previous submission tag `Technical-0.67` reproduces at 0.672.

## Quick start

Python 3.9+ (the organizer recommends 3.10+; the code compiles and the unit tests pass on 3.9.6 and
3.12.5, all measurements below are from 3.12.5). Everything is CPU-only and needs no network once the
catalog is present.

```bash
pip install -r requirements.txt                 # numpy + python-dotenv (both optional at runtime)
gzip -dk data/catalog.jsonl.gz                  # -> data/catalog.jsonl (verify with SHA256SUMS)
python3 -m evaluator.local_evaluator            # official harness, writes results.json (~15 s)
```

Instrumented runs (per-scenario / per-difficulty tables, miss log, latency, git metadata) go to
`results/<short-sha>-<label>.json`; the wrapper inserts `-dirty-` after the sha when the tree had
uncommitted changes, so `results/72bd980-dirty-*` are the step-by-step history produced before those
steps were committed (kept as evidence) and `results/697ff18-*` are the current headline runs:

```bash
AGENT_USE_LLM=0 python3 scripts/run_eval.py --label my-run
python3 scripts/run_eval.py --limit 50 --scenario browsing --label quick   # subsets (never judge on <50)
python3 scripts/demo_session.py --sample public_0006                        # one session as a transcript
python3 scripts/perturb_eval.py --configs base,gate                         # robustness study (below)
python3 -m unittest tests.test_evaluator tests.test_cardspec tests.test_agent
RUN_SLOW=1 python3 -m unittest tests.test_public_regression                 # asserts >= 0.90 / >= 0.96
```

## Offline mode (submission_rules.md "Model Policy")

**The submission does not require network access or any credentials.** The default agent uses only the
Python standard library (SQLite FTS5 + in-memory indexes built from `data/catalog.jsonl` at start-up,
≈5 s). Two optional tiers exist and are **off by default**:

| tier | switch | behaviour without network / credentials |
|---|---|---|
| dense re-scoring (MiniLM sentence-transformer) | `AGENT_USE_EMBEDDINGS=1` | model is loaded strictly from local files (`AGENT_MODEL_PATH`, `models/all-MiniLM-L6-v2`, or the local HF cache, `local_files_only=True`); `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set to `1` by the agent itself; any failure silently falls back to the lexical/structural ranking. The catalog is never encoded at construction; the tier only activates if `data/catalog.embeddings.npy` exists (`scripts/build_embeddings.py`). |
| LLM phrasing of the customer-facing message | `AGENT_USE_LLM=1` + `OPENAI_API_KEY` | falls back to templated messages; the LLM never influences ranking or `ask_attribute`. |

Neither tier changes the score on the public set (dense: −0.001; LLM: 0 by construction), which is why
both are disabled. Construction (`Agent.__init__`) and `reset` never raise, `respond` catches every
internal error and still returns a valid dict (bucket or global popularity list, ask `other`), and the
agent never writes to disk. If the host's SQLite lacks FTS5 (or `AGENT_DISABLE_FTS=1`), a pure-Python
token index replaces it: identical scores (0.9791 / 0.9140), p95 latency ≈ 50 ms
(`results/*-nofts-*.json`). The catalog path is the harness's argument, else `AGENT_CATALOG_PATH`, else
`<repo>/data/catalog.jsonl`; a relative path that does not exist from the current working directory is
retried relative to the repository, so the harness can run from any directory.

Verified: a real `git clone` of this branch into a fresh virtualenv with only `requirements.txt`
installed (no `sentence-transformers`, no `openai`, no `.env`, no `OPENAI_API_KEY`, empty `HF_HOME`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) scores 0.9791 through `python -m evaluator.local_evaluator`
and passes `python -m unittest`; the same clone with `AGENT_USE_EMBEDDINGS=1` or `AGENT_USE_LLM=1`
degrades at construction (no exception, same score, `HF_HOME` stays empty) —
`results/697ff18-offline-clean-clone.json`.

## How it works

The public simulator is deterministic and its rules are public (`evaluator/local_evaluator.py`):
the hidden intent card is verbatim product metadata, the opener names `coarse_category(target)`, and
an ask for attribute *a* reveals the first two undisclosed card constraints of class *a* (`other`
matches everything). The agent is built around those rules, with tolerant fallbacks for anything that
deviates from them.

Per turn (`starter/agent.py`):

1. **Parse** (`starter/parsing.py`) the customer message into `SessionState` (`starter/state.py`):
   category / bucket, hard and soft constraints (verbatim strings), a mirror of the simulator's
   disclosure set, exhausted attributes, override and boundary flags, items already shown. Templates
   are matched by anchored prefixes (never "text after the first colon" — constraints contain colons,
   periods and semicolons); a `;`-split inside a reveal is resolved by checking which split a bucket
   product could actually have produced. Paraphrased messages fall through to keyword override
   detection, category recognition anywhere in the sentence, exact-index lookup of any verbatim
   constraint pieces, and free text for the lexical tiers.
2. **Rank** (`starter/retrieval.py`) a bounded candidate pool — bucket members ∪ exact-index postings of
   the known constraints ∪ FTS5 hits ∪ the 300 most popular products, never a full catalog scan — by a
   lexicographic key: *in the opener's bucket* › *number of constraints found in the product's own
   card* (`starter/cardspec.py` reproduces `intent_card` for every product; parity is unit-tested on
   all 50,000) › *reply-consistency tie-break* › *case-insensitive substring matches in
   title/features/details* › *price inside a stated budget window (±5 %)* › *not yet shown this
   session* (paging among equally supported candidates; a product from another bucket never outranks
   an in-bucket match just because the latter was shown) › *popularity `log1p(rating_number)`*.
   Tiers, never filters: the list is never short, and a mis-parsed bucket or constraint demotes the
   target instead of hiding it.
3. **Show** the top 10 (or, with the confidence gate, fewer — see below). The message states the
   actual count; when the gate holds items back it says the agent has a strong candidate and is
   confirming one detail before showing more.
4. **Ask** (`starter/ask_policy.py`): `other` every turn, never `null` (a null ask yields a
   zero-information reply), also after "no preference" replies and through the last turn. The
   question-value estimator (expected next-turn reciprocal rank under the reveal rule with a
   popularity prior) is available as `AGENT_ASK_POLICY=value`; on the public set it never beats
   `other`, so it is used only as the deferral estimate for the gate.
5. **Phrase** a templated message (or the optional LLM).

Intent overrides keep every earlier constraint (the "ignored" preference is still true of the target;
the new requirement joins the hard constraints) and reset the shown-items bookkeeping, because
pre-override turns are never scored. In sessions whose opener looks like an override, nothing is
recorded as shown until the override arrives.

### Confidence gate (`AGENT_CONFIDENCE_GATE`, default 1)

Per session the score decomposes as `0.5·hit + 0.30/rank + 0.02·(11 − turn)`, so a rank-1 hit one turn
later is worth more than a rank-2 hit now. The gate shows rank *r* on turn *t* only when
`0.30/r − 0.02·t ≥ 0.30·E[rr_next] − 0.02·(t+1)`, where `E[rr_next]` is the estimated reciprocal rank
after one more `other` reply; guards keep the full list whenever the message did not match a template,
the card is exhausted, the turn is late (> 4), or the session is still pre-override. It never returns
an empty list. It raises the public score from 0.914 to 0.979 (MRR 0.749 → 0.995, MTTC 1.53 → 1.97).
The behaviour is metric-driven (the agent shows its single best pick and asks while it is still
uncertain, then the full list once the answer pins the target), so the default was not decided on the
public score alone: the robustness study below shows the gate scoring at least as high as full lists
under every simulator perturbation, which is why it is on. `AGENT_CONFIDENCE_GATE=0` restores full
top-10 lists on every turn (0.914).

## Results (public set, 200 sessions)

| step | results file | score | HR@10 | MRR | MTTC |
|---|---|---|---|---|---|
| organizer weak baseline (`docs/baseline_results.json`) | – | 0.107 | 0.125 | 0.068 | 9.81 |
| tag `Technical-0.67` reproduced (old agent, embeddings on) | `420d011-baseline` | 0.672 | 0.795 | 0.462 | 4.19 |
| 1b offline-safe construction (old ranking, embeddings off) | `61a30f7-step1b-offline-safe` | 0.660 | 0.780 | 0.454 | 4.33 |
| 2 ask policy: `other` every turn, never null | `72bd980-step2-ask-policy` | 0.696 | 0.800 | 0.515 | 3.94 |
| 3 coarse-category tier + popularity prior | `72bd980-dirty-step3-category-filter` | 0.792 | 0.870 | 0.636 | 2.69 |
| 4 constraint parser + exact/substring tiers | `72bd980-dirty-step4-constraints` | 0.911 | 1.000 | 0.740 | 1.53 |
| 5 override handling + shown demotion, full lists (`AGENT_CONFIDENCE_GATE=0`) | `697ff18-dirty-final-default` | **0.914** | 1.000 | 0.749 | 1.53 |
| 8 confidence gate (**default**) | `697ff18-dirty-final-gate-on`, `697ff18-offline-clean-clone` | **0.979** | 1.000 | 0.995 | 1.97 |

(`697ff18-dirty-*` were produced on top of commit 697ff18 with the ranker key-order fix and message
change of the verification pass uncommitted; `scripts/regenerate_results.sh` re-creates them with a
clean sha after committing. Scores are identical before and after that fix.)

Per scenario (HR / MRR / MTTC, full lists → gate): buying 1.000 / 0.771 / 1.09 → 1.000 / 1.000 / 1.48;
browsing 1.000 / 0.674 / 1.21 → 1.000 / 1.000 / 1.79; intent override 1.000 / 0.967 / 3.60 (both; the
evaluator ignores hits before the override turn, so 3.6 is the floor); boundary 1.000 / 0.514 / 1.40 →
1.000 / 1.000 / 2.50. Zero misses in either setting.

Ablations (same code, environment switches): dense tier on (`AGENT_USE_EMBEDDINGS=1 AGENT_W_EMBED=0.2`)
0.913; question-value ask policy 0.914; secondary ordering by popularity+BM25 (`AGENT_W_FTS=0.15`)
0.912, (`0.3`) 0.907, BM25 only 0.896.

Tuned parameters (all chosen on the public set; every other constant is derived from the simulator's
rules or the score formula): secondary-score blend `AGENT_W_POP=1.0, AGENT_W_FTS=0.0, AGENT_W_EMBED=0.0`;
candidate-pool caps `EXACT_POSTING_CAP=1500`, `FTS_POOL_LIMIT=300`, `POPULARITY_HEAD=300`; gate guard
`GATE_MAX_TURN=4`; question-value margin `0.02`.

## Robustness study (`scripts/perturb_eval.py`)

The simulator, not the agent, is modified at runtime (the evaluator file is untouched) to mimic ways
the private set could differ: different card generation, paraphrased templates, a moved override turn.

| perturbation of the simulator | full lists — score (HR / MRR / MTTC) | gate — score (HR / MRR / MTTC) |
|---|---|---|
| none (public rules) | 0.9140 (1.000 / 0.749 / 1.53) | **0.9791** (1.000 / 0.995 / 1.97) |
| cards without the material / colour insertion | 0.9268 (1.000 / 0.791 / 1.52) | 0.9640 (0.995 / 0.950 / 1.92) |
| constraints truncated at 120 instead of 180 chars | 0.9140 (1.000 / 0.749 / 1.53) | 0.9791 (1.000 / 0.995 / 1.97) |
| hard and soft constraints swapped | 0.9114 (1.000 / 0.743 / 1.57) | 0.9772 (1.000 / 0.993 / 2.02) |
| all constraint strings lower-cased | 0.9034 (1.000 / 0.714 / 1.53) | 0.9627 (1.000 / 0.945 / 2.04) |
| reveal template paraphrased ("What I care about most is …") | 0.9053 (1.000 / 0.720 / 1.54) | 0.9462 (1.000 / 0.881 / 1.91) |
| opener paraphrased ("Hi! I want to buy X. It must have: …") | 0.9121 (1.000 / 0.742 / 1.53) | 0.9173 (1.000 / 0.762 / 1.56) |
| override message paraphrased ("Forget what I said before …") | 0.9140 (1.000 / 0.749 / 1.53) | 0.9791 (1.000 / 0.995 / 1.97) |
| override turn moved to 2–6 | 0.9128 (1.000 / 0.747 / 1.57) | 0.9795 (1.000 / 1.000 / 2.02) |
| opener + reveal + override all paraphrased | 0.9043 (1.000 / 0.717 / 1.54) | 0.9055 (1.000 / 0.721 / 1.54) |

Hit Rate stays at 1.000 in every row but one (the gate loses a single override session when cards
are generated without the material/colour insertion), and the gate never scores below the full-list
variant — the reason it is the default. Paraphrasing costs MRR, not hits: the exact tiers stop firing
and the substring / popularity tiers carry the session. Source: `results/697ff18-dirty-perturb.json` / `.md`.

## Latency, tokens and cost

Measured on an Apple M4 Pro, CPU only: construction ≈ 5 s (FTS5 build + card replica for 50k
products); per-turn latency mean ≈ 20 ms, p95 ≈ 50 ms, max < 0.2 s (`meta.latency` in every results
file). Reported token usage is 0 by default. With `AGENT_USE_LLM=1` (gpt-4o-mini) one call per turn of
≈150 prompt + ≈40 completion tokens ≈ $0.00005 per turn; it only rewrites the customer-facing sentence.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `AGENT_CATALOG_PATH` | `<repo>/data/catalog.jsonl` | catalog location (the harness's explicit argument wins) |
| `AGENT_USE_BUCKET` / `AGENT_PARSE_REPLIES` / `AGENT_DEMOTE_SHOWN` | `1` | ablation switches for the three structural tiers |
| `AGENT_ASK_POLICY` | `other` | `other` or `value` |
| `AGENT_CONFIDENCE_GATE` | `1` | confidence-gated list length (`0` = full top-10 every turn) |
| `AGENT_USE_EMBEDDINGS`, `AGENT_MODEL_PATH`, `AGENT_W_EMBED` | `0`, `models/all-MiniLM-L6-v2`, `0.0` | optional dense tier |
| `AGENT_W_POP`, `AGENT_W_FTS` | `1.0`, `0.0` | secondary-score blend |
| `AGENT_USE_LLM`, `OPENAI_API_KEY`, `OPENAI_MODEL` | `0`, –, `gpt-4o-mini` | optional message phrasing |

## Limitations

* The exact-match tiers assume private intent cards are generated like the public ones (verbatim
  metadata, same cleaning). The study above shows what survives when they are not; the substring and
  lexical tiers plus the popularity prior are the safety net.
* Intent-override sessions cannot convert before the override turn, so their MTTC floor is 3.6.
* The popularity prior is learned from the data, not the customer: it helps the first turn but would
  not help a customer looking for an obscure item.
* The user profile carries no information about the target in this simulator and is ignored for
  retrieval.

## Files

```text
starter/agent.py          Agent (reset / respond), pipeline orchestration, fallbacks
starter/cardspec.py       replica of the simulator's card / category / class rules (+ reveal rule)
starter/parsing.py        message templates -> structured state, paraphrase fallbacks
starter/state.py          SessionState
starter/retrieval.py      CatalogIndex (FTS5 + structural indexes) and rank_candidates
starter/ask_policy.py     fixed "other" policy, question-value estimator, message templates
starter/embedder.py       optional dense tier, offline-only loader
scripts/run_eval.py       instrumented wrapper around the organizer's evaluator
scripts/perturb_eval.py   robustness study against perturbed simulators
scripts/demo_session.py   transcript of one session
scripts/fetch_model.py    optional: download the MiniLM model into models/
scripts/build_embeddings.py optional: build data/catalog.embeddings.npy
tests/                    organizer tests + replica parity + agent behaviour + slow regression
results/                  one JSON per measured configuration (score, per-scenario, misses, latency, git sha)
```

Data source: Amazon Reviews 2023 (McAuley Lab, UCSD) — see `DATA_ATTRIBUTION.md`.

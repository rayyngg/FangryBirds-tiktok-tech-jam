# FangryBirds: TechJam 2026 Track 4, Conversational E-Commerce Search

A multi-turn shopping agent for the organizer's Clothing, Shoes and Jewelry challenge. It keeps a structured record of what the customer has said, retrieves from the frozen 50,000-product catalog with local indexes only, asks one clarifying question per turn, and returns a ranked top 10.

Public-set score 0.979 (Hit Rate@10 1.000, MRR 0.995, MTTC 1.97). With a full top-10 list on every turn it scores 0.914. The organizer's baseline is 0.107; our first agent, tagged `Technical-0.67`, reproduces at 0.672. The agent needs no network, no API keys and no model downloads, uses zero tokens, and answers in about 20 ms per turn.

Demo video: https://youtu.be/Wkq7LaqMDVI. It walks through the harness run, one session per scenario, and the robustness study.

In one sentence: the simulated customer is a deterministic function of the target product's metadata, so instead of re-searching the catalog every turn the agent keeps a structured record of what has been said, mirrors what the simulator has disclosed, and shows fewer items while one more question is expected to pin the target down.

## Quick start

Python 3.9 or later (the organizer recommends 3.10+; we tested 3.9.6 and 3.12.5, and every number below comes from 3.12.5). CPU only.

```bash
pip install -r requirements.txt        # numpy and python-dotenv; both optional at runtime
gzip -dk data/catalog.jsonl.gz         # produces data/catalog.jsonl, check against SHA256SUMS
python3 -m evaluator.local_evaluator   # the organizer's harness, writes results.json in about 15 s
```

## Reproduce the numbers in this README

```bash
AGENT_USE_LLM=0 python3 -m evaluator.local_evaluator                            # 0.9791, default, gate on
AGENT_USE_LLM=0 AGENT_CONFIDENCE_GATE=0 python3 -m evaluator.local_evaluator    # 0.9140, full lists
AGENT_USE_LLM=0 python3 scripts/perturb_eval.py --configs base,gate             # robustness table, about 3 min
bash scripts/regenerate_results.sh                                              # four headline files plus the clean-clone check
```

The agent has no randomness and the evaluator seeds its generator per sample, so the scores reproduce exactly; a clean clone on another machine and Python 3.9 both printed the same numbers. Only timings vary.

Our instrumented runs add per-scenario and per-difficulty tables, a miss log, latency and the git sha, and write to `results/<sha>-<label>.json`:

```bash
AGENT_USE_LLM=0 python3 scripts/run_eval.py --label my-run
python3 scripts/run_eval.py --limit 50 --scenario browsing --label quick   # subsets; do not judge on fewer than 50 sessions
python3 scripts/demo_session.py --sample public_0006                        # one session as a transcript
python3 -m unittest tests.test_evaluator tests.test_cardspec tests.test_agent
RUN_SLOW=1 python3 -m unittest tests.test_public_regression                 # asserts at least 0.90 and 0.96
```

Files named `results/72bd980-dirty-*` were produced from the working tree while we were building step by step and are kept as history. The current headline runs are `results/64220b8-*`.

## What we found in the evaluator

Our first agent was a hybrid of SQLite BM25 and MiniLM embeddings with a bag of recent messages as state. It scored 0.672 and we could not see why it missed. So before rebuilding, we read `evaluator/local_evaluator.py` line by line and checked each finding against the data. The simulator is deterministic and its rules are public:

- The hidden intent card is the target product's own metadata: a material word and a colour pulled out by regex, then its feature bullets and detail fields, capped at four constraints. Every public card has exactly four. A budget line exists in the generator but almost never makes the cut.
- The opening message names `coarse_category(target)`, the last two parts of the category path. That gives 1,115 buckets; the median target bucket has 184 products.
- Asking about attribute *a* reveals the next two undisclosed constraints of class *a*, word for word. Asking `other` reveals the next two of any class. So `other` twice discloses the whole card, and `brand` or `category` never reveal anything.
- The target sits at the 99th popularity percentile of its bucket (median), so `rating_number` is a strong prior before any constraint arrives.
- In override sessions the "earlier preference" the customer asks us to ignore is a true property of the target, and hits before the override turn are not scored.
- The harness does not catch exceptions in the agent's constructor, and the organizer may disable network access for final scoring.

Everything below follows from those facts. Where the private set might behave differently, we measured what happens rather than guessed (see the robustness study).

## How the agent works

Each turn, `starter/agent.py` does four things.

**Parse** (`starter/parsing.py`). The customer message is turned into a `SessionState` (`starter/state.py`): category and bucket, hard and soft constraints as verbatim strings, a mirror of what the simulator considers disclosed, exhausted attributes, override and boundary flags, and the products already shown. Templates are matched by anchored prefixes rather than "everything after the first colon", because constraints themselves contain colons, periods and semicolons. When a reveal contains a semicolon we resolve the split by checking which split a product in the bucket could actually have produced. Messages that match no template fall through to keyword override detection, category recognition anywhere in the sentence, exact-index lookup of any verbatim pieces, and free text for the lexical tiers.

**Rank** (`starter/retrieval.py`). We score a bounded pool (bucket members, exact-index postings of the known constraints, FTS5 hits, and the 300 most popular products; never the whole catalog) with a lexicographic key:

1. in the opener's bucket
2. number of known constraints found in the product's own card (`starter/cardspec.py` reproduces the simulator's `intent_card` for every product; parity is unit-tested on all 50,000)
3. consistency with the replies seen so far
4. case-insensitive substring matches in title, features and details
5. price inside a stated budget window (plus or minus 5%)
6. not yet shown in this session
7. popularity, `log1p(rating_number)`

These are tiers, not filters. The list is never short, and a mis-parsed bucket or constraint demotes the target instead of hiding it. Products already shown page to the back of their tier, but a product from another bucket never outranks an in-bucket match just because that match was shown before.

**Show** the top 10, or fewer when the confidence gate holds items back (next section). The message states the actual count; when the gate is holding back, it says the agent has a strong candidate and is confirming one detail before showing more.

**Ask** (`starter/ask_policy.py`). We ask `other` every turn, never `null` (a null ask returns a zero-information reply), including after "no preference" replies and on the last turn. Under the reveal rule `other` is the only question that cannot come back empty while something is left to reveal, and at most two constraints arrive per turn whatever you ask. We also built a question-value estimator, the expected next-turn reciprocal rank under the reveal rule with a popularity prior, available as `AGENT_ASK_POLICY=value`. On the public set it never beats `other`, so it is used only as the deferral estimate for the gate.

Overrides keep every earlier constraint, because the "ignored" preference is still true of the target; the new requirement joins the hard constraints. The shown-items bookkeeping resets at the override, since pre-override turns are never scored, and in sessions whose opener looks like an override nothing is recorded as shown until the override arrives.

### The confidence gate

Per session the score decomposes as 0.5 for the hit, plus 0.30 / rank, minus 0.02 per turn. So a rank-1 hit one turn later is worth more than a rank-2 hit now. The gate shows rank *r* on turn *t* only when 0.30 / r minus 0.02 t is at least 0.30 times the expected reciprocal rank after one more reply, minus 0.02 (t + 1). Guards keep the full list whenever the message did not match a template, the card is exhausted, the turn is later than 4, or the session is still pre-override. It never returns an empty list.

In practice this means: while the customer is still describing what they want, show the single best candidate and ask; once the reply pins the product, show the full list. It raises the public score from 0.914 to 0.979 (MRR 0.749 to 0.995, MTTC 1.53 to 1.97). We did not switch it on because of the public score alone: in the robustness study below the gate never scores below full lists under any simulator change, and under total paraphrase it switches itself off. `AGENT_CONFIDENCE_GATE=0` restores full top-10 lists on every turn.

## Results on the public set (200 sessions)

| step | results file | score | HR@10 | MRR | MTTC |
|---|---|---|---|---|---|
| organizer weak baseline (`docs/baseline_results.json`) | | 0.107 | 0.125 | 0.068 | 9.81 |
| our first agent, tag `Technical-0.67` (embeddings on) | `420d011-baseline` | 0.672 | 0.795 | 0.462 | 4.19 |
| same ranking, offline-safe construction, embeddings off | `61a30f7-step1b-offline-safe` | 0.660 | 0.780 | 0.454 | 4.33 |
| ask `other` every turn, never null | `72bd980-step2-ask-policy` | 0.696 | 0.800 | 0.515 | 3.94 |
| category tier and popularity prior | `72bd980-dirty-step3-category-filter` | 0.792 | 0.870 | 0.636 | 2.69 |
| constraint parser, exact and substring tiers | `72bd980-dirty-step4-constraints` | 0.911 | 1.000 | 0.740 | 1.53 |
| override handling and shown demotion, full lists | `64220b8-final-default` | 0.914 | 1.000 | 0.749 | 1.53 |
| confidence gate (default) | `64220b8-final-gate-on`, `64220b8-offline-clean-clone` | 0.979 | 1.000 | 0.995 | 1.97 |

`scripts/regenerate_results.sh` recreates the four headline files, including the clean-clone offline run, for the current commit.

Per scenario, full lists then gate (HR / MRR / MTTC): buying 1.000 / 0.771 / 1.09, then 1.000 / 1.000 / 1.48. Browsing 1.000 / 0.674 / 1.21, then 1.000 / 1.000 / 1.79. Intent override 1.000 / 0.967 / 3.60 in both settings; the evaluator ignores hits before the override turn, so 3.6 is the floor. Boundary 1.000 / 0.514 / 1.40, then 1.000 / 1.000 / 2.50. Zero misses in either setting.

Things we measured and left off. The dense tier (`AGENT_USE_EMBEDDINGS=1 AGENT_W_EMBED=0.2`) scores 0.913, slightly below the default. The question-value ask policy scores 0.914, the same as `other`. Blending BM25 into the secondary order scores 0.912 at weight 0.15, 0.907 at 0.3, and 0.896 with BM25 alone. The replies are verbatim catalog strings, so lexical and semantic similarity have nothing to add once exact matching works.

Parameters chosen on the public set, all listed here so nothing is hidden: the secondary blend `AGENT_W_POP=1.0, AGENT_W_FTS=0.0, AGENT_W_EMBED=0.0`; the pool caps `EXACT_POSTING_CAP=1500`, `FTS_POOL_LIMIT=300`, `POPULARITY_HEAD=300`; the gate's last turn `GATE_MAX_TURN=4`; the question-value margin 0.02. Every other constant comes from the simulator's rules or the score formula.

## Robustness study (`scripts/perturb_eval.py`)

The private set is scored by the same harness on 800 unseen sessions, and the specification says the organizer may add paraphrasing. We could not test that directly, so we changed the simulator at runtime (the evaluator file is untouched) in ten ways the private set might differ, and ran the unchanged agent against each.

| change to the simulator | full lists, score (HR / MRR / MTTC) | gate, score (HR / MRR / MTTC) |
|---|---|---|
| none (public rules) | 0.9140 (1.000 / 0.749 / 1.53) | 0.9791 (1.000 / 0.995 / 1.97) |
| cards without the material and colour insertion | 0.9268 (1.000 / 0.791 / 1.52) | 0.9640 (0.995 / 0.950 / 1.92) |
| constraints truncated at 120 characters instead of 180 | 0.9140 (1.000 / 0.749 / 1.53) | 0.9791 (1.000 / 0.995 / 1.97) |
| hard and soft constraints swapped | 0.9114 (1.000 / 0.743 / 1.57) | 0.9772 (1.000 / 0.993 / 2.02) |
| all constraint strings lower-cased | 0.9034 (1.000 / 0.714 / 1.53) | 0.9627 (1.000 / 0.945 / 2.04) |
| reveal template paraphrased ("What I care about most is ...") | 0.9053 (1.000 / 0.720 / 1.54) | 0.9462 (1.000 / 0.881 / 1.91) |
| opener paraphrased ("Hi! I want to buy X. It must have: ...") | 0.9121 (1.000 / 0.742 / 1.53) | 0.9173 (1.000 / 0.762 / 1.56) |
| override message paraphrased ("Forget what I said before ...") | 0.9140 (1.000 / 0.749 / 1.53) | 0.9791 (1.000 / 0.995 / 1.97) |
| override turn moved to anywhere in 2 to 6 | 0.9128 (1.000 / 0.747 / 1.57) | 0.9795 (1.000 / 1.000 / 2.02) |
| opener, reveal and override all paraphrased | 0.9043 (1.000 / 0.717 / 1.54) | 0.9055 (1.000 / 0.721 / 1.54) |

Hit rate stays at 1.000 in every row but one, where the gate loses a single override session. Paraphrasing costs MRR, not hits: the exact tiers stop firing and the substring and popularity tiers carry the session. Source: `results/64220b8-perturb.json` and `.md`.

## How this maps to the brief

The problem statement describes four pillars. We built against each of them, and where we deviated it was because a measurement told us to.

**Intent routing and hybrid pipeline.** The opener is classified as buying, browsing or override. A buying constraint goes straight into the top evidence tier, which locks it in effect; a browsing session leans on the category bucket, the popularity prior and an immediate clarifying question. Retrieval is multi-route: keyword (FTS5), category (bucket), exact card strings, and an optional vector tier. The ranking stage uses local scoring logic, which the brief lists as in scope, instead of an LLM. We measured the alternatives: the dense tier is worth minus 0.001 here, and there is nothing for an LLM to rerank when the customer's replies are the catalog's own strings. Both remain available behind switches.

**Dialogue strategy.** The state machine accumulates constraints across turns. On an intent override we keep the earlier constraint and demote it rather than erase it, because in this simulator the earlier preference is still true of the product; override sessions score HR 1.000 with the floor MTTC. Proactive guidance is the confidence gate: when the candidate pool is still ambiguous the agent cuts the list to its single best candidate and asks a structured clarification (`ask_attribute`).

**Context distillation and adaptive orchestration.** The transcript is compressed into the structured state each turn rather than replayed. The per-turn choice between asking and showing, and how much to show, is made from an expected-value estimate rather than a fixed script. The user profile is not used: in this data it carries tags like "fit, comfort, durability" and no information about the target.

**Evaluation metrics.** Hit Rate@10, MRR and MTTC are optimised directly; every design choice above has its number in the results table.

## Offline mode and reliability

The submission needs no network access and no credentials. The default agent uses the Python standard library and numpy: SQLite FTS5 plus in-memory indexes built from `data/catalog.jsonl` at start-up in about 5 s. Two optional tiers are off by default. The dense tier (`AGENT_USE_EMBEDDINGS=1`) loads a MiniLM sentence-transformer strictly from local files (`AGENT_MODEL_PATH`, `models/all-MiniLM-L6-v2`, or the local Hugging Face cache) with `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` set by the agent itself, and only activates if `data/catalog.embeddings.npy` exists (`scripts/build_embeddings.py`); any failure falls back silently to the lexical ranking. LLM phrasing (`AGENT_USE_LLM=1` plus `OPENAI_API_KEY`) rewrites the customer-facing sentence and nothing else; without a key it uses templates.

The constructor and `reset` never raise. `respond` catches every internal error and still returns a valid response (a bucket or global popularity list and the ask `other`). The agent never writes to disk. If the host's SQLite lacks FTS5, or `AGENT_DISABLE_FTS=1`, a pure-Python token index takes over with identical scores (0.9791 and 0.9140) and p95 latency of about 50 ms (`results/*-nofts-*.json`). The catalog path is the harness's argument, else `AGENT_CATALOG_PATH`, else `<repo>/data/catalog.jsonl`; a relative path that does not exist from the working directory is retried relative to the repository.

We verified this the hard way: a real `git clone` into a fresh virtualenv with only `requirements.txt` installed (no sentence-transformers, no openai, no `.env`, no API key, an empty `HF_HOME`, both offline flags set) scores 0.9791 through `python -m evaluator.local_evaluator` and passes `python -m unittest`. The same clone with `AGENT_USE_EMBEDDINGS=1` or `AGENT_USE_LLM=1` degrades at construction with no exception and the same score, and `HF_HOME` stays empty. That run is `results/64220b8-offline-clean-clone.json`.

## Latency, tokens and cost

Measured on an Apple M4 Pro, CPU only: construction about 5 s (FTS5 build plus the card replica for 50,000 products); per-turn latency mean about 20 ms, p95 about 50 ms, maximum under 0.2 s (`meta.latency` in every results file). Reported token usage is 0 by default. With `AGENT_USE_LLM=1` and gpt-4o-mini, one call per turn of roughly 150 prompt and 40 completion tokens costs about $0.00005; it only rewrites the customer-facing sentence.

## Reflection: limitations, and what we would do with more time

The honest summary is that this agent is very good at the game the public simulator plays, and we do not fully know how far that carries.

- The exact-match tiers assume the private cards are built the way the public ones are: verbatim metadata, same cleaning. The robustness study is our best answer to "what if they are not": hit rate survives every perturbation we tried, but MRR falls to between 0.72 and 0.88 once messages are paraphrased, because the substring and popularity tiers are doing the work alone. A real customer would never quote "75% Polyester, 20% Rayon, 5% Spandex"; they would say "something stretchy". Given more time, the first thing we would build is a fuzzy constraint matcher between the exact tier and the substring tier (token overlap or a small cross-encoder, trained on paraphrases generated from the catalog itself), or an LLM-backed parser used only when no template matches, so the gate can switch back on under paraphrase.
- Asking `other` every turn is optimal for this simulator and a bad habit anywhere else. The question-value estimator in `ask_policy.py` is the right shape for a real product (pick the attribute that best splits the current candidates); it never wins here because the reveal rule makes an open question dominant. We would keep it on in production.
- Intent-override sessions cannot convert before the override turn, so their MTTC floor is 3.6 and the composite cannot reach 1.0.
- The popularity prior comes from the catalog, not the customer. It helps on turn 1 (targets are overwhelmingly popular products) and would hurt someone shopping for an obscure item. The user profile carries no target information in this simulator and is ignored; in a real system it is the obvious place to replace the popularity prior with a personal one.
- Two cases we did not test: an override message with no recognisable keyword after the target was already shown, and an opener that names a neighbouring category. The ranking key handles the first by design (evidence outranks "already shown"); the second could hold the target below the wrong bucket, and moving exact matches ahead of the bucket tier is the one-line fix we would try first.
- The messages are templated. They are honest about what the agent is doing (how many items, why it is holding back), but they read like templates. The optional LLM tier exists for phrasing; we left it off because it changes nothing measurable and adds a network dependency.
- No explanation per item. The ranking key already knows which constraints each product satisfies; surfacing that as "matches: 100% Leather, Buckle closure" is a small change we ran out of time for and would add first.

## What would change for real customers

The simulator's customers speak in catalog strings. Real ones do not, so the parser would need a language layer in front of it, and vague or cross-category requests ("something for a beach wedding") would need the dense tier that adds nothing here. The rest transfers: the structured state with override handling, the evidence-tier ranking with a popularity prior, the expected-value decision about whether to ask or show, and the cost profile of zero API calls and 20 ms per turn. The robustness study is also a habit we would keep; it is the closest thing we had to a private set.

## Team

FangryBirds is four people. Contributions, roughly in the order things happened:

- Ray: set up the repository and catalog, ran the starter kit, built the first intent-override handling that became the `Technical-0.67` submission, reviewed and merged the structured-agent pull request.
- Cedric: the first retrieval upgrade, a vector index and sentence-transformer semantic search over the catalog, merged into the shared agent.
- Jeevan: session memory and code annotation in the early agent, broadened the intent-override trigger, integrated the sentence-transformer scores into the ranking.
- Fang Chenyu: the structured-state rebuild (simulator-rule replica, template parser, tiered retrieval, ask policy and confidence gate), the test suite, the instrumented evaluator, the perturbation study, offline hardening.

Design decisions (the `other` policy, making the gate the default) were discussed as a group with the perturbation numbers in front of us. We used Claude Code as an AI pair programmer for implementation and the verification scripts; every design decision and every number above was checked by us against the evaluator source and the data. Video and README were done together.

## Switches

Every switch has a default that matches the reported results.

- `AGENT_CATALOG_PATH`: catalog location; the harness's explicit argument wins.
- `AGENT_USE_BUCKET`, `AGENT_PARSE_REPLIES`, `AGENT_DEMOTE_SHOWN` (default 1): ablation switches for the three structural tiers.
- `AGENT_ASK_POLICY` (default `other`): `other` or `value`.
- `AGENT_CONFIDENCE_GATE` (default 1): 0 gives full top-10 lists every turn.
- `AGENT_USE_EMBEDDINGS`, `AGENT_MODEL_PATH`, `AGENT_W_EMBED` (defaults 0, `models/all-MiniLM-L6-v2`, 0.0): optional dense tier.
- `AGENT_W_POP`, `AGENT_W_FTS` (defaults 1.0, 0.0): secondary-score blend.
- `AGENT_USE_LLM`, `OPENAI_API_KEY`, `OPENAI_MODEL` (defaults 0, none, gpt-4o-mini): optional message phrasing.
- `AGENT_DISABLE_FTS` (default 0): force the pure-Python token index.

## Files

```text
starter/agent.py            Agent (reset / respond), pipeline orchestration, fallbacks
starter/cardspec.py         replica of the simulator's card, category and class rules, plus the reveal rule
starter/parsing.py          message templates to structured state, paraphrase fallbacks
starter/state.py            SessionState
starter/retrieval.py        CatalogIndex (FTS5 and structural indexes) and rank_candidates
starter/ask_policy.py       fixed "other" policy, question-value estimator, message templates
starter/embedder.py         optional dense tier, offline-only loader
scripts/run_eval.py         instrumented wrapper around the organizer's evaluator
scripts/perturb_eval.py     robustness study against perturbed simulators
scripts/demo_session.py     transcript of one session
scripts/fetch_model.py      optional: download the MiniLM model into models/
scripts/build_embeddings.py optional: build data/catalog.embeddings.npy
scripts/regenerate_results.sh  recreate the headline results for the current commit
tests/                      organizer tests, replica parity, agent behaviour, slow regression
results/                    one JSON per measured configuration (score, per-scenario, misses, latency, git sha)
```

Data source: Amazon Reviews 2023 (McAuley Lab, UCSD). See `DATA_ATTRIBUTION.md`.
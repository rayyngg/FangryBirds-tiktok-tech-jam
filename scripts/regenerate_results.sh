#!/usr/bin/env bash
# Re-create the headline results files for the current commit:
#   results/<sha>-final-default.json      full top-10 lists (AGENT_CONFIDENCE_GATE=0)
#   results/<sha>-final-gate-on.json      confidence gate (default)
#   results/<sha>-perturb.json / .md      robustness study
#   results/<sha>-offline-clean-clone.json  real git clone of HEAD, fresh venv from requirements.txt only,
#                                          empty HF_HOME, offline env, no OPENAI_API_KEY, no .env
# Run after committing so the file names carry a clean sha (the wrapper inserts "-dirty-" otherwise).
# The clone step needs network once for `pip install -r requirements.txt`; the evaluation itself is offline.
# Takes about three minutes.  Usage: scripts/regenerate_results.sh [--clone-only]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export AGENT_USE_LLM=0 TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python3}"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "warning: tracked files are modified; main-tree results will be suffixed -dirty and the clone will not contain those changes" >&2
fi

if [ "${1:-}" != "--clone-only" ]; then
  echo "== main tree: final-default (gate off)"
  AGENT_CONFIDENCE_GATE=0 "$PY" scripts/run_eval.py --label final-default
  echo "== main tree: final-gate-on"
  AGENT_CONFIDENCE_GATE=1 "$PY" scripts/run_eval.py --label final-gate-on
  echo "== main tree: perturbation study"
  "$PY" scripts/perturb_eval.py --configs base,gate
fi

echo "== offline clean clone"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse --short HEAD)"
git clone -q --branch "$BRANCH" "$ROOT" "$TMP/clone"
"$PY" -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q -r "$TMP/clone/requirements.txt"
mkdir -p "$TMP/hf"
(
  cd "$TMP/clone"
  [ -f data/catalog.jsonl ] || gzip -dk data/catalog.jsonl.gz
  [ ! -f .env ] || { echo "unexpected .env in clone" >&2; exit 1; }
  env -u OPENAI_API_KEY HF_HOME="$TMP/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AGENT_USE_LLM=0 \
    "$TMP/venv/bin/python" scripts/run_eval.py --label offline-clean-clone
)
if [ -n "$(ls -A "$TMP/hf")" ]; then
  echo "error: HF_HOME was written to during the offline run" >&2
  exit 1
fi
cp "$TMP/clone/results/${SHA}-offline-clean-clone.json" "$ROOT/results/"
echo "copied results/${SHA}-offline-clean-clone.json (HF_HOME stayed empty)"

#!/usr/bin/env bash
# Re-create the headline results files for the current commit (run after committing, so the file
# names carry a clean sha instead of the "<sha>-dirty-" prefix). Takes about two minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
export AGENT_USE_LLM=0 TOKENIZERS_PARALLELISM=false
AGENT_CONFIDENCE_GATE=0 python3 scripts/run_eval.py --label final-default
AGENT_CONFIDENCE_GATE=1 python3 scripts/run_eval.py --label final-gate-on
python3 scripts/perturb_eval.py --configs base,gate
echo "offline clean-clone run: see README 'Offline mode' (git clone + empty HF_HOME), or run:"
echo "  HF_HOME=\$(mktemp -d) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_eval.py --label offline-clean-clone"

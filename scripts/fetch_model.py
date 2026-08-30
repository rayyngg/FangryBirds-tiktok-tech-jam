#!/usr/bin/env python3
"""Download the optional sentence-transformer into ``models/`` for the dense tier (needs network once).

The agent never needs this: by default it runs lexical/structural retrieval only. Run this script
on a machine with network access if you want to evaluate with ``AGENT_USE_EMBEDDINGS=1``; the model
directory is gitignored and loaded strictly from local files afterwards.

    python3 scripts/fetch_model.py                      # -> models/all-MiniLM-L6-v2
    python3 scripts/build_embeddings.py                 # -> data/catalog.embeddings.npy (optional cache)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the optional embedding model into models/")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--target", default=str(ROOT / "models" / "all-MiniLM-L6-v2"))
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is not installed (pip install -r requirements.txt)")
    model = SentenceTransformer(args.model, device="cpu")
    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.target)
    print(f"saved {args.model} to {args.target}")


if __name__ == "__main__":
    main()

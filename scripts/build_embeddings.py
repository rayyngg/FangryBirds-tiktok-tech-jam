#!/usr/bin/env python3
"""Encode the catalog once into ``data/catalog.embeddings.npy`` for the optional dense tier.

The agent never encodes the catalog at construction time (it would take minutes on a CPU grader);
it only memory-maps this cache when ``AGENT_USE_EMBEDDINGS=1`` and the row count matches the catalog.
The cache is gitignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the catalog embedding cache")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import os

    os.environ["AGENT_USE_EMBEDDINGS"] = "1"
    from starter import embedder, retrieval

    model = embedder.load_embedder()
    if model is None:
        sys.exit("no local model available; run scripts/fetch_model.py first")
    import numpy as np

    texts: list[str] = []
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                product = json.loads(line)
                texts.append(" ".join(retrieval._text(product.get(field)) for field in ("title", "categories", "features", "details", "description")))
    matrix = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    target = Path(args.catalog).with_name("catalog.embeddings.npy")
    np.save(target, matrix.astype(np.float32))
    print(f"wrote {target} {matrix.shape}")


if __name__ == "__main__":
    main()

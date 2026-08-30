"""Optional dense tier: a sentence-transformer loaded strictly from local files.

Off by default (``AGENT_USE_EMBEDDINGS=0``). When enabled the model is loaded from
``AGENT_MODEL_PATH`` / ``models/all-MiniLM-L6-v2`` / the local Hugging Face cache with
``local_files_only=True``; the hub is never contacted, and any failure returns ``None`` so the agent
runs lexical-only. The catalog embedding matrix is only read from a cache file
(``data/catalog.embeddings.npy``) — the catalog is never encoded at construction time.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default every Hugging Face client to offline mode *before* the libraries are imported, so a
# missing model fails fast instead of waiting on network retries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "all-MiniLM-L6-v2"


def flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


def load_embedder(model_path: str | Path | None = None):
    """Return a SentenceTransformer or None; never raises, never touches the network."""
    if not flag("AGENT_USE_EMBEDDINGS", "0"):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    device = os.getenv("AGENT_EMBED_DEVICE", "cpu")
    directories = [Path(p) for p in (model_path, os.getenv("AGENT_MODEL_PATH"), DEFAULT_MODEL_PATH) if p]
    for directory in directories:
        if (directory / "modules.json").exists():
            try:
                return SentenceTransformer(str(directory), device=device, local_files_only=True)
            except Exception:
                continue
    try:
        return SentenceTransformer(os.getenv("AGENT_SENTENCE_MODEL", DEFAULT_MODEL_NAME), device=device, local_files_only=True)
    except Exception:
        return None


def load_cached_embeddings(cache_path: Path, expected_rows: int):
    """Memory-map a cached (rows x dim) matrix if it exists and matches the catalog; else None."""
    try:
        import numpy as np

        if not cache_path.exists():
            return None
        matrix = np.load(cache_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
            return None
        return matrix
    except Exception:
        return None


def encode_query(model, text: str):
    try:
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    except Exception:
        return None

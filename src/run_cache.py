"""
Per-run result caches that are invalidated automatically when the run's
model outputs change.

A cache file is reused only if the fingerprint stored next to it matches the
current fingerprint of the run directory (config.json plus every *_logits.npy).
Retraining a run therefore never reuses numbers computed from the old model.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

MANIFEST = "cache_manifest.json"


def run_fingerprint(run_dir: Path) -> str:
    h = hashlib.sha1()
    for f in [run_dir / "config.json", *sorted(run_dir.glob("*_logits.npy"))]:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_cache(run_dir: Path, name: str, fingerprint: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = run_dir / name
    if not path.exists():
        return None
    fingerprint = fingerprint or run_fingerprint(run_dir)
    if _read_manifest(run_dir).get(name) != fingerprint:
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return None


def save_cache(run_dir: Path, name: str, df: pd.DataFrame, fingerprint: Optional[str] = None) -> None:
    df.to_csv(run_dir / name, index=False)
    manifest = _read_manifest(run_dir)
    manifest[name] = fingerprint or run_fingerprint(run_dir)
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

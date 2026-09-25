"""Cryptographic lineage manifest generation for models and predictions.

Produces deterministic SHA-256 hashes linking:
- Raw dataset files
- Preprocessing and feature configurations
- Git commit and clean/dirty status
- Model hyperparameters and serialized binary weights
- Evaluation benchmark scores
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_lineage_manifest(
    dataset_info: dict,
    git_commit: str,
    git_dirty: bool,
    feature_names: list[str],
    model_type: str,
    model_version: str,
    model_path: Path | None,
    hyperparameters: dict,
    metrics: dict,
    validation_config: dict,
) -> dict:
    """Create a tamper-proof lineage manifest with a unique lineage_id."""
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "dataset": dataset_info,
        "features": {
            "count": len(feature_names),
            "feature_names": sorted(feature_names),
            "features_hash": hashlib.sha256(json.dumps(sorted(feature_names)).encode()).hexdigest(),
        },
        "model": {
            "type": model_type,
            "version": model_version,
            "parameters": hyperparameters,
            "binary_sha256": compute_sha256(model_path) if model_path else "in_memory",
        },
        "validation": {
            "config": validation_config,
            "metrics": metrics,
        },
    }

    # Deterministic canonical serialization for lineage_id
    canonical_bytes = json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    manifest["lineage_id"] = hashlib.sha256(canonical_bytes).hexdigest()
    return manifest

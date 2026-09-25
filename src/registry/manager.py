"""Central registry manager for production demand forecasting models.

Handles atomic registration, version management (v1, v2, ...), lineage linkage,
promotion gating, and model loading.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd

from src.lineage.manifest import create_lineage_manifest
from src.registry.promotion_gate import evaluate_promotion

LOG = logging.getLogger("registry_manager")
ROOT = Path(__file__).resolve().parent.parent.parent


class ModelRegistry:
    def __init__(self, registry_dir: Path | None = None):
        self.registry_dir = registry_dir or (ROOT / "registry")
        self.models_dir = self.registry_dir / "models"
        self.registry_json_path = self.registry_dir / "registry.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_registry(self) -> dict:
        if self.registry_json_path.exists():
            return json.loads(self.registry_json_path.read_text(encoding="utf-8"))
        return {
            "active_version": None,
            "models": {},
            "promotion_history": [],
        }

    def save_registry(self, registry_data: dict) -> None:
        temp_path = self.registry_json_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(registry_data, indent=2, default=str), encoding="utf-8")
        temp_path.replace(self.registry_json_path)

    def get_active_model_version(self) -> str | None:
        reg = self.load_registry()
        return reg.get("active_version")

    def get_next_version(self) -> str:
        reg = self.load_registry()
        existing = [k for k in reg.get("models", {}).keys() if k.startswith("v") and k[1:].isdigit()]
        if not existing:
            return "v1"
        highest = max(int(k[1:]) for k in existing)
        return f"v{highest + 1}"

    def register_candidate(
        self,
        model_object: object,
        features: list[str],
        metrics: dict[str, float],
        dataset_info: dict,
        validation_config: dict,
        hyperparameters: dict,
        git_commit: str,
        git_dirty: bool,
        sku_segments: pd.DataFrame | None = None,
        promotion_config: dict | None = None,
        model_type: str = "lightgbm",
    ) -> dict:
        """Register a trained model, run the promotion gate, and update registry."""
        registry = self.load_registry()
        current_active = registry.get("active_version")
        next_ver = self.get_next_version()

        version_dir = self.models_dir / next_ver
        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model binary
        model_binary_path = version_dir / "model.joblib"
        joblib.dump(model_object, model_binary_path)

        # 2. Save feature list
        (version_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

        # 3. Save metrics
        (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # 4. Save sku_segments if provided
        if sku_segments is not None:
            sku_segments.to_csv(version_dir / "sku_segments.csv", index=False)

        # 5. Create lineage manifest
        lineage = create_lineage_manifest(
            dataset_info=dataset_info,
            git_commit=git_commit,
            git_dirty=git_dirty,
            feature_names=features,
            model_type=model_type,
            model_version=next_ver,
            model_path=model_binary_path,
            hyperparameters=hyperparameters,
            metrics=metrics,
            validation_config=validation_config,
        )
        (version_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")

        # 6. Metadata
        metadata = {
            "version": next_ver,
            "lineage_id": lineage["lineage_id"],
            "model_type": model_type,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "metrics": metrics,
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # 7. Evaluate Promotion Gate
        if current_active is None:
            promotion_result = {
                "decision": "PROMOTE",
                "summary_reason": "First model registered",
                "checks": {"first_model": True},
            }
        else:
            incumbent_metrics = registry["models"][current_active]["metrics"]
            promotion_result = evaluate_promotion(
                incumbent_metrics=incumbent_metrics,
                candidate_metrics=metrics,
                promotion_config=promotion_config,
            )

        promoted = promotion_result["decision"] == "PROMOTE"

        # 8. Update registry record
        registry["models"][next_ver] = {
            "registered_at": metadata["registered_at"],
            "status": "active" if promoted else "candidate_rejected",
            "metrics": metrics,
            "lineage_id": lineage["lineage_id"],
            "model_type": model_type,
            "path": str(version_dir),
        }

        if promoted:
            if current_active and current_active in registry["models"]:
                registry["models"][current_active]["status"] = "archived"
            registry["active_version"] = next_ver
            registry["promotion_history"].append({
                "timestamp": metadata["registered_at"],
                "from_version": current_active,
                "to_version": next_ver,
                "reason": promotion_result.get("summary_reason", "Passed promotion gate"),
                "metrics_delta": {
                    k: metrics[k] - (registry["models"][current_active]["metrics"].get(k, 0) if current_active else 0)
                    for k in metrics if isinstance(metrics[k], (int, float))
                },
            })

        self.save_registry(registry)

        return {
            "registered_version": next_ver,
            "promoted": promoted,
            "promotion_result": promotion_result,
            "lineage_id": lineage["lineage_id"],
        }

    def load_active_model_bundle(self) -> dict:
        """Load currently active model bundle including binary, features, and segments."""
        registry = self.load_registry()
        active = registry.get("active_version")
        if not active or active not in registry.get("models", {}):
            raise FileNotFoundError("No active model found in registry.")

        model_dir = self.models_dir / active
        if not model_dir.exists():
            raise FileNotFoundError(f"Active model directory {model_dir} does not exist.")

        # Load model binary (support both model.joblib and legacy lgb_model.pkl)
        model_file = model_dir / "model.joblib"
        if not model_file.exists():
            model_file = model_dir / "lgb_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model binary not found in {model_dir}")

        model_obj = joblib.load(model_file)
        features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))

        sku_segments = None
        seg_file = model_dir / "sku_segments.csv"
        if seg_file.exists():
            sku_segments = pd.read_csv(seg_file)

        lineage_id = registry["models"][active].get("lineage_id", "unknown")
        if (model_dir / "lineage.json").exists():
            lineage = json.loads((model_dir / "lineage.json").read_text(encoding="utf-8"))
            lineage_id = lineage.get("lineage_id", lineage_id)

        return {
            "version": active,
            "model": model_obj,
            "features": features,
            "segments": sku_segments,
            "lineage_id": lineage_id,
            "metrics": registry["models"][active].get("metrics", {}),
        }

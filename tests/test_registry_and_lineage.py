import numpy as np
import pandas as pd
import pytest

from src.registry.manager import ModelRegistry


class DummyModel:
    def predict(self, X):
        return np.ones(len(X))


def test_registry_lifecycle_and_promotion(tmp_path):
    reg = ModelRegistry(registry_dir=tmp_path)
    assert reg.get_active_model_version() is None

    # 1. Register first model (should auto-promote)
    res1 = reg.register_candidate(
        model_object=DummyModel(),
        features=["f1", "f2"],
        metrics={"MAE": 1.5, "RMSE": 2.5, "WAPE": 0.5, "Bias": 0.05},
        dataset_info={"source": "M5"},
        validation_config={"folds": 3},
        hyperparameters={"lr": 0.05},
        git_commit="abcdef",
        git_dirty=False,
    )
    assert res1["promoted"] is True
    assert res1["registered_version"] == "v1"
    assert reg.get_active_model_version() == "v1"

    bundle1 = reg.load_active_model_bundle()
    assert bundle1["version"] == "v1"
    assert bundle1["features"] == ["f1", "f2"]
    assert "lineage_id" in bundle1

    # 2. Register degraded candidate (should reject)
    res2 = reg.register_candidate(
        model_object=DummyModel(),
        features=["f1", "f2"],
        metrics={"MAE": 1.6, "RMSE": 2.7, "WAPE": 0.55, "Bias": 0.08},
        dataset_info={"source": "M5"},
        validation_config={"folds": 3},
        hyperparameters={"lr": 0.05},
        git_commit="abcdef",
        git_dirty=False,
        promotion_config={"min_mae_improvement_pct": 5.0},
    )
    assert res2["promoted"] is False
    assert res2["registered_version"] == "v2"
    # Active version should remain v1
    assert reg.get_active_model_version() == "v1"

    # 3. Register superior candidate (should promote to v3)
    res3 = reg.register_candidate(
        model_object=DummyModel(),
        features=["f1", "f2"],
        metrics={"MAE": 1.3, "RMSE": 2.3, "WAPE": 0.42, "Bias": 0.01},
        dataset_info={"source": "M5"},
        validation_config={"folds": 3},
        hyperparameters={"lr": 0.05},
        git_commit="abcdef",
        git_dirty=False,
        promotion_config={"min_mae_improvement_pct": 5.0},
    )
    assert res3["promoted"] is True
    assert res3["registered_version"] == "v3"
    assert reg.get_active_model_version() == "v3"

    bundle3 = reg.load_active_model_bundle()
    assert bundle3["version"] == "v3"
    assert bundle3["metrics"]["MAE"] == 1.3

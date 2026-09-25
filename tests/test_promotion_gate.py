import pytest
from src.registry.promotion_gate import evaluate_promotion


def test_promotion_gate_improved_candidate():
    incumbent = {"MAE": 1.20, "RMSE": 2.40, "WAPE": 0.45, "Bias": 0.05}
    candidate = {"MAE": 1.10, "RMSE": 2.35, "WAPE": 0.42, "Bias": 0.02}

    result = evaluate_promotion(incumbent, candidate, {"min_mae_improvement_pct": 5.0})
    assert result["decision"] == "PROMOTE"
    assert result["checks"]["mae_improvement"] is True
    assert result["checks"]["rmse_guardrail"] is True
    assert result["checks"]["wape_guardrail"] is True
    assert result["checks"]["bias_guardrail"] is True


def test_promotion_gate_rejects_insufficient_mae():
    incumbent = {"MAE": 1.20, "RMSE": 2.40, "WAPE": 0.45, "Bias": 0.05}
    candidate = {"MAE": 1.19, "RMSE": 2.35, "WAPE": 0.44, "Bias": 0.02}

    # Requires 5% improvement, but achieved only ~0.83%
    result = evaluate_promotion(incumbent, candidate, {"min_mae_improvement_pct": 5.0})
    assert result["decision"] == "REJECT"
    assert result["checks"]["mae_improvement"] is False
    assert result["checks"]["rmse_guardrail"] is True


def test_promotion_gate_rejects_rmse_degradation():
    incumbent = {"MAE": 1.20, "RMSE": 2.40, "WAPE": 0.45, "Bias": 0.05}
    candidate = {"MAE": 1.05, "RMSE": 2.65, "WAPE": 0.40, "Bias": 0.02}

    # Strict zero RMSE degradation
    result = evaluate_promotion(incumbent, candidate, {"max_rmse_degradation_pct": 0.0})
    assert result["decision"] == "REJECT"
    assert result["checks"]["mae_improvement"] is True
    assert result["checks"]["rmse_guardrail"] is False


def test_promotion_gate_allows_configurable_rmse_tolerance():
    incumbent = {"MAE": 1.20, "RMSE": 2.40, "WAPE": 0.45, "Bias": 0.05}
    candidate = {"MAE": 1.05, "RMSE": 2.42, "WAPE": 0.40, "Bias": 0.02}

    # 1.0% RMSE tolerance allows 2.40 -> 2.424
    result = evaluate_promotion(incumbent, candidate, {"max_rmse_degradation_pct": 1.0})
    assert result["decision"] == "PROMOTE"


def test_promotion_gate_rejects_severe_bias():
    incumbent = {"MAE": 1.20, "RMSE": 2.40, "WAPE": 0.45, "Bias": 0.05}
    candidate = {"MAE": 1.10, "RMSE": 2.35, "WAPE": 0.42, "Bias": 0.25}

    # Bias exceeds max allowed 0.15
    result = evaluate_promotion(incumbent, candidate, {"max_absolute_bias": 0.15})
    assert result["decision"] == "REJECT"
    assert result["checks"]["bias_guardrail"] is False

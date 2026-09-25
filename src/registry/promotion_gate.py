"""Deterministic and configurable model promotion gate.

Evaluates candidate models against active incumbents across multiple guardrails:
1. Primary metric: MAE improvement >= configured threshold
2. RMSE guardrail: candidate RMSE <= incumbent RMSE * (1 + max_rmse_degradation_pct / 100)
3. WAPE guardrail: candidate WAPE <= incumbent WAPE * (1 + max_wape_degradation_pct / 100)
4. Bias guardrail: abs(candidate Bias) <= max_absolute_bias
"""

from __future__ import annotations

import logging

LOG = logging.getLogger("promotion_gate")


def evaluate_promotion(
    incumbent_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    promotion_config: dict | None = None,
) -> dict:
    """Evaluate candidate model against incumbent model using configurable rules.

    Args:
        incumbent_metrics: Dict with MAE, RMSE, WAPE, and optionally Bias.
        candidate_metrics: Dict with MAE, RMSE, WAPE, and optionally Bias.
        promotion_config: Dict containing thresholds:
            - min_mae_improvement_pct (default: 0.0)
            - max_rmse_degradation_pct (default: 0.0)
            - max_wape_degradation_pct (default: 0.0)
            - max_absolute_bias (default: 0.20)

    Returns:
        dict: Decision ("PROMOTE" or "REJECT"), detailed reasons, and improvement fractions.
    """
    config = promotion_config or {}
    min_mae_imp = float(config.get("min_mae_improvement_pct", config.get("promotion_min_improvement", 0.0)))
    max_rmse_deg = float(config.get("max_rmse_degradation_pct", 0.0))
    max_wape_deg = float(config.get("max_wape_degradation_pct", 0.0))
    max_abs_bias = float(config.get("max_absolute_bias", 0.20))

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. MAE check
    inc_mae, cand_mae = incumbent_metrics["MAE"], candidate_metrics["MAE"]
    mae_improvement_pct = ((inc_mae - cand_mae) / inc_mae) * 100.0 if inc_mae > 0 else 0.0
    mae_pass = mae_improvement_pct >= min_mae_imp
    checks["mae_improvement"] = mae_pass
    if not mae_pass:
        reasons.append(f"MAE improvement ({mae_improvement_pct:+.2f}%) below minimum requirement ({min_mae_imp:.2f}%)")

    # 2. RMSE check
    inc_rmse, cand_rmse = incumbent_metrics["RMSE"], candidate_metrics["RMSE"]
    allowed_rmse = inc_rmse * (1.0 + max_rmse_deg / 100.0)
    rmse_pass = cand_rmse <= allowed_rmse
    checks["rmse_guardrail"] = rmse_pass
    if not rmse_pass:
        reasons.append(f"RMSE ({cand_rmse:.4f}) degraded beyond allowed limit ({allowed_rmse:.4f})")

    # 3. WAPE check (if provided)
    if "WAPE" in incumbent_metrics and "WAPE" in candidate_metrics:
        inc_wape, cand_wape = incumbent_metrics["WAPE"], candidate_metrics["WAPE"]
        allowed_wape = inc_wape * (1.0 + max_wape_deg / 100.0)
        wape_pass = cand_wape <= allowed_wape
        checks["wape_guardrail"] = wape_pass
        if not wape_pass:
            reasons.append(f"WAPE ({cand_wape:.4f}) degraded beyond allowed limit ({allowed_wape:.4f})")

    # 4. Bias check (if provided)
    if "Bias" in candidate_metrics:
        cand_bias = candidate_metrics["Bias"]
        bias_pass = abs(cand_bias) <= max_abs_bias
        checks["bias_guardrail"] = bias_pass
        if not bias_pass:
            reasons.append(f"Absolute Forecast Bias ({abs(cand_bias):.4f}) exceeded threshold ({max_abs_bias:.4f})")

    promote = all(checks.values())
    decision = "PROMOTE" if promote else "REJECT"

    summary_reason = "; ".join(reasons) if reasons else f"Candidate passed all gates with MAE improvement of {mae_improvement_pct:+.2f}%"

    return {
        "decision": decision,
        "summary_reason": summary_reason,
        "checks": checks,
        "mae_improvement_pct": mae_improvement_pct,
        "reasons": reasons,
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
        "thresholds_used": {
            "min_mae_improvement_pct": min_mae_imp,
            "max_rmse_degradation_pct": max_rmse_deg,
            "max_wape_degradation_pct": max_wape_deg,
            "max_absolute_bias": max_abs_bias,
        },
    }

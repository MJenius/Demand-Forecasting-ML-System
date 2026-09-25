"""Delayed Actuals Reconciler for Closed-Loop Monitoring.

Pairs incoming delayed actual retail sales with previously logged predictions
in SQLite or JSONL, updating `actual_value` and calculating ground-truth
operational error metrics:
- Rolling MAE
- Rolling RMSE
- Rolling WAPE
- Rolling Forecast Bias
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_forecasts

ROOT = Path(__file__).resolve().parent.parent.parent


class ActualsReconciler:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else (ROOT / "monitoring" / "predictions.db")

    def reconcile_from_dataframe(self, actuals_df: pd.DataFrame) -> dict:
        """Reconcile predictions in SQLite with actual sales DataFrame.

        Expected actuals_df columns:
        - item_id (str)
        - store_id (str)
        - date (str or datetime: YYYY-MM-DD)
        - sales (float/int)
        """
        if not self.db_path.exists():
            return {"updated_records": 0, "message": "Database does not exist."}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updated_count = 0
        df = actuals_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        for row in df.itertuples():
            item_id = str(row.item_id)
            store_id = str(row.store_id)
            date_str = str(row.date)
            sales_val = float(getattr(row, "sales", getattr(row, "target", 0.0)))

            cursor.execute(
                """
                UPDATE predictions
                SET actual_value = ?
                WHERE item_id = ? AND store_id = ? AND date = ? AND (actual_value IS NULL OR actual_value != ?)
                """,
                (sales_val, item_id, store_id, date_str, sales_val),
            )
            updated_count += cursor.rowcount

        conn.commit()
        conn.close()

        return {
            "updated_records": updated_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def compute_operational_performance(
        self,
        model_version: str | None = None,
        days: int = 30,
    ) -> dict[str, float | str | int]:
        """Compute live operational forecast performance over the last N days with actuals."""
        if not self.db_path.exists():
            return {"status": "NO_DATABASE", "count": 0}

        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT prediction, actual_value, date, model_version, item_id, store_id
            FROM predictions
            WHERE actual_value IS NOT NULL
        """
        params = []
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return {"status": "NO_RECONCILED_ACTUALS", "count": 0}

        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=days)
        recent = df[df["date"] >= cutoff]

        if recent.empty:
            recent = df

        m = evaluate_forecasts(recent["actual_value"], recent["prediction"])

        return {
            "status": "OK",
            "model_version": model_version or "ALL",
            "window_days": days,
            "count": len(recent),
            "start_date": str(recent["date"].min().date()),
            "end_date": str(recent["date"].max().date()),
            **m,
        }

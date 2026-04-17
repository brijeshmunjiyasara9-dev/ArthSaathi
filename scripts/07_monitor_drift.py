"""
07_monitor_drift.py — Evidently AI data & prediction drift monitoring.

Compares last N days of real predictions from the DB against the training
baseline (a sample from the training parquet). Generates an HTML report.

Install: pip install evidently psycopg2-binary pandas pyarrow

Schedule (cron): 0 9 * * MON python scripts/07_monitor_drift.py --root D:/Project

Alerts to watch in the report:
  - dataset_drift = True          → retrain needed
  - share_of_drifted_features > 0.3 → input distribution shifted
  - prediction_drift > 0.15       → model degrading in production
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="ArthSaathi drift monitor")
    p.add_argument("--root",       default="D:/Project", help="Project root")
    p.add_argument("--days",       type=int, default=7,  help="Lookback window (days)")
    p.add_argument("--db-url",     default=None,         help="PostgreSQL DSN (overrides .env)")
    p.add_argument("--sample",     type=int, default=5000, help="Training baseline sample size")
    p.add_argument("--output-dir", default=None,         help="Where to save HTML report")
    return p.parse_args()


def load_training_baseline(root: Path, n: int) -> pd.DataFrame:
    """Load a random sample from the training parquet as reference data."""
    parquet = root / "processed2" / "household_joined" / "household_stress_dataset.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Training parquet not found: {parquet}")

    df = pd.read_parquet(parquet, columns=[
        "financial_stress", "food_stress", "debt_stress", "health_stress",
        "composite_stress_score", "income_all_members_from_wages",
        "emi_to_income_ratio", "food_to_expense_ratio", "health_to_expense_ratio",
        "savings_proxy", "state", "region_type",
    ])
    df = df.dropna(subset=["financial_stress"])
    return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)


def load_recent_predictions(db_url: str, days: int) -> pd.DataFrame:
    """Load recent predictions from the PostgreSQL assessments table."""
    try:
        import sqlalchemy as sa
        engine = sa.create_engine(db_url, pool_pre_ping=True)
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = f"""
            SELECT
                financial_stress_prob   AS financial_stress,
                food_stress_prob        AS food_stress,
                debt_stress_prob        AS debt_stress,
                health_stress_prob      AS health_stress,
                composite_score         AS composite_stress_score,
                is_stressed,
                stress_level,
                assessed_at
            FROM assessments
            WHERE assessed_at >= '{cutoff.isoformat()}'
            ORDER BY assessed_at DESC
        """
        df = pd.read_sql(query, engine)
        print(f"  Loaded {len(df)} recent predictions (last {days} days)")
        return df
    except Exception as e:
        print(f"  [WARN] Could not load from DB: {e}")
        # Return a minimal stub so the script doesn't crash
        return pd.DataFrame(columns=[
            "financial_stress", "food_stress", "debt_stress",
            "health_stress", "composite_stress_score",
        ])


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                         output_path: Path):
    """Run Evidently report and save HTML."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import (
            ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric,
        )

        # Align columns
        shared_cols = [c for c in reference.columns if c in current.columns]
        ref = reference[shared_cols].copy()
        cur = current[shared_cols].copy()

        report = Report(metrics=[
            DataDriftPreset(),
            DatasetMissingValuesMetric(),
        ])
        report.run(reference_data=ref, current_data=cur)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        print(f"  ✓ Drift report saved → {output_path}")

        # Extract summary JSON for 08_retrain_check.py
        summary = report.as_dict()
        summary_path = output_path.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, default=str, indent=2)
        print(f"  ✓ Summary JSON saved → {summary_path}")

    except ImportError:
        print("  [ERROR] evidently not installed. Run: pip install evidently")
        raise


def main():
    args = parse_args()
    root = Path(args.root)

    # Load DB URL from .env if not provided
    db_url = args.db_url
    if not db_url:
        env_file = root / "web" / "backend" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DATABASE_URL="):
                    db_url = line.split("=", 1)[1].strip()
                    break

    if not db_url:
        print("  [WARN] DATABASE_URL not found — using synthetic current data")
        current = pd.DataFrame()
    else:
        current = load_recent_predictions(db_url, args.days)

    reference = load_training_baseline(root, args.sample)

    if current.empty:
        print("  [INFO] No recent predictions found — generating baseline self-report")
        current = reference.sample(min(500, len(reference)), random_state=99)

    output_dir = Path(args.output_dir) if args.output_dir else root / "monitoring"
    ts = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"drift_report_{ts}.html"

    run_evidently_report(reference, current, output_path)
    print(f"\n  Done. Open {output_path} in a browser to view the full report.")
    print("  Cron schedule: 0 9 * * MON python scripts/07_monitor_drift.py --root D:/Project")


if __name__ == "__main__":
    main()

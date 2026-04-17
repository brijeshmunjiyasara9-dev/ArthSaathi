"""
08_retrain_check.py — Automated retraining trigger for ArthSaathi.

Checks drift score from the latest Evidently JSON report and the live
stress rate in the DB. Triggers retraining (or prints instructions) if:
  - dataset_drift_share > 0.30
  - |current_stress_rate - baseline_stress_rate| > 0.15

Install: pip install sqlalchemy psycopg2-binary pandas

Schedule (cron): 0 6 * * * python scripts/08_retrain_check.py --root D:/Project
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASELINE_STRESS_RATE = 0.42   # average from training data
DRIFT_THRESHOLD      = 0.30   # max acceptable drift share
RATE_SHIFT_THRESHOLD = 0.15   # max acceptable stress rate shift


def parse_args():
    p = argparse.ArgumentParser(description="ArthSaathi retraining trigger")
    p.add_argument("--root",    default="D:/Project", help="Project root")
    p.add_argument("--db-url",  default=None,         help="PostgreSQL DSN")
    p.add_argument("--days",    type=int, default=30,  help="Lookback window (days)")
    p.add_argument("--dry-run", action="store_true",   help="Print decision without triggering")
    return p.parse_args()


def load_db_url(root: Path, override: str = None) -> str:
    if override:
        return override
    env_file = root / "web" / "backend" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    return None


def load_latest_drift_json(root: Path) -> dict:
    """Load the most recent drift report JSON from monitoring/."""
    monitoring = root / "monitoring"
    jsons = sorted(monitoring.glob("drift_report_*.json"), reverse=True)
    if not jsons:
        print("  [INFO] No drift report found — run 07_monitor_drift.py first")
        return {}
    j = jsons[0]
    print(f"  Reading drift report: {j.name}")
    with open(j) as f:
        return json.load(f)


def get_drift_share(report: dict) -> float:
    """Extract share_of_drifted_features from Evidently report dict."""
    try:
        for metric in report.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                return float(metric["result"].get("share_of_drifted_columns", 0))
    except Exception:
        pass
    return 0.0


def get_recent_stress_rate(db_url: str, days: int) -> float:
    """Query the DB for the stress rate over the last N days."""
    try:
        import sqlalchemy as sa
        engine = sa.create_engine(db_url, pool_pre_ping=True)
        cutoff = datetime.utcnow() - timedelta(days=days)
        with engine.connect() as conn:
            result = conn.execute(sa.text(
                f"SELECT AVG(CASE WHEN is_stressed THEN 1.0 ELSE 0.0 END) "
                f"FROM assessments WHERE assessed_at >= :cutoff"
            ), {"cutoff": cutoff})
            row = result.fetchone()
            rate = float(row[0]) if row and row[0] is not None else None
            if rate is None:
                print("  [INFO] No assessments in window yet")
                return BASELINE_STRESS_RATE
            return rate
    except Exception as e:
        print(f"  [WARN] Could not query DB: {e}")
        return BASELINE_STRESS_RATE


def trigger_retraining(root: Path, dry_run: bool):
    """Print retraining command or execute it."""
    cmd = f"venv\\Scripts\\python.exe scripts/03_train_models.py --root {root}"
    print(f"\n  🔁 RETRAINING TRIGGERED\n  Command: {cmd}")
    if dry_run:
        print("  [DRY RUN] Not executing")
        return
    try:
        subprocess.run(cmd, shell=True, cwd=str(root), check=True)
        print("  ✓ Retraining completed")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Retraining failed: {e}")


def main():
    args = parse_args()
    root = Path(args.root)

    print(f"\n{'='*55}")
    print(f"  ArthSaathi Retrain Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    # ── 1. Drift check ────────────────────────────────────────────────────────
    drift_report = load_latest_drift_json(root)
    drift_share  = get_drift_share(drift_report)
    print(f"  Dataset drift share:  {drift_share:.2%}  (threshold: {DRIFT_THRESHOLD:.0%})")

    # ── 2. Stress rate check ──────────────────────────────────────────────────
    db_url = load_db_url(root, args.db_url)
    if db_url:
        current_rate = get_recent_stress_rate(db_url, args.days)
    else:
        print("  [WARN] No DATABASE_URL — skipping stress rate check")
        current_rate = BASELINE_STRESS_RATE

    rate_shift = abs(current_rate - BASELINE_STRESS_RATE)
    print(f"  Current stress rate: {current_rate:.2%}  (baseline: {BASELINE_STRESS_RATE:.0%})")
    print(f"  Stress rate shift:   {rate_shift:.2%}  (threshold: {RATE_SHIFT_THRESHOLD:.0%})")

    # ── 3. Decision ───────────────────────────────────────────────────────────
    reasons = []
    if drift_share > DRIFT_THRESHOLD:
        reasons.append(f"dataset drift {drift_share:.2%} > {DRIFT_THRESHOLD:.0%}")
    if rate_shift > RATE_SHIFT_THRESHOLD:
        reasons.append(f"stress rate shift {rate_shift:.2%} > {RATE_SHIFT_THRESHOLD:.0%}")

    if reasons:
        print(f"\n  ⚠  Retraining needed: {'; '.join(reasons)}")
        trigger_retraining(root, args.dry_run)
    else:
        print("\n  ✅ Models are healthy — no retraining needed")

    print()


if __name__ == "__main__":
    main()

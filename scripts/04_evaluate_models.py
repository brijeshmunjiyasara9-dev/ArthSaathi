"""
04_evaluate_models.py  (v4)
---------------------------
Step 4: Evaluate trained models with temporal split + composite per-class metrics.

Fixes:
  v3: Time-based split, composite per-class confusion matrix
  v4: CHE label override (health_stress = health_to_expense_ratio > 0.10)
      so evaluation is against the same label the v4 model was trained on.
      Threshold lookup updated for v4 nested metadata structure.

Usage:
    python scripts/04_evaluate_models.py --root D:/Project
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import train_test_split

STRESS_LABELS    = ["financial_stress", "food_stress", "debt_stress", "health_stress"]
COMPOSITE_LABEL  = "composite_stress_score"
MAX_SAMPLE       = 500_000
TEMPORAL_CUTOFF  = pd.Timestamp("2023-01-01")


def temporal_split_df(df, feature_cols, label):
    """Return X_train, X_test, y_train, y_test using month_slot cutoff."""
    if "month_slot" in df.columns:
        ms = pd.to_datetime(df["month_slot"], errors="coerce")
        tr = ms <  TEMPORAL_CUTOFF
        te = ms >= TEMPORAL_CUTOFF
        if tr.sum() >= 100 and te.sum() >= 20:
            return (df.loc[tr, feature_cols], df.loc[te, feature_cols],
                    df.loc[tr, label].astype(int), df.loc[te, label].astype(int))
    print("    [WARN] Temporal split insufficient — using random 80/20 fallback")
    return train_test_split(df[feature_cols], df[label].astype(int),
                            test_size=0.2, random_state=42, stratify=df[label].astype(int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/Project")
    args  = parser.parse_args()

    root      = Path(args.root)
    model_dir = root / "models"
    data_path = root / "processed2" / "household_joined" / "household_stress_dataset.parquet"

    meta_path = model_dir / "feature_metadata.json"
    if not meta_path.exists():
        print("ERROR: feature_metadata.json not found. Run 03_train_models.py first.")
        return
    with open(meta_path) as f:
        meta = json.load(f)

    feature_order  = meta["feature_order"]
    excluded_per   = meta.get("excluded_features_per_model", {})
    split_method   = meta.get("split_method", "random")
    train_cutoff   = meta.get("train_cutoff", str(TEMPORAL_CUTOFF.date()))

    # v4 nested thresholds, v3 flat fallback
    def get_threshold(label):
        thresholds = meta.get("thresholds", {})
        if label in thresholds:
            return thresholds[label].get("default", 0.5)
        return meta.get("health_stress_threshold", 0.5) if label == "health_stress" else 0.5

    # CHE label definition (v4)
    che_threshold = meta.get("health_stress_label", {}).get("threshold", 0.10)
    model_version = meta.get("model_version", "3.0")

    shared_prep_path = model_dir / "preprocessor.pkl"
    shared_prep = joblib.load(shared_prep_path) if shared_prep_path.exists() else None

    print(f"\n=== Loading dataset ===")
    import pyarrow.parquet as pq
    pf  = pq.ParquetFile(data_path)
    n_rg = pf.num_row_groups
    spg  = max(1, int(MAX_SAMPLE / max(n_rg, 1)))

    dfs = []
    for i in range(n_rg):
        chunk = pf.read_row_group(i).to_pandas()
        if len(chunk) > spg:
            chunk = chunk.sample(n=spg, random_state=42)
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    if len(df) > MAX_SAMPLE:
        df = df.sample(MAX_SAMPLE, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(df):,} rows  |  split={split_method}  cutoff={train_cutoff}")

    if "month_slot" in df.columns:
        ms = pd.to_datetime(df["month_slot"], errors="coerce")
        print(f"  Train rows (pre-cutoff) : {(ms < TEMPORAL_CUTOFF).sum():,}")
        print(f"  Test rows  (post-cutoff): {(ms >= TEMPORAL_CUTOFF).sum():,}")

    # v4: apply the same CHE label override used during training
    if model_version >= "4.0" and "health_to_expense_ratio" in df.columns:
        df["health_stress"] = (df["health_to_expense_ratio"] > che_threshold).astype(int)
        stress_cols = [c for c in ["financial_stress","food_stress","debt_stress","health_stress"]
                       if c in df.columns]
        df["composite_stress_score"] = df[stress_cols].sum(axis=1)
        print(f"  [Label override v4] health_stress = (health_to_expense_ratio > {che_threshold})")
        print(f"    Positive rate: {df['health_stress'].mean()*100:.2f}%")

    # v5: engineer new features (must match 03_train_models.py exactly)
    EPS = 1e-9
    inc_col  = next((c for c in ["income_all_members_from_wages","total_income_hh"] if c in df.columns), None)
    size_col = next((c for c in ["hh_size","household_size","num_members_hh"] if c in df.columns), None)

    df["dependents_ratio"] = (
        df[size_col] / (df[inc_col] / 10000 + EPS)
        if (inc_col and size_col) else np.nan
    )
    if "dependents_ratio" in df.columns:
        df["dependents_ratio"] = df["dependents_ratio"].clip(upper=200)

    if "savings_proxy" in df.columns and "total_expenditure_adjusted" in df.columns:
        df["savings_buffer_months"] = df["savings_proxy"] / (df["total_expenditure_adjusted"] + EPS)
    elif "savings_proxy" in df.columns and inc_col:
        df["savings_buffer_months"] = df["savings_proxy"] / (df["savings_proxy"].abs() + df[inc_col] + EPS)
    else:
        df["savings_buffer_months"] = np.nan

    if "has_health_insurance_hh_any" in df.columns and "health_to_expense_ratio" in df.columns:
        df["no_insurance_vulnerability"] = (
            (1 - df["has_health_insurance_hh_any"].fillna(0)) *
            df["health_to_expense_ratio"].fillna(0)
        )
    else:
        df["no_insurance_vulnerability"] = np.nan

    if "region_type" in df.columns and inc_col:
        is_rural  = df["region_type"].astype(str).str.upper().isin(["RURAL","VILLAGE"])
        is_low    = df[inc_col].fillna(0) < 25000
        df["rural_low_income_flag"] = (is_rural & is_low).astype(int)
    else:
        df["rural_low_income_flag"] = np.nan

    if "emi_to_income_ratio" in df.columns and "is_hospitalised_hh_any" in df.columns:
        is_sick = (
            df["is_hospitalised_hh_any"].fillna(0) +
            df.get("is_on_regular_medication_hh_any", pd.Series(0, index=df.index)).fillna(0)
        ).clip(upper=1)
        df["emi_health_interaction"] = df["emi_to_income_ratio"].fillna(0) * is_sick
    else:
        df["emi_health_interaction"] = np.nan

    # Only keep feature_order columns that actually exist
    feature_order = [f for f in feature_order if f in df.columns]
    print(f"  Feature order: {len(feature_order)} features (after v5 engineering)")

    results = {}

    # ── Binary models ──────────────────────────────────────────────────────────
    for label in STRESS_LABELS:
        model_path = model_dir / f"{label}_model.pkl"
        if not model_path.exists() or label not in df.columns:
            print(f"\n  SKIP {label}"); continue

        print(f"\n{'='*60}\n  Model: {label}\n{'='*60}")

        prep_path = model_dir / f"{label}_preprocessor.pkl"
        prep = joblib.load(prep_path) if prep_path.exists() else shared_prep

        excl        = excluded_per.get(label, [])
        model_feats = [f for f in feature_order if f not in excl]
        if excl:
            print(f"  [Leakage-fix] Excluded: {excl}")

        model = joblib.load(model_path)
        sub   = df[model_feats + [label] +
                   (["month_slot"] if "month_slot" in df.columns else [])
                   ].dropna(subset=[label])

        print(f"  Positive prevalence: {sub[label].astype(int).mean()*100:.2f}%  "
              f"({sub[label].astype(int).sum():,} / {len(sub):,})")

        _, X_test, _, y_test = temporal_split_df(sub, model_feats, label)
        print(f"  Test set size: {len(y_test):,}")

        X_test_t = prep.transform(X_test) if prep else X_test.values
        y_prob   = model.predict_proba(X_test_t)[:, 1]

        threshold = get_threshold(label)
        y_pred    = (y_prob >= threshold).astype(int)
        print(f"  Threshold: {threshold}")

        print(classification_report(y_test, y_pred, digits=4))
        cm  = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:\n{cm}")

        auc  = roc_auc_score(y_test, y_prob)
        f1   = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        print(f"  AUC={auc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}")

        results[label] = {
            "f1_macro":              round(f1, 4),
            "auc_roc":               round(auc, 4),
            "precision":             round(prec, 4),
            "recall":                round(rec, 4),
            "decision_threshold":    threshold,
            "excluded_features":     excl,
            "confusion_matrix":      cm.tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "n_test":                int(len(y_test)),
            "split_method":          split_method,
            "train_cutoff":          train_cutoff,
        }

    # ── Composite model — Issue 4 ──────────────────────────────────────────────
    comp_path = model_dir / "composite_stress_model.pkl"
    if comp_path.exists() and COMPOSITE_LABEL in df.columns:
        print(f"\n{'='*60}\n  Model: {COMPOSITE_LABEL}\n{'='*60}")

        comp_model = joblib.load(comp_path)
        sub = df[feature_order + [COMPOSITE_LABEL] +
                  (["month_slot"] if "month_slot" in df.columns else [])
                  ].dropna(subset=[COMPOSITE_LABEL])
        y = sub[COMPOSITE_LABEL].astype(float)
        X = sub[feature_order]

        # Temporal split for regression
        if "month_slot" in sub.columns:
            ms2 = pd.to_datetime(sub["month_slot"], errors="coerce")
            tr  = ms2 <  TEMPORAL_CUTOFF
            te  = ms2 >= TEMPORAL_CUTOFF
            if tr.sum() >= 100 and te.sum() >= 20:
                X_test = X[te]; y_test = y[te]
                print(f"  Temporal split: test={len(y_test):,}")
            else:
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test_t    = shared_prep.transform(X_test) if shared_prep else X_test.values
        y_pred_cont = comp_model.predict(X_test_t)
        y_pred_cls  = np.clip(np.round(y_pred_cont), 0, 4).astype(int)
        y_test_cls  = y_test.astype(int)

        mae  = mean_absolute_error(y_test, y_pred_cont)
        rmse = mean_squared_error(y_test, y_pred_cont) ** 0.5
        print(f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

        # Issue 4: per-class report
        all_labels = [0, 1, 2, 3, 4]
        cr  = classification_report(y_test_cls, y_pred_cls,
                                    labels=all_labels, zero_division=0)
        cr_dict = classification_report(y_test_cls, y_pred_cls,
                                        labels=all_labels, zero_division=0, output_dict=True)
        cm  = confusion_matrix(y_test_cls, y_pred_cls, labels=all_labels)

        print(f"\n  Per-class report (rounded predictions, 0–4 stress score):")
        print(cr)
        print(f"  Confusion matrix (rows=actual, cols=predicted, classes 0-4):")
        print(cm)

        # Highlight underrepresented classes
        print("\n  Class distribution in test set:")
        for cls in all_labels:
            n = int((y_test_cls == cls).sum())
            pct = n / len(y_test_cls) * 100
            flag = " <- RARE" if pct < 1.0 else ""
            print(f"    Class {cls}: {n:,} ({pct:.2f}%){flag}")

        results[COMPOSITE_LABEL] = {
            "mae":                     round(mae, 4),
            "rmse":                    round(rmse, 4),
            "classification_report":   cr_dict,
            "confusion_matrix_5class": cm.tolist(),
            "n_test":                  int(len(y_test)),
            "split_method":            split_method,
        }

    # Save
    eval_path = model_dir / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone! Evaluation report saved -> {eval_path}")

    # Summary table
    print(f"\n=== EVALUATION SUMMARY  [split={split_method}, cutoff={train_cutoff}] ===")
    print(f"  {'Model':<25} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'Thr':>6}")
    print(f"  {'-'*65}")
    for lbl, m in results.items():
        if "f1_macro" in m:
            print(f"  {lbl:<25} {m['f1_macro']:>7.4f} {m['auc_roc']:>7.4f} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['decision_threshold']:>6.2f}")
        else:
            print(f"  {lbl:<25} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")


if __name__ == "__main__":
    main()

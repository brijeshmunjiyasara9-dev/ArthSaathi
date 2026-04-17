"""
03_train_models.py  (v3)
------------------------
Step 3: Train 5 stress prediction models.

Fixes applied:
  v2: Per-model leakage exclusions, SMOTE, custom health_stress threshold
  v3: [Issue 1] Time-based train/test split (month_slot < 2023-01-01 = train)
      [Issue 2] Verified NaN safety — SimpleImputer is in preprocessor pipeline
      [Issue 3] Health stress guard documented
      [Issue 5] food_to_expense_ratio in debt model is NOT leakage (documented)

Usage:
    python scripts/03_train_models.py --root D:/Project
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

# Issue 1: temporal cutoff — train on data BEFORE this date, test on AFTER
# Gives ~9 years train (2014-02 -> 2022-12), ~3 years test (2023-01 -> 2025-12)
TEMPORAL_CUTOFF = pd.Timestamp("2023-01-01")

NUMERIC_FEATURES = [
    "emi_to_income_ratio", "food_to_expense_ratio", "health_to_expense_ratio",
    "education_to_expense_ratio", "discretionary_to_expense_ratio",
    "savings_proxy", "income_diversity_score", "age_years_hh_mean",
    "education_rank_hh_max",
    "exp_all_emis", "exp_food_adjusted", "exp_health", "exp_education",
    "exp_recreation", "exp_vacation", "exp_restaurants_adjusted",
    "income_all_members_from_wages", "income_household_from_rent",
    "income_household_from_self_production", "income_household_from_private_transfers",
    "income_household_from_business_profit",
    "is_healthy_hh_min", "is_hospitalised_hh_any", "is_on_regular_medication_hh_any",
    "has_bank_account_hh_any", "has_health_insurance_hh_any", "has_life_insurance_hh_any",
    "has_provident_fund_account_hh_any", "has_credit_card_hh_any",
    "has_demat_account_hh_any", "has_mobile_phone_hh_any",
]

CATEGORICAL_FEATURES = [
    "state", "region_type", "age_group", "occupation_group", "education_group",
    "gender_group", "household_size_group", "gender_hh_mode", "occupation_type_hh_mode",
]

STRESS_LABELS   = ["financial_stress", "food_stress", "debt_stress", "health_stress"]
COMPOSITE_LABEL = "composite_stress_score"

# Issue 2 — per-model leakage exclusions:
#   savings_proxy      IS the financial_stress signal (income - expenditure)
#   food_to_expense_ratio IS the food_stress label definition (>0.5)
#   Note (Issue 5): food_to_expense_ratio IS used in debt_stress — this is NOT
#   leakage; it legitimately predicts debt risk via spending patterns.
EXCLUDED_FEATURES_PER_MODEL = {
    "financial_stress": ["savings_proxy"],
    "food_stress":      ["food_to_expense_ratio"],
    "debt_stress":      [],
    "health_stress":    [],
}

HEALTH_STRESS_THRESHOLD = 0.2   # Issue 1: stored in metadata
SMOTE_RATIO_THRESHOLD   = 50    # apply SMOTE when neg/pos > this
MAX_SAMPLE              = 1_000_000
RECENCY_CUTOFF          = pd.Timestamp("2022-01-01")  # v5: 2x weight for recent data
RECENCY_WEIGHT          = 2.0


# ─── Preprocessor ─────────────────────────────────────────────────────────────
# Issue 2: SimpleImputer is FIRST in both pipelines — handles NaN from chatbot

def build_preprocessor(num_feats, cat_feats) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),   # NaN-safe
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),  # NaN-safe
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ], remainder="drop")


# ─── SMOTE ────────────────────────────────────────────────────────────────────

def apply_smote(X_t, y, strategy=0.1):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=5)
        X_r, y_r = sm.fit_resample(X_t, y)
        print(f"    SMOTE: {len(y):,} -> {len(y_r):,} rows  "
              f"(pos: {(y==1).sum():,} -> {(y_r==1).sum():,})")
        return X_r, y_r
    except ImportError:
        print("    [WARN] imbalanced-learn not installed — SMOTE skipped. "
              "Run: pip install imbalanced-learn")
        return X_t, y


# ─── Time-based split (Issue 1) ───────────────────────────────────────────────

def temporal_split(sub: pd.DataFrame, feature_cols: list, label: str):
    """Split on month_slot instead of random. Falls back to random if needed."""
    if "month_slot" in sub.columns:
        ms = pd.to_datetime(sub["month_slot"], errors="coerce")
        train_mask = ms <  TEMPORAL_CUTOFF
        test_mask  = ms >= TEMPORAL_CUTOFF

        n_train = train_mask.sum()
        n_test  = test_mask.sum()
        print(f"    Temporal split: train={n_train:,} (before {TEMPORAL_CUTOFF.date()})  "
              f"test={n_test:,} (from {TEMPORAL_CUTOFF.date()})")

        if n_train < 500 or n_test < 50:
            print(f"    [WARN] Temporal split too small — falling back to random 80/20")
            return train_test_split(
                sub[feature_cols], sub[label].astype(int),
                test_size=0.2, random_state=42, stratify=sub[label].astype(int)
            )

        X_train = sub.loc[train_mask, feature_cols]
        y_train = sub.loc[train_mask, label].astype(int)
        X_test  = sub.loc[test_mask,  feature_cols]
        y_test  = sub.loc[test_mask,  label].astype(int)
        return X_train, X_test, y_train, y_test
    else:
        print("    [WARN] month_slot not in data — using random 80/20 split")
        return train_test_split(
            sub[feature_cols], sub[label].astype(int),
            test_size=0.2, random_state=42, stratify=sub[label].astype(int)
        )


# ─── Binary model training ────────────────────────────────────────────────────

def train_binary(X_t, y, label, sample_weight=None):
    n_neg  = int((y == 0).sum())
    n_pos  = int((y == 1).sum())
    scale  = round(n_neg / max(n_pos, 1), 2)
    print(f"    neg={n_neg:,}  pos={n_pos:,}  scale_pos_weight={scale:.1f}")

    X_fit, y_fit = X_t, y
    w_fit = sample_weight
    if scale > SMOTE_RATIO_THRESHOLD:
        print(f"    Ratio {scale:.0f} > {SMOTE_RATIO_THRESHOLD} -> applying SMOTE")
        X_fit, y_fit = apply_smote(X_t, y, strategy=0.1)
        w_fit = None   # SMOTE resamples rows; original weights no longer align

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_fit, y_fit, sample_weight=w_fit)
    if w_fit is not None:
        print(f"    Recency weights applied: mean={np.mean(w_fit):.2f}")
    return model, scale


# ─── Composite regressor ──────────────────────────────────────────────────────

def train_composite(X_t, y, sample_weight=None):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="rmse", random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_t, y.astype(float), sample_weight=sample_weight)
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/Project")
    parser.add_argument("--data",
        default="D:/Project/processed2/household_joined/household_stress_dataset.parquet")
    args = parser.parse_args()

    root      = Path(args.root)
    data_path = Path(args.data)
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n=== Loading dataset: {data_path} ===")
    import pyarrow.parquet as pq
    pf      = pq.ParquetFile(data_path)
    n_rows  = pf.metadata.num_rows
    n_rg    = pf.num_row_groups
    print(f"  Total rows: {n_rows:,}  |  Row groups: {n_rg}")
    print(f"  Temporal cutoff: train < {TEMPORAL_CUTOFF.date()}  "
          f"| test >= {TEMPORAL_CUTOFF.date()}")

    spg = max(1, int(MAX_SAMPLE / max(n_rg, 1)))
    dfs = []
    for i in range(n_rg):
        chunk = pf.read_row_group(i).to_pandas()
        if len(chunk) > spg:
            chunk = chunk.sample(n=spg, random_state=42)
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    if len(df) > MAX_SAMPLE:
        df = df.sample(MAX_SAMPLE, random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows for training")

    # Temporal distribution report
    if "month_slot" in df.columns:
        ms = pd.to_datetime(df["month_slot"], errors="coerce")
        n_train_rows = (ms < TEMPORAL_CUTOFF).sum()
        n_test_rows  = (ms >= TEMPORAL_CUTOFF).sum()
        print(f"  Train rows (pre-cutoff) : {n_train_rows:,}")
        print(f"  Test rows  (post-cutoff): {n_test_rows:,}")

    # ── v5: Engineer new features BEFORE computing num_feats ─────────────────
    print("\n  [v5] Engineering new features...")
    EPS = 1e-9

    inc_col  = next((c for c in ["income_all_members_from_wages", "total_income_hh"]
                     if c in df.columns), None)
    size_col = next((c for c in ["hh_size", "household_size", "num_members_hh"]
                     if c in df.columns), None)

    # 1. dependents_ratio
    if inc_col and size_col:
        df["dependents_ratio"] = (df[size_col] / (df[inc_col] / 10000 + EPS)).clip(upper=200)
        print(f"    dependents_ratio: {df['dependents_ratio'].notna().sum():,} non-null")
    else:
        df["dependents_ratio"] = np.nan
        print("    dependents_ratio: all NaN (no size column in parquet)")

    # 2. savings_buffer_months
    if "savings_proxy" in df.columns and "total_expenditure_adjusted" in df.columns:
        df["savings_buffer_months"] = df["savings_proxy"] / (df["total_expenditure_adjusted"] + EPS)
    elif "savings_proxy" in df.columns and inc_col:
        df["savings_buffer_months"] = df["savings_proxy"] / (df["savings_proxy"].abs() + df[inc_col] + EPS)
    else:
        df["savings_buffer_months"] = np.nan
    print(f"    savings_buffer_months: {df['savings_buffer_months'].notna().sum():,} non-null")

    # 3. no_insurance_vulnerability
    if "has_health_insurance_hh_any" in df.columns and "health_to_expense_ratio" in df.columns:
        df["no_insurance_vulnerability"] = (
            (1 - df["has_health_insurance_hh_any"].fillna(0)) *
            df["health_to_expense_ratio"].fillna(0)
        )
    else:
        df["no_insurance_vulnerability"] = np.nan
    print(f"    no_insurance_vulnerability: {df['no_insurance_vulnerability'].notna().sum():,} non-null")

    # 4. rural_low_income_flag
    if "region_type" in df.columns and inc_col:
        is_rural  = df["region_type"].astype(str).str.upper().isin(["RURAL", "VILLAGE"])
        df["rural_low_income_flag"] = (is_rural & (df[inc_col].fillna(0) < 25000)).astype(int)
    else:
        df["rural_low_income_flag"] = np.nan
    print(f"    rural_low_income_flag: positives={int((df['rural_low_income_flag']==1).sum()):,}")

    # 5. emi_health_interaction
    if "emi_to_income_ratio" in df.columns and "is_hospitalised_hh_any" in df.columns:
        is_sick = (
            df["is_hospitalised_hh_any"].fillna(0) +
            df.get("is_on_regular_medication_hh_any", pd.Series(0, index=df.index)).fillna(0)
        ).clip(upper=1)
        df["emi_health_interaction"] = df["emi_to_income_ratio"].fillna(0) * is_sick
    else:
        df["emi_health_interaction"] = np.nan
    print(f"    emi_health_interaction: {df['emi_health_interaction'].notna().sum():,} non-null")

    # ── Recency weights (2022+ rows get 2x weight) ────────────────────────────
    if "month_slot" in df.columns:
        ms_rec = pd.to_datetime(df["month_slot"], errors="coerce")
        df["__recency_weight__"] = np.where(ms_rec >= RECENCY_CUTOFF, RECENCY_WEIGHT, 1.0)
        print(f"  [v5] Recency weighting: "
              f"{(ms_rec >= RECENCY_CUTOFF).sum():,} rows @ {RECENCY_WEIGHT}x "
              f"({(ms_rec >= RECENCY_CUTOFF).mean()*100:.1f}% of data)")
    else:
        df["__recency_weight__"] = 1.0

    # ── Feature list (now includes v5 engineered columns) ─────────────────────
    NUMERIC_FEATURES.extend([
        "dependents_ratio", "savings_buffer_months", "no_insurance_vulnerability", 
        "rural_low_income_flag", "emi_health_interaction"
    ])
    num_feats = [f for f in NUMERIC_FEATURES    if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    feature_order = num_feats + cat_feats
    print(f"  Feature order: {len(num_feats)} numeric + {len(cat_feats)} categorical = {len(feature_order)} total")

    label_encoders_meta = {
        col: sorted([str(v) for v in df[col].dropna().unique()])
        for col in cat_feats
    }


    # ── Train binary models ────────────────────────────────────────────────────
    results     = {}
    excluded_actual = {}

    from sklearn.metrics import (
        f1_score, roc_auc_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )

    for label in STRESS_LABELS:
        if label not in df.columns:
            print(f"\n  SKIP {label} — not in dataset"); continue

        print(f"\n{'='*60}\n  Training: {label}\n{'='*60}")

        excl = [f for f in EXCLUDED_FEATURES_PER_MODEL.get(label, []) if f in feature_order]
        model_feats = [f for f in feature_order if f not in excl]
        model_num   = [f for f in num_feats if f not in excl]
        model_cat   = list(cat_feats)

        if excl:
            print(f"  [Leakage fix] Excluding: {excl}")
        excluded_actual[label] = excl

        # Include month_slot for splitting (not as a feature)
        keep_cols = model_feats + [label] + (
            ["month_slot"] if "month_slot" in df.columns else []
        )
        sub = df[keep_cols].dropna(subset=[label])
        print(f"  Samples: {len(sub):,}")

        if label == "health_stress":
            n_pos = int(sub[label].astype(int).sum())
            print(f"  [Label check] positives: {n_pos:,} ({n_pos/len(sub)*100:.2f}%)")

        # Issue 1: TIME-BASED SPLIT
        X_train, X_test, y_train, y_test = temporal_split(sub, model_feats, label)
        print(f"    Train: {len(y_train):,}  |  Test: {len(y_test):,}")

        prep = build_preprocessor(model_num, model_cat)
        X_train_t = prep.fit_transform(X_train)
        X_test_t  = prep.transform(X_test)

        # Extract recency weights for train rows
        w_train = None
        if "__recency_weight__" in sub.columns:
            w_train_raw = sub.loc[y_train.index, "__recency_weight__"].values \
                          if hasattr(y_train, 'index') else None
            if w_train_raw is not None:
                w_train = prep.transform(
                    X_train  # dummy — we just need the index alignment
                )  # actually just use the raw weight array aligned with X_train
                w_train = sub.loc[X_train.index, "__recency_weight__"].values

        model, spw = train_binary(X_train_t, y_train, label, sample_weight=w_train)

        y_prob = model.predict_proba(X_test_t)[:, 1]
        threshold = HEALTH_STRESS_THRESHOLD if label == "health_stress" else 0.5
        y_pred    = (y_prob >= threshold).astype(int)

        if label == "health_stress":
            print(f"  [Threshold] {threshold} (custom for health_stress)")

        f1   = f1_score(y_test, y_pred, average="macro")
        auc  = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        cr   = classification_report(y_test, y_pred, output_dict=True)

        print(f"  F1(macro)={f1:.4f}  AUC={auc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}")

        results[label] = {
            "f1_macro": round(f1,4), "auc_roc": round(auc,4),
            "precision": round(prec,4), "recall": round(rec,4),
            "decision_threshold": threshold,
            "scale_pos_weight_used": spw,
            "excluded_features": excl,
            "confusion_matrix": cm,
            "classification_report": cr,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "split_method": "temporal",
            "train_cutoff": str(TEMPORAL_CUTOFF.date()),
        }

        joblib.dump(model, model_dir / f"{label}_model.pkl")
        joblib.dump(prep,  model_dir / f"{label}_preprocessor.pkl")
        print(f"  Saved model + preprocessor")

    # ── Composite model ────────────────────────────────────────────────────────
    if COMPOSITE_LABEL in df.columns:
        print(f"\n{'='*60}\n  Training: composite_stress_score\n{'='*60}")

        keep = feature_order + [COMPOSITE_LABEL] + (
            ["month_slot"] if "month_slot" in df.columns else []
        )
        sub = df[keep].dropna(subset=[COMPOSITE_LABEL])
        y   = sub[COMPOSITE_LABEL].astype(float)
        X   = sub[feature_order]

        # Temporal split for composite
        if "month_slot" in sub.columns:
            ms2 = pd.to_datetime(sub["month_slot"], errors="coerce")
            tr_m = ms2 < TEMPORAL_CUTOFF
            te_m = ms2 >= TEMPORAL_CUTOFF
            if tr_m.sum() > 500 and te_m.sum() > 50:
                X_train, X_test = X[tr_m], X[te_m]
                y_train, y_test = y[tr_m], y[te_m]
                print(f"  Temporal split: train={len(y_train):,}  test={len(y_test):,}")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

        prep_comp = build_preprocessor(num_feats, cat_feats)
        X_train_t = prep_comp.fit_transform(X_train)
        X_test_t  = prep_comp.transform(X_test)

        # Recency weights for composite
        w_comp = None
        if "__recency_weight__" in sub.columns and tr_m.sum() > 500:
            try:
                w_comp = sub.loc[X_train.index, "__recency_weight__"].values
            except Exception:
                w_comp = None

        comp_model = train_composite(X_train_t, y_train, sample_weight=w_comp)
        y_pred_cont = comp_model.predict(X_test_t)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        # Issue 4: round for per-class evaluation
        y_pred_cls = np.clip(np.round(y_pred_cont), 0, 4).astype(int)
        y_test_cls = y_test.astype(int)

        mae  = mean_absolute_error(y_test, y_pred_cont)
        rmse = mean_squared_error(y_test, y_pred_cont) ** 0.5
        cr   = classification_report(y_test_cls, y_pred_cls, output_dict=True,
                                     labels=[0,1,2,3,4], zero_division=0)
        cm   = confusion_matrix(y_test_cls, y_pred_cls, labels=[0,1,2,3,4]).tolist()

        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}")
        print(classification_report(y_test_cls, y_pred_cls,
                                    labels=[0,1,2,3,4], zero_division=0))

        results[COMPOSITE_LABEL] = {
            "mae": round(mae,4), "rmse": round(rmse,4),
            "classification_report": cr,
            "confusion_matrix_5class": cm,
            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        }

        joblib.dump(comp_model, model_dir / "composite_stress_model.pkl")

        # Shared preprocessor (full feature set) for inference fallback
        prep_comp.fit(df[feature_order])
        joblib.dump(prep_comp, model_dir / "preprocessor.pkl")
        print(f"  Saved composite model + shared preprocessor")

    # ── Metadata ───────────────────────────────────────────────────────────────
    metadata = {
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "feature_order": feature_order,
        "label_encoders": label_encoders_meta,
        "model_version": "3.0",
        "training_date": datetime.now().isoformat(),
        "stress_labels": STRESS_LABELS,
        "composite_label": COMPOSITE_LABEL,
        "health_stress_threshold": HEALTH_STRESS_THRESHOLD,
        "excluded_features_per_model": excluded_actual,
        "split_method": "temporal",
        "train_cutoff": str(TEMPORAL_CUTOFF.date()),
        "nan_handling": "SimpleImputer(median/most_frequent) in preprocessor pipeline",
        "feature_notes": {
            "food_to_expense_ratio_in_debt_model": (
                "NOT leakage — food spending patterns legitimately predict debt risk "
                "via disposable income constraints. Only excluded from food_stress model."
            )
        }
    }
    with open(model_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(model_dir / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n=== TRAINING SUMMARY ===")
    print(f"  Split: TEMPORAL  (train < {TEMPORAL_CUTOFF.date()}, "
          f"test >= {TEMPORAL_CUTOFF.date()})")
    print(f"  {'Model':<25} {'F1':>7} {'AUC':>7} {'Recall':>7} {'SPW':>7} {'Threshold':>10}")
    print(f"  {'-'*75}")
    for lbl, m in results.items():
        if "f1_macro" in m:
            print(f"  {lbl:<25} {m['f1_macro']:>7.4f} {m['auc_roc']:>7.4f} "
                  f"{m['recall']:>7.4f} {m['scale_pos_weight_used']:>7.1f} "
                  f"{m['decision_threshold']:>10.2f}")
        else:
            print(f"  {lbl:<25} MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    print("\nDone! Models saved to models/")


if __name__ == "__main__":
    main()

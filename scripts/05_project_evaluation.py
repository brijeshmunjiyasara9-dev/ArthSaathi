# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
ArthSaathi -- Comprehensive Project Evaluation Script
======================================================
Evaluates the entire pipeline:
  1.  Raw data inventory & stats
  2.  Processed data (Step 1) stats & column validation
  3.  Raw vs Processed diff analysis
  4.  Household joined dataset (schema + sample)
  5.  POI aggregated cache
  6.  Model artefact inventory & sizes
  7.  Model performance deep-dive
  8.  Feature importance (top-10 per model)
  9.  System health checklist
  10. Pipeline status summary
  --> Saves full report to models/project_evaluation_report.json

Usage:
    python scripts/05_project_evaluation.py --root D:/Project
"""

import argparse
import json
import os
import pathlib
import datetime
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─────────────────────────── helpers ────────────────────────────

DIV  = "=" * 70
DIV2 = "-" * 70

def section(title):
    print(f"\n{DIV}\n  {title}\n{DIV}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def err(msg):  print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")

def file_mb(p):  return round(os.path.getsize(p) / 1_048_576, 2)
def dir_mb(p):
    return round(sum(f.stat().st_size for f in pathlib.Path(p).rglob("*") if f.is_file()) / 1_048_576, 2)

def count_ext(path, ext):
    return len(list(pathlib.Path(path).rglob(f"*.{ext}")))

def count_raw(path):
    z = count_ext(path, "zip")
    return z if z else count_ext(path, "parquet")

def raw_fmt(path):
    return "zip" if count_ext(path, "zip") else "parquet"

def has_raw(path):
    return pathlib.Path(path).exists() and count_raw(path) > 0

# ─────────────────────────── main ───────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/Project")
    args = parser.parse_args()

    ROOT     = pathlib.Path(args.root)
    RAW_INC  = ROOT / "Dataset" / "Income_Pyramid"
    RAW_POI  = ROOT / "Dataset" / "People_of_India"
    RAW_CON  = ROOT / "Dataset" / "Consumption_Pyramid"
    PROC_INC = ROOT / "processed" / "income_pyramid"
    PROC_POI = ROOT / "processed" / "people_of_india"
    PROC_CON = ROOT / "processed" / "consumption_pyramid"
    HH_JOIN  = ROOT / "processed2" / "household_joined" / "household_stress_dataset.parquet"
    POI_AGG  = ROOT / "processed2" / "household_joined" / "poi_household_agg.parquet"
    MODELS   = ROOT / "models"
    EVAL_RPT = MODELS / "evaluation_report.json"
    FEAT_META= MODELS / "feature_metadata.json"
    BACKEND  = ROOT / "web" / "backend" / "main.py"
    BACKEND_ENV = ROOT / "web" / "backend" / ".env"
    FRONTEND = ROOT / "web" / "frontend" / "src" / "App.jsx"

    report = {"generated_at": datetime.datetime.now().isoformat(),
               "root": str(ROOT), "sections": {}}

    # ══════════════════════════════════════════════════════
    # 1. RAW DATA INVENTORY
    # ══════════════════════════════════════════════════════
    section("1. RAW DATA INVENTORY")
    raw_info = {}
    for label, path in [("INC (Income Pyramid)", RAW_INC),
                         ("POI (People of India)", RAW_POI),
                         ("CON (Consumption Pyramid)", RAW_CON)]:
        if not path.exists():
            err(f"{label}: NOT FOUND at {path}")
            raw_info[label] = {"exists": False}
            continue
        n = count_raw(path)
        sz = dir_mb(path)
        fmt = raw_fmt(path)
        ok(f"{label}: {n} {fmt} files | {sz:.1f} MB")
        raw_info[label] = {"exists": True, "files": n, "format": fmt, "size_mb": sz}
    report["sections"]["1_raw_data"] = raw_info

    # ══════════════════════════════════════════════════════
    # 2. PROCESSED DATA (Step 1) INVENTORY
    # ══════════════════════════════════════════════════════
    section("2. PROCESSED DATA -- Step 1 (cleaned parquets)")
    proc_info = {}
    for label, path in [("INC cleaned", PROC_INC),
                         ("POI cleaned", PROC_POI),
                         ("CON cleaned", PROC_CON)]:
        if not path.exists():
            err(f"{label}: NOT FOUND at {path}")
            proc_info[label] = {"exists": False}
            continue
        n = count_ext(path, "parquet")
        sz = dir_mb(path)
        ok(f"{label}: {n} parquet files | {sz:.1f} MB")
        proc_info[label] = {"exists": True, "parquet_files": n, "size_mb": sz}
    report["sections"]["2_processed_step1"] = proc_info

    # ══════════════════════════════════════════════════════
    # 3. RAW vs PROCESSED DIFF
    # ══════════════════════════════════════════════════════
    section("3. RAW vs PROCESSED DIFF")
    diff_info = {}
    try:
        proc_files = sorted(PROC_INC.glob("*.parquet")) if PROC_INC.exists() else []
        if proc_files:
            proc_f = proc_files[0]
            info(f"Analysing processed INC sample: {proc_f.name}")
            df_p = pd.read_parquet(proc_f)
            pr, pc = df_p.shape
            miss_pct = df_p.isna().sum().sum() / df_p.size * 100
            num_cols = df_p.select_dtypes(include="number").columns
            sentinel = int((df_p[num_cols] == -99).sum().sum())

            print(f"\n  Processed INC file (one reference month):")
            print(f"  {DIV2}")
            print(f"  {'Rows (accepted responses)':<45} {pr:>10,}")
            print(f"  {'Columns (after cleaning + selection)':<45} {pc:>10,}")
            print(f"  {'Overall NaN %':<45} {miss_pct:>9.2f}%")
            if sentinel == 0:
                ok("No sentinel -99 values remain in processed INC data")
            else:
                warn(f"{sentinel} sentinel -99 values still present!")

            # Column detail
            print(f"\n  {'Column':<45} {'dtype':<14}  {'Missing%'}")
            print(f"  {DIV2}")
            for c in df_p.columns:
                mp = df_p[c].isna().sum() / pr * 100
                flag = "[WARN]" if mp > 30 else "      "
                print(f"  {flag}  {c:<43} {str(df_p[c].dtype):<14}  {mp:5.1f}%")

            # Overall size comparison
            raw_sz  = dir_mb(RAW_INC) + dir_mb(RAW_POI) + dir_mb(RAW_CON)
            proc_sz = dir_mb(PROC_INC) + dir_mb(PROC_POI) + dir_mb(PROC_CON)
            print(f"\n  {'Metric':<45} {'Value':>15}")
            print(f"  {DIV2}")
            print(f"  {'Total raw size (all 3 datasets)':<45} {raw_sz:>13.1f} MB")
            print(f"  {'Total processed size (cleaned parquets)':<45} {proc_sz:>13.1f} MB")
            pct_smaller = (1 - proc_sz / raw_sz) * 100 if raw_sz > 0 else 0
            print(f"  {'Compression / reduction':<45} {pct_smaller:>13.1f}% smaller")
            print(f"  {'INC: raw -> processed':<45}  {count_raw(RAW_INC):>5} {raw_fmt(RAW_INC)} -> {count_ext(PROC_INC,'parquet'):>5} parquet")
            print(f"  {'POI: raw -> processed':<45}  {count_raw(RAW_POI):>5} {raw_fmt(RAW_POI)} -> {count_ext(PROC_POI,'parquet'):>5} parquet")
            print(f"  {'CON: raw -> processed':<45}  {count_raw(RAW_CON):>5} {raw_fmt(RAW_CON)} -> {count_ext(PROC_CON,'parquet'):>5} parquet")

            diff_info = {
                "note": "Raw files are ZIP; cleaned output is parquet",
                "sample_rows": pr, "sample_cols": pc,
                "sample_missing_pct": round(miss_pct, 2),
                "sentinel_remaining": sentinel,
                "raw_total_mb": raw_sz, "proc_total_mb": proc_sz,
                "reduction_pct": round(pct_smaller, 1),
                "columns_kept": list(df_p.columns)
            }
            del df_p
        else:
            warn("No processed INC files found -- run 01_clean_data.py first")
    except Exception as e:
        err(f"Diff analysis error: {e}")
        traceback.print_exc()
    report["sections"]["3_raw_vs_processed_diff"] = diff_info

    # ══════════════════════════════════════════════════════
    # 4. HOUSEHOLD JOINED DATASET
    # ══════════════════════════════════════════════════════
    section("4. HOUSEHOLD JOINED DATASET -- Final ML Dataset")
    hh_info = {}
    if not HH_JOIN.exists():
        err(f"household_stress_dataset.parquet NOT FOUND")
        err("Run: python scripts/02_build_household_dataset.py --root D:/Project")
    else:
        try:
            import pyarrow.parquet as pq
            sz = file_mb(HH_JOIN)
            ok(f"File found: {sz:.1f} MB")

            info("Reading schema & row count from metadata (no full load)...")
            pf      = pq.ParquetFile(HH_JOIN)
            meta    = pf.metadata
            n_rows  = meta.num_rows
            n_cols  = meta.num_columns
            n_rg    = meta.num_row_groups
            cols    = pf.schema_arrow.names

            ok(f"Total rows  : {n_rows:,}")
            ok(f"Total cols  : {n_cols}")
            ok(f"Row groups  : {n_rg}")

            info(f"Reading first row-group as sample...")
            df = pf.read_row_group(0).to_pandas()
            sample_n = len(df)
            info(f"Sample rows : {sample_n:,}")

            # Stress label distribution
            stress_labels = ["financial_stress", "food_stress", "debt_stress",
                             "health_stress", "composite_stress_score"]
            label_info = {}
            print(f"\n  {'Label':<35} {'Positive in sample':>20} {'% (sample)':>12}")
            print(f"  {DIV2}")
            for col in stress_labels:
                if col not in df.columns:
                    warn(f"Label '{col}' not found!"); continue
                if col == "composite_stress_score":
                    dist = df[col].value_counts().sort_index().to_dict()
                    print(f"  {col:<35} distribution={dist}")
                    label_info[col] = {str(k): int(v) for k, v in dist.items()}
                else:
                    n1 = int(df[col].sum())
                    pct = n1 / sample_n * 100
                    print(f"  {col:<35} {n1:>20,} {pct:>11.1f}%")
                    label_info[col] = {"positive_sample": n1, "pct_sample": round(pct, 2)}

            # Missing analysis on key columns
            key_cols = ["financial_stress","food_stress","debt_stress","health_stress",
                        "emi_to_income_ratio","food_to_expense_ratio","savings_proxy",
                        "is_healthy_hh_min","has_bank_account_hh_any","state"]
            print(f"\n  {'Column':<42} {'Missing':>10} {'Missing%':>10}  (sample={sample_n:,})")
            print(f"  {DIV2}")
            miss_info = {}
            for col in key_cols:
                if col in df.columns:
                    nm = int(df[col].isna().sum())
                    pm = nm / sample_n * 100
                    flag = "[WARN]" if pm > 20 else "      "
                    print(f"  {flag} {col:<40} {nm:>10,} {pm:>9.1f}%")
                    miss_info[col] = {"missing": nm, "missing_pct": round(pm, 2)}

            # Descriptive stats for ratio features
            ratio_cols = [c for c in df.columns if "ratio" in c or c == "savings_proxy"]
            if ratio_cols:
                print(f"\n  Ratio Feature Statistics (sample):")
                print(f"  {'Feature':<45} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
                print(f"  {DIV2}")
                for c in ratio_cols:
                    if c in df.columns:
                        s = df[c].dropna()
                        if len(s):
                            print(f"  {c:<45} {s.mean():>10.4f} {s.std():>10.4f} "
                                  f"{s.median():>10.4f} {s.min():>10.2f} {s.max():>10.2f}")

            # Date & geographic
            if "month_slot" in df.columns:
                dates = pd.to_datetime(df["month_slot"], errors="coerce")
                ok(f"\n  Date range (first row-group): {dates.min().strftime('%b %Y')} -> {dates.max().strftime('%b %Y')}")
            if "state" in df.columns:
                ok(f"  States in sample: {df['state'].nunique()}")
                state_top = df["state"].value_counts().head(5).to_dict()
                info(f"  Top 5 states (sample): {state_top}")
            if "region_type" in df.columns:
                rt = df["region_type"].value_counts().to_dict()
                info(f"  Region type: {rt}")

            # All column names
            print(f"\n  All {n_cols} columns:")
            for c in cols:
                dtype = str(df[c].dtype) if c in df.columns else "n/a"
                print(f"    {c:<52} {dtype}")

            hh_info = {"file_size_mb": sz, "total_rows": n_rows, "total_cols": n_cols,
                       "num_row_groups": n_rg, "schema_columns": cols,
                       "stress_labels": label_info, "missing_analysis": miss_info}
            del df

        except Exception as e:
            err(f"Household dataset analysis failed: {e}")
            traceback.print_exc()
    report["sections"]["4_household_dataset"] = hh_info

    # ══════════════════════════════════════════════════════
    # 5. POI AGGREGATED CACHE
    # ══════════════════════════════════════════════════════
    section("5. POI HOUSEHOLD AGGREGATED CACHE")
    poi_info = {}
    if POI_AGG.exists():
        try:
            sz = file_mb(POI_AGG)
            df_poi = pd.read_parquet(POI_AGG)
            r, c = df_poi.shape
            ok(f"poi_household_agg.parquet: {r:,} rows x {c} cols | {sz:.1f} MB")
            poi_info = {"exists": True, "rows": r, "cols": c, "size_mb": sz,
                        "columns": list(df_poi.columns)}
            del df_poi
        except Exception as e:
            err(f"POI agg load error: {e}")
    else:
        err("poi_household_agg.parquet NOT FOUND")
        poi_info = {"exists": False}
    report["sections"]["5_poi_agg"] = poi_info

    # ══════════════════════════════════════════════════════
    # 6. MODEL ARTEFACTS
    # ══════════════════════════════════════════════════════
    section("6. MODEL ARTEFACTS INVENTORY")
    model_files = {
        "financial_stress_model.pkl":  "Financial Stress Classifier (XGBoost)",
        "food_stress_model.pkl":       "Food Stress Classifier (XGBoost)",
        "debt_stress_model.pkl":       "Debt Stress Classifier (XGBoost)",
        "health_stress_model.pkl":     "Health Stress Classifier (XGBoost)",
        "composite_stress_model.pkl":  "Composite Stress Regressor (XGBoost)",
        "preprocessor.pkl":            "Sklearn Preprocessor Pipeline",
        "feature_metadata.json":       "Feature Metadata JSON",
        "evaluation_report.json":      "Evaluation Report JSON",
    }
    art_info = {}
    print(f"\n  {'File':<38} {'Size':>9}  Description")
    print(f"  {DIV2}")
    for fname, desc in model_files.items():
        fp = MODELS / fname
        if fp.exists():
            sz = file_mb(fp)
            print(f"  [OK]  {fname:<36} {sz:>7.2f} MB  {desc}")
            art_info[fname] = {"exists": True, "size_mb": sz}
        else:
            print(f"  [FAIL] {fname:<36} {'MISSING':>9}  {desc}")
            art_info[fname] = {"exists": False}
    report["sections"]["6_model_artefacts"] = art_info

    # ══════════════════════════════════════════════════════
    # 7. MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════
    section("7. MODEL PERFORMANCE DEEP-DIVE")
    perf_info = {}
    if EVAL_RPT.exists():
        with open(EVAL_RPT) as f:
            ev = json.load(f)

        binary = ["financial_stress","food_stress","debt_stress","health_stress"]
        print(f"\n  {'Model':<27} {'F1-Macro':>10} {'AUC-ROC':>10} {'Accuracy':>10} {'N Test':>12}")
        print(f"  {DIV2}")
        for m in binary:
            d   = ev.get(m, {})
            f1  = d.get("f1_macro", 0)
            auc = d.get("auc_roc", 0)
            acc = d.get("classification_report", {}).get("accuracy", 0)
            n   = d.get("n_test", 0)
            flag = "[OK]  " if f1 >= 0.8 else ("[WARN]" if f1 >= 0.5 else "[FAIL]")
            print(f"  {flag} {m:<25} {f1:>10.4f} {auc:>10.4f} {acc:>10.4f} {n:>12,}")
            perf_info[m] = {"f1_macro": f1, "auc_roc": auc,
                            "accuracy": round(acc,4), "n_test": n}

        comp = ev.get("composite_stress_score", {})
        print(f"\n  Composite Regressor -> MAE={comp.get('mae')}  RMSE={comp.get('rmse')}")
        perf_info["composite_stress_score"] = comp

        # Per-class precision/recall
        print(f"\n  Per-class detail:")
        print(f"  {'Model':<27} {'Cls':>4} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {DIV2}")
        for m in binary:
            cr = ev.get(m,{}).get("classification_report",{})
            for cls in ["0","1"]:
                if cls in cr:
                    r = cr[cls]
                    print(f"  {m:<27} {cls:>4} {r['precision']:>10.4f} "
                          f"{r['recall']:>10.4f} {r['f1-score']:>10.4f} "
                          f"{int(r['support']):>10,}")

        # Confusion matrices
        print(f"\n  Confusion Matrices:")
        for m in binary:
            cm = ev.get(m,{}).get("confusion_matrix",[])
            if cm and len(cm)==2:
                tn,fp = cm[0]; fn,tp = cm[1]
                print(f"\n  {m}:")
                print(f"    TN={tn:,}  FP={fp:,}")
                print(f"    FN={fn:,}  TP={tp:,}  "
                      f"Recall(1)={tp/(tp+fn+1e-9)*100:.1f}%  "
                      f"FPR={fp/(fp+tn+1e-9)*100:.2f}%")
    else:
        err("evaluation_report.json NOT FOUND -- run 04_evaluate_models.py")
    report["sections"]["7_model_performance"] = perf_info

    # ══════════════════════════════════════════════════════
    # 8. FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════
    section("8. FEATURE IMPORTANCE (Top-10 per model)")
    fi_info = {}
    try:
        import joblib
        prep_path = MODELS / "preprocessor.pkl"
        if prep_path.exists() and FEAT_META.exists():
            with open(FEAT_META) as f:
                meta = json.load(f)
            feat_order = meta.get("feature_order", [])
            prep = joblib.load(prep_path)
            try:
                feat_names = list(prep.get_feature_names_out())
            except Exception:
                feat_names = feat_order

            for fname, label in [
                ("financial_stress_model.pkl","financial_stress"),
                ("food_stress_model.pkl","food_stress"),
                ("debt_stress_model.pkl","debt_stress"),
                ("health_stress_model.pkl","health_stress"),
            ]:
                mp = MODELS / fname
                if not mp.exists(): continue
                model = joblib.load(mp)
                fi = None
                if hasattr(model,"feature_importances_"):
                    fi = model.feature_importances_
                elif hasattr(model,"best_estimator_"):
                    be = model.best_estimator_
                    if hasattr(be,"feature_importances_"):
                        fi = be.feature_importances_

                if fi is not None:
                    names = feat_names[:len(fi)] if len(feat_names) >= len(fi) \
                            else [f"f{i}" for i in range(len(fi))]
                    s = pd.Series(fi, index=names).sort_values(ascending=False)
                    top = s.head(10)
                    print(f"\n  {label} -- Top 10 Features:")
                    for fn_feat, imp in top.items():
                        bar = "#" * max(1, int(imp * 50))
                        print(f"    {fn_feat:<48} {imp:.4f}  {bar}")
                    fi_info[label] = top.to_dict()
        else:
            warn("preprocessor.pkl or feature_metadata.json missing")
    except Exception as e:
        warn(f"Feature importance skipped: {e}")
    report["sections"]["8_feature_importance"] = fi_info

    # ══════════════════════════════════════════════════════
    # 9. SYSTEM HEALTH CHECKLIST
    # ══════════════════════════════════════════════════════
    section("9. SYSTEM HEALTH CHECKLIST")
    checks = []
    def chk(label, cond, fix=None):
        if cond:
            ok(label); checks.append({"check":label,"passed":True})
        else:
            err(label)
            if fix: info(f"    Fix: {fix}")
            checks.append({"check":label,"passed":False,"fix":fix})

    chk("Raw INC data present",    has_raw(RAW_INC))
    chk("Raw POI data present",    has_raw(RAW_POI))
    chk("Raw CON data present",    has_raw(RAW_CON))
    chk("Processed INC (Step 1)",  count_ext(PROC_INC,"parquet") > 0 if PROC_INC.exists() else False,
        "python scripts/01_clean_data.py --root D:/Project")
    chk("Processed POI (Step 1)",  count_ext(PROC_POI,"parquet") > 0 if PROC_POI.exists() else False,
        "python scripts/01_clean_data.py --root D:/Project")
    chk("Processed CON (Step 1)",  count_ext(PROC_CON,"parquet") > 0 if PROC_CON.exists() else False,
        "python scripts/01_clean_data.py --root D:/Project")
    chk("Household joined dataset", HH_JOIN.exists(),
        "python scripts/02_build_household_dataset.py --root D:/Project")
    chk("POI aggregated cache",    POI_AGG.exists())
    for mf in ["financial_stress_model.pkl","food_stress_model.pkl",
               "debt_stress_model.pkl","health_stress_model.pkl",
               "composite_stress_model.pkl","preprocessor.pkl"]:
        chk(f"Model: {mf}", (MODELS/mf).exists(),
            "python scripts/03_train_models.py --root D:/Project")
    chk("evaluation_report.json",  EVAL_RPT.exists(),
        "python scripts/04_evaluate_models.py --root D:/Project")
    chk("feature_metadata.json",   FEAT_META.exists())
    chk("Backend main.py",         BACKEND.exists())
    chk("Backend .env",            BACKEND_ENV.exists())
    chk("Frontend App.jsx",        FRONTEND.exists())

    passed = sum(1 for c in checks if c["passed"])
    total  = len(checks)
    print(f"\n  RESULT: {passed}/{total} checks passed")
    if passed == total:
        ok("All systems GO! Pipeline is complete end-to-end.")
    else:
        warn(f"{total-passed} checks FAILED -- see fixes above.")
    report["sections"]["9_health_checklist"] = {
        "passed": passed, "total": total, "checks": checks}

    # ══════════════════════════════════════════════════════
    # 10. PIPELINE STATUS SUMMARY
    # ══════════════════════════════════════════════════════
    section("10. PIPELINE STATUS SUMMARY")
    steps = {
        "Step 1 -- Clean raw data":          count_ext(PROC_INC,"parquet") > 0 if PROC_INC.exists() else False,
        "Step 2 -- Build household dataset":  HH_JOIN.exists(),
        "Step 3 -- Train models":             (MODELS/"financial_stress_model.pkl").exists(),
        "Step 4 -- Evaluate models":          EVAL_RPT.exists(),
        "Step 5 -- FastAPI backend":          BACKEND.exists(),
        "Step 6 -- React frontend":           FRONTEND.exists(),
    }
    all_done = True
    print()
    for step, done in steps.items():
        if done: ok(step)
        else:
            err(f"{step}  <-- INCOMPLETE")
            all_done = False

    if all_done:
        print(f"\n  >>> FULL PIPELINE COMPLETE -- ArthSaathi is ready to deploy! <<<")
    else:
        print(f"\n  [!] Some pipeline steps need attention (see above).")
    report["sections"]["10_pipeline_status"] = steps

    # ══════════════════════════════════════════════════════
    # SAVE JSON REPORT
    # ══════════════════════════════════════════════════════
    out = MODELS / "project_evaluation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    section("DONE")
    ok(f"Full report saved -> {out}")
    print()


if __name__ == "__main__":
    main()

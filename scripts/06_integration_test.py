"""
06_integration_test.py
----------------------
Issue 6: Full end-to-end integration test of the chatbot -> backend pipeline.

Tests:
  A. Full input   — all 25+ fields provided
  B. Partial input — key financial fields only (chatbot progression)
  C. Missing health fields — health_stress guard must trigger
  D. Empty input  — graceful fallback
  E. POST /api/assess — live backend test (if server running)

Usage:
    python scripts/06_integration_test.py --root D:/Project [--url http://localhost:8000]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def run_test(name: str, fn):
    try:
        result = fn()
        print(f"  [PASS] {name}")
        return True, result
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        return False, None
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/Project")
    parser.add_argument("--url",  default=None,
                        help="Backend URL for live API test, e.g. http://localhost:8000")
    args = parser.parse_args()

    root      = Path(args.root)
    model_dir = root / "models"
    backend   = root / "web" / "backend"

    # Add backend to path for imports
    sys.path.insert(0, str(backend))

    print("\n" + "="*65)
    print("  ArthSaathi — Integration Test Suite")
    print("="*65)

    # ── Load models ────────────────────────────────────────────────────────────
    from models.predict import predict, map_chat_inputs_to_features, _load_models
    _load_models(str(model_dir))
    from models.predict import _metadata, _models, _preprocessors

    print(f"\n  Models loaded: {list(_models.keys())}")
    print(f"  Preprocessors: {list(_preprocessors.keys())}")
    print(f"  Feature order length: {len(_metadata.get('feature_order', []))}")

    results = {}
    passed  = 0
    total   = 0

    # ─────────────────────────────────────────────────────────────────────────
    # A. Full Input
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- A. Full Input (all fields provided) ---")

    FULL_INPUT = {
        "monthly_income":              45000,
        "monthly_total_expense":       38000,
        "monthly_food_expense":        12000,
        "monthly_emi":                 8000,
        "monthly_health_expense":      2000,
        "monthly_education_expense":   3000,
        "monthly_recreation_expense":  1500,
        "monthly_vacation_expense":    500,
        "monthly_restaurant_expense":  1000,
        "income_wages":                45000,
        "income_rent":                 0,
        "income_business":             0,
        "is_hospitalised":             False,
        "is_on_medication":            False,
        "has_bank_account":            True,
        "has_health_insurance":        True,
        "has_life_insurance":          False,
        "has_provident_fund":          True,
        "has_credit_card":             False,
        "has_demat_account":           False,
        "state":                       "Maharashtra",
        "region_type":                 "Urban",
        "age_head":                    38,
        "occupation":                  "Salaried",
        "education":                   "Graduate",
        "gender":                      "M",
    }

    total += 1
    def test_full_returns_dict():
        p = predict(FULL_INPUT, str(model_dir))
        assert isinstance(p, dict), f"Expected dict, got {type(p)}"
        return p
    ok, preds = run_test("Full input — returns dict", test_full_returns_dict)
    passed += ok

    total += 1
    def test_full_all_labels():
        p = predict(FULL_INPUT, str(model_dir))
        for k in ["financial_stress","food_stress","debt_stress","health_stress"]:
            assert k in p, f"Missing key {k}"
            assert p[k] is not None, f"Key {k} is None"
        return p
    ok, preds = run_test("Full input — all 4 stress labels present", test_full_all_labels)
    passed += ok

    # Simpler direct test
    total += 1
    def test_full_labels():
        p = predict(FULL_INPUT, str(model_dir))
        for k in ["financial_stress", "food_stress", "debt_stress", "health_stress"]:
            assert k in p, f"Missing key: {k}"
            assert p[k] is not None, f"{k} is None (health guard incorrectly triggered?)"
        assert "composite_stress_score" in p, "Missing composite_stress_score"
        return p
    ok, preds = run_test("Full input — all labels non-None", test_full_labels)
    passed += ok
    if preds:
        print(f"    -> Predictions: { {k: v for k, v in preds.items() if not k.endswith('message')} }")

    total += 1
    def test_score_range():
        p = predict(FULL_INPUT, str(model_dir))
        for k in ["financial_stress", "food_stress", "debt_stress", "health_stress"]:
            if p.get(k) is not None:
                assert 0.0 <= p[k] <= 1.0, f"{k}={p[k]} out of [0,1]"
        cs = p.get("composite_stress_score", 0)
        assert 0.0 <= cs <= 4.0, f"composite={cs} out of [0,4]"
        return p
    ok, _ = run_test("Full input — scores in valid range", test_score_range)
    passed += ok

    # ─────────────────────────────────────────────────────────────────────────
    # B. Partial Input (financial fields only — typical chatbot state)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- B. Partial Input (financial fields only) ---")

    PARTIAL_INPUT = {
        "monthly_income":         30000,
        "monthly_total_expense":  27000,
        "monthly_food_expense":   9000,
        "monthly_emi":            5000,
        "is_hospitalised":        False,
        "is_on_medication":       False,
        "state":                  "Uttar Pradesh",
        "region_type":            "Rural",
    }

    total += 1
    def test_partial():
        p = predict(PARTIAL_INPUT, str(model_dir))
        assert isinstance(p, dict), "Must return dict"
        # NaN fields → imputed by preprocessor, must not raise
        return p
    ok, preds = run_test("Partial input — no exception with missing fields", test_partial)
    passed += ok
    if preds:
        print(f"    -> {preds}")

    total += 1
    def test_partial_nan_safe():
        # Verify feature matrix contains NaN but preprocessor handles it
        df_feat, _ = map_chat_inputs_to_features(PARTIAL_INPUT)
        nan_cols = df_feat.columns[df_feat.isna().any()].tolist()
        assert len(nan_cols) > 0, "Expected some NaN (missing optional fields)"
        print(f"    NaN columns ({len(nan_cols)}): {nan_cols[:5]}...")
        # Should be transformable without error
        prep = _preprocessors.get("financial_stress")
        if prep:
            excl = _metadata.get("excluded_features_per_model", {}).get("financial_stress", [])
            feat_order = [f for f in _metadata["feature_order"] if f not in excl]
            X = df_feat[feat_order]
            X_t = prep.transform(X)
            assert not np.isnan(X_t).any(), "Preprocessor left NaN in output!"
        return True
    ok, _ = run_test("Partial input — SimpleImputer removes NaN", test_partial_nan_safe)
    passed += ok

    # ─────────────────────────────────────────────────────────────────────────
    # C. Missing health fields — guard must trigger (Issue 3)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- C. Health Stress Guard (both health fields absent) ---")

    NO_HEALTH_INPUT = {
        "monthly_income":         25000,
        "monthly_total_expense":  22000,
        "monthly_food_expense":   7000,
        "state":                  "Tamil Nadu",
        # is_hospitalised and is_on_medication deliberately NOT provided
    }

    total += 1
    def test_health_guard():
        p = predict(NO_HEALTH_INPUT, str(model_dir))
        assert p.get("health_stress") is None, \
            f"Expected health_stress=None, got {p.get('health_stress')}"
        assert "health_stress_message" in p, \
            "Expected health_stress_message in response"
        print(f"    -> Message: \"{p['health_stress_message']}\"")
        return p
    ok, _ = run_test("Health guard — returns None + message when fields absent", test_health_guard)
    passed += ok

    total += 1
    def test_health_guard_not_triggered():
        # When is_hospitalised=False is explicitly provided, guard should NOT trigger
        inp = {**NO_HEALTH_INPUT, "is_hospitalised": False}
        p = predict(inp, str(model_dir))
        # health_stress should be a number, not None
        assert p.get("health_stress") is not None, \
            "Guard incorrectly triggered when is_hospitalised was provided"
        return p
    ok, _ = run_test("Health guard — NOT triggered when is_hospitalised=False provided", test_health_guard_not_triggered)
    passed += ok

    # ─────────────────────────────────────────────────────────────────────────
    # D. Empty input — graceful fallback
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  --- D. Empty Input (fallback behaviour) ---")

    total += 1
    def test_empty():
        p = predict({}, str(model_dir))
        assert isinstance(p, dict), "Must return dict even for empty input"
        return p
    ok, preds = run_test("Empty input — returns dict without crash", test_empty)
    passed += ok
    if preds:
        print(f"    -> {preds}")

    # ─────────────────────────────────────────────────────────────────────────
    # E. Live API test (optional — only if --url provided)
    # ─────────────────────────────────────────────────────────────────────────
    if args.url:
        print(f"\n  --- E. Live API Test ({args.url}) ---")
        try:
            import requests

            total += 1
            def test_api_assess():
                r = requests.post(
                    f"{args.url}/api/assess",
                    json={"user_inputs": FULL_INPUT},
                    timeout=15
                )
                assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
                data = r.json()
                assert "predictions" in data or "stress_scores" in data or "results" in data, \
                    f"Unexpected response shape: {list(data.keys())}"
                return data
            ok, resp = run_test("POST /api/assess — returns 200 with predictions", test_api_assess)
            passed += ok
            if resp:
                print(f"    -> Response keys: {list(resp.keys())}")

            total += 1
            def test_api_partial():
                r = requests.post(
                    f"{args.url}/api/assess",
                    json={"user_inputs": PARTIAL_INPUT},
                    timeout=15
                )
                assert r.status_code == 200, f"HTTP {r.status_code}"
                return r.json()
            ok, _ = run_test("POST /api/assess — partial input returns 200", test_api_partial)
            passed += ok

        except ImportError:
            print("  [SKIP] requests not installed — run: pip install requests")
    else:
        print(f"\n  --- E. Live API Test --- SKIPPED (pass --url http://localhost:8000)")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULT: {passed}/{total} tests passed")
    if passed == total:
        print("  All integration tests PASSED!")
    else:
        print(f"  {total-passed} test(s) FAILED — see above")
    print("="*65 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

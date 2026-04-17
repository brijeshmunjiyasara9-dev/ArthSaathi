"""
02_build_household_dataset.py
-----------------------------
Step 2: Read cleaned parquets, aggregate POI, join INC+CON+POI,
        compute stress labels and ratio features, save final dataset.

Joins INC+CON MONTH BY MONTH (lowest RAM footprint) then concatenates.
POI aggregation is cached after first run.

Usage:
    python scripts/02_build_household_dataset.py --root D:/Project
"""

import argparse
import gc
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────────────

JOIN_KEYS_COMMON = [
    'household_id', 'state', 'homogeneous_region', 'district',
    'region_type', 'stratum', 'month_slot',
]
JOIN_KEYS_WITH_MONTH = JOIN_KEYS_COMMON + ['reference_month']

EDU_ORDER = [
    'Illiterate', 'Literate but no formal schooling', 'Below Primary',
    'Primary', 'Middle', 'Secondary', 'Higher Secondary',
    'Diploma', 'Graduate', 'Post Graduate', 'Doctorate',
]
EDU_RANK = {v: i for i, v in enumerate(EDU_ORDER)}
EPS = 1e-9

INCOME_SOURCE_COLS = [
    'income_all_members_from_wages',
    'income_household_from_rent',
    'income_household_from_self_production',
    'income_household_from_private_transfers',
    'income_household_from_business_profit',
]


# ─── Match INC and CON files by month-date ────────────────────────────────────

def match_files_by_date(inc_folder: Path, con_folder: Path) -> list:
    """Return list of (inc_path, con_path) pairs matched by the YYYYMMDD date stamp."""
    def date_key(p: Path) -> str:
        m = re.search(r'(20\d\d\d\d\d\d)', p.stem)
        return m.group(1) if m else ''

    inc_map = {date_key(f): f for f in inc_folder.glob('*.parquet') if date_key(f)}
    con_map = {date_key(f): f for f in con_folder.glob('*.parquet') if date_key(f)}

    common = sorted(set(inc_map) & set(con_map))
    print(f"  Matched {len(common)} INC+CON month pairs")
    return [(inc_map[d], con_map[d]) for d in common]


# ─── POI aggregation (vectorized, cached) ─────────────────────────────────────

def agg_one_poi_file(f: Path) -> pd.DataFrame:
    df = pd.read_parquet(f)
    if 'education_rank' not in df.columns and 'education_level' in df.columns:
        df['education_rank'] = df['education_level'].map(EDU_RANK).astype('Int64')

    gkeys = [k for k in JOIN_KEYS_COMMON if k in df.columns]

    src_out = [
        ('is_healthy',                 'is_healthy_hh_min',                 'min'),
        ('is_hospitalised',            'is_hospitalised_hh_any',            'max'),
        ('is_on_regular_medication',   'is_on_regular_medication_hh_any',   'max'),
        ('has_bank_account',           'has_bank_account_hh_any',           'max'),
        ('has_health_insurance',       'has_health_insurance_hh_any',       'max'),
        ('has_life_insurance',         'has_life_insurance_hh_any',         'max'),
        ('has_provident_fund_account', 'has_provident_fund_account_hh_any', 'max'),
        ('has_credit_card',            'has_credit_card_hh_any',            'max'),
        ('has_demat_account',          'has_demat_account_hh_any',          'max'),
        ('has_mobile_phone',           'has_mobile_phone_hh_any',           'max'),
        ('age_years',                  'age_years_hh_mean',                 'mean'),
        ('education_rank',             'education_rank_hh_max',             'max'),
    ]
    agg_d     = {src: func for src, _, func in src_out if src in df.columns}
    rename_d  = {src: out  for src, out, _  in src_out if src in df.columns}

    result = df.groupby(gkeys)[list(agg_d)].agg(agg_d).reset_index()
    result.rename(columns=rename_d, inplace=True)

    for src, out in [('gender', 'gender_hh_mode'),
                     ('occupation_type', 'occupation_type_hh_mode'),
                     ('education_level', 'education_level_hh_mode')]:
        if src in df.columns:
            first_df = (df[df[src].notna()]
                        .groupby(gkeys, as_index=False)[src]
                        .first()
                        .rename(columns={src: out}))
            result = result.merge(first_df, on=gkeys, how='left')

    del df
    gc.collect()
    return result


def build_poi_cache(poi_folder: Path, poi_cache: Path) -> pd.DataFrame:
    if poi_cache.exists():
        print(f"  Loading cached POI from {poi_cache.name}")
        return pd.read_parquet(poi_cache)

    print("  Building POI aggregation (file by file, vectorized)...")
    files  = sorted(poi_folder.glob('*.parquet'))
    chunks = []
    for i, f in enumerate(files):
        print(f"    [{i+1}/{len(files)}] {f.name}", flush=True)
        chunks.append(agg_one_poi_file(f))

    print("  Concatenating and re-aggregating...")
    df_all = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    gkeys = [k for k in JOIN_KEYS_COMMON if k in df_all.columns]
    num_agg = {c: f for c, f in {
        'is_healthy_hh_min': 'min', 'is_hospitalised_hh_any': 'max',
        'is_on_regular_medication_hh_any': 'max', 'has_bank_account_hh_any': 'max',
        'has_health_insurance_hh_any': 'max', 'has_life_insurance_hh_any': 'max',
        'has_provident_fund_account_hh_any': 'max', 'has_credit_card_hh_any': 'max',
        'has_demat_account_hh_any': 'max', 'has_mobile_phone_hh_any': 'max',
        'age_years_hh_mean': 'mean', 'education_rank_hh_max': 'max',
    }.items() if c in df_all.columns}

    df_poi = df_all.groupby(gkeys, as_index=False).agg(num_agg)
    for cat in ['gender_hh_mode', 'occupation_type_hh_mode', 'education_level_hh_mode']:
        if cat in df_all.columns:
            first_df = (df_all[df_all[cat].notna()]
                        .groupby(gkeys, as_index=False)[cat].first())
            df_poi = df_poi.merge(first_df, on=gkeys, how='left')

    del df_all; gc.collect()
    df_poi.to_parquet(poi_cache, index=False)
    print(f"  POI cached: {len(df_poi):,} rows")
    return df_poi


# ─── Join one month pair ───────────────────────────────────────────────────────

def join_one_month(inc_path: Path, con_path: Path,
                   df_poi: pd.DataFrame, poi_keys: list) -> pd.DataFrame:
    df_inc = pd.read_parquet(inc_path)
    df_con = pd.read_parquet(con_path)

    jkeys = [k for k in JOIN_KEYS_WITH_MONTH
             if k in df_inc.columns and k in df_con.columns]

    # Drop from CON any columns already in INC (except join keys) — avoids dupes
    con_extra = [c for c in df_con.columns if c not in jkeys and c in df_inc.columns]
    df_con.drop(columns=con_extra, inplace=True, errors='ignore')

    df = df_inc.merge(df_con, on=jkeys, how='inner', suffixes=('_inc', '_con'))
    del df_inc, df_con; gc.collect()

    # Clean suffix collisions
    to_drop, to_rename = [], {}
    for col in list(df.columns):
        if col.endswith('_inc') or col.endswith('_con'):
            base = col[:-4]
            if base in df.columns:
                to_drop.append(col)
            else:
                to_rename[col] = base
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    if to_rename:
        df.rename(columns=to_rename, inplace=True)

    # Final dedup by first-occurrence
    seen, keep = set(), []
    for c in df.columns:
        if c not in seen:
            keep.append(c); seen.add(c)
    df = df[keep]

    # POI join
    df = df.merge(df_poi, on=poi_keys, how='left')

    # Dedup again after POI join
    seen2, keep2 = set(), []
    for c in df.columns:
        if c not in seen2:
            keep2.append(c); seen2.add(c)
    return df[keep2]



# ─── Stress labels ────────────────────────────────────────────────────────────

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    te = df.get('total_expenditure_adjusted', pd.Series(np.nan, index=df.index))
    ti = df.get('total_income', pd.Series(np.nan, index=df.index))

    if 'total_expenditure_adjusted' in df.columns and 'total_income' in df.columns:
        df['financial_stress'] = ((te - ti) / (ti + EPS) > 0.1).astype('Int8')
    else:
        df['financial_stress'] = pd.NA

    if 'exp_food_adjusted' in df.columns and 'total_expenditure_adjusted' in df.columns:
        df['food_stress'] = (df['exp_food_adjusted'] / (te + EPS) > 0.5).astype('Int8')
    else:
        df['food_stress'] = pd.NA

    if 'exp_all_emis' in df.columns and 'total_income' in df.columns:
        df['debt_stress'] = (df['exp_all_emis'] / (ti + EPS) > 0.3).astype('Int8')
    else:
        df['debt_stress'] = pd.NA

    hosp = df.get('is_hospitalised_hh_any', pd.Series(0, index=df.index)).fillna(0)
    meds = df.get('is_on_regular_medication_hh_any', pd.Series(0, index=df.index)).fillna(0)
    df['health_stress'] = ((hosp > 0) | (meds > 0)).astype('Int8')

    df['composite_stress_score'] = (
        df['financial_stress'].fillna(0).astype(int) +
        df['food_stress'].fillna(0).astype(int) +
        df['debt_stress'].fillna(0).astype(int) +
        df['health_stress'].fillna(0).astype(int)
    ).astype('Int8')
    return df


# ─── Ratio features ───────────────────────────────────────────────────────────

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    te = df.get('total_expenditure_adjusted', pd.Series(0.0, index=df.index)).fillna(0)
    ti = df.get('total_income', pd.Series(0.0, index=df.index)).fillna(0)

    df['emi_to_income_ratio']        = df.get('exp_all_emis', 0.0).fillna(0) / (ti + EPS)
    df['food_to_expense_ratio']      = df.get('exp_food_adjusted', 0.0).fillna(0) / (te + EPS)
    df['health_to_expense_ratio']    = df.get('exp_health', 0.0).fillna(0) / (te + EPS)
    df['education_to_expense_ratio'] = df.get('exp_education', 0.0).fillna(0) / (te + EPS)

    rec  = df.get('exp_recreation', pd.Series(0.0, index=df.index)).fillna(0)
    vac  = df.get('exp_vacation',   pd.Series(0.0, index=df.index)).fillna(0)
    rest = df.get('exp_restaurants_adjusted', pd.Series(0.0, index=df.index)).fillna(0)
    df['discretionary_to_expense_ratio'] = (rec + vac + rest) / (te + EPS)
    df['savings_proxy'] = ti - te

    present = [(df[c].fillna(0) > 0) for c in INCOME_SOURCE_COLS if c in df.columns]
    df['income_diversity_score'] = (sum(present).astype('Int8') if present else 0)
    return df


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='D:/Project')
    args = parser.parse_args()
    root = Path(args.root)

    proc    = root / 'processed'
    out_dir = root / 'processed2' / 'household_joined'
    out_dir.mkdir(parents=True, exist_ok=True)

    poi_cache = out_dir / 'poi_household_agg.parquet'
    final_out = out_dir / 'household_stress_dataset.parquet'

    # ── POI aggregation (file-by-file, cached) ────────────────────────────────
    print("\n=== Step A: POI household aggregation ===")
    df_poi = build_poi_cache(proc / 'people_of_india', poi_cache)
    poi_keys = [k for k in JOIN_KEYS_COMMON if k in df_poi.columns]
    print(f"  POI shape: {df_poi.shape}")

    # ── Match INC+CON by month date ───────────────────────────────────────────
    print("\n=== Step B: Matching INC + CON files by month ===")
    pairs = match_files_by_date(proc / 'income_pyramid', proc / 'consumption_pyramid')

    # ── Join month by month, save intermediate parquets ───────────────────────
    interim_dir = out_dir / 'interim_months'
    interim_dir.mkdir(exist_ok=True)

    print(f"\n=== Step C: Processing {len(pairs)} month pairs ===")
    saved_months = []
    for i, (inc_p, con_p) in enumerate(pairs):
        date_str = re.search(r'(20\d\d\d\d\d\d)', inc_p.stem).group(1)
        out_month = interim_dir / f'month_{date_str}.parquet'

        if out_month.exists():
            print(f"  [{i+1:3}/{len(pairs)}] SKIP {date_str}", flush=True)
            saved_months.append(out_month)
            continue

        print(f"  [{i+1:3}/{len(pairs)}] {date_str} ...", end=' ', flush=True)
        df_m = join_one_month(inc_p, con_p, df_poi, poi_keys)
        df_m = compute_labels(df_m)
        df_m = compute_ratios(df_m)
        df_m.drop(columns=[c for c in ['total_income', 'total_expenditure_adjusted']
                            if c in df_m.columns], inplace=True)
        df_m.to_parquet(out_month, index=False)
        print(f"{len(df_m):,} rows")
        saved_months.append(out_month)
        del df_m; gc.collect()

    # ── Step D: Stream-merge with unified schema ────────────────────────────────
    print(f"\n=== Step D: Stream-merging {len(saved_months)} monthly parquets ===")
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Pass 1: collect union of all schemas
    print("  Pass 1: collecting union schema...")
    all_schemas = [pq.read_schema(str(p)) for p in saved_months]
    # Build unified schema: for each name, take first type seen
    name_type = {}
    for s in all_schemas:
        for field in s:
            if field.name not in name_type:
                name_type[field.name] = field.type
    unified_schema = pa.schema([pa.field(n, t) for n, t in name_type.items()])
    print(f"  Union schema: {len(unified_schema)} columns")

    # Pass 2: write each file using the unified schema (cast + add missing cols)
    print("  Pass 2: streaming write...")
    writer = pq.ParquetWriter(str(final_out), unified_schema, compression='snappy')
    total_rows = 0
    label_sums, label_counts = {}, {}

    for p in saved_months:
        tbl = pq.read_table(p)
        # Add missing columns as null
        for field in unified_schema:
            if field.name not in tbl.schema.names:
                null_col = pa.array([None] * len(tbl), type=field.type)
                tbl = tbl.append_column(field, null_col)
        # Reorder to match unified schema
        tbl = tbl.select(unified_schema.names)
        # Cast to unified types where needed
        cast_cols = []
        for field in unified_schema:
            col = tbl.column(field.name)
            if col.type != field.type:
                try:
                    col = col.cast(field.type, safe=False)
                except Exception:
                    col = pa.array([None] * len(tbl), type=field.type)
            cast_cols.append(col)
        tbl = pa.table(
            {field.name: cast_cols[i] for i, field in enumerate(unified_schema)},
            schema=unified_schema
        )
        writer.write_table(tbl)
        total_rows += len(tbl)

        # Accumulate label stats
        for lbl in ['financial_stress', 'food_stress', 'debt_stress',
                    'health_stress', 'composite_stress_score']:
            if lbl in tbl.schema.names:
                col_list = tbl.column(lbl).to_pylist()
                non_null = [v for v in col_list if v is not None]
                label_sums[lbl]   = label_sums.get(lbl, 0) + sum(non_null)
                label_counts[lbl] = label_counts.get(lbl, 0) + len(non_null)
        del tbl
        gc.collect()

    writer.close()
    print(f"  Total rows written: {total_rows:,}")
    for lbl in ['financial_stress', 'food_stress', 'debt_stress', 'health_stress']:
        if label_counts.get(lbl, 0) > 0:
            pct = label_sums[lbl] / label_counts[lbl] * 100
            print(f"  {lbl}: {pct:.1f}% stressed")
    if label_counts.get('composite_stress_score', 0) > 0:
        mean = label_sums['composite_stress_score'] / label_counts['composite_stress_score']
        print(f"  composite mean: {mean:.2f}")

    print(f"\nDone! {total_rows:,} rows saved -> {final_out}")


if __name__ == '__main__':
    main()

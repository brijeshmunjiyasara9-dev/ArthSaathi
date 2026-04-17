"""
01_clean_data.py
----------------
Step 1: Read raw ZIP/CSV files for Income Pyramid (INC), People of India (POI),
        and Consumption Pyramid (CON). Apply cleaning rules from the spec and
        save cleaned datasets to D:/Project/processed/<dataset>/.

Usage:
    python scripts/01_clean_data.py --root D:/Project
"""

import argparse
import zipfile
import io
import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_NULLS = [
    'Data Not Available', 'Not Applicable', 'Not applicable',
    'data not available', 'not applicable', 'DK',
]

EDU_ORDER = [
    'Illiterate', 'Literate but no formal schooling', 'Below Primary',
    'Primary', 'Middle', 'Secondary', 'Higher Secondary',
    'Diploma', 'Graduate', 'Post Graduate', 'Doctorate',
]
EDU_RANK = {v: i for i, v in enumerate(EDU_ORDER)}

# ─── Column mappings: raw ALL-CAPS → clean snake_case ─────────────────────────

INC_RENAME = {
    'HH_ID':                        'household_id',
    'STATE':                        'state',
    'HR':                           'homogeneous_region',
    'DISTRICT':                     'district',
    'REGION_TYPE':                  'region_type',
    'STRATUM':                      'stratum',
    'MONTH_SLOT':                   'month_slot',
    'MONTH':                        'reference_month',
    'AGE_GROUP':                    'age_group',
    'OCCUPATION_GROUP':             'occupation_group',
    'EDU_GROUP':                    'education_group',
    'GENDER_GROUP':                 'gender_group',
    'SIZE_GROUP':                   'household_size_group',
    'TOT_INC':                      'total_income',
    'INC_OF_ALL_MEMS_FRM_WAGES':    'income_all_members_from_wages',
    'INC_OF_HH_FRM_RENT':           'income_household_from_rent',
    'INC_OF_HH_FRM_SELF_PRODN':     'income_household_from_self_production',
    'INC_OF_HH_FRM_PVT_TRF':        'income_household_from_private_transfers',
    'INC_OF_HH_FRM_BIZ_PROFIT':     'income_household_from_business_profit',
    'PSU_ID':                       'primary_sampling_unit_id',
    'RESPONSE_STATUS':              'response_status',
}

INC_KEEP = [
    'household_id', 'state', 'homogeneous_region', 'district', 'region_type',
    'stratum', 'month_slot', 'reference_month', 'age_group', 'occupation_group',
    'education_group', 'gender_group', 'household_size_group', 'total_income',
    'income_all_members_from_wages', 'income_household_from_rent',
    'income_household_from_self_production', 'income_household_from_private_transfers',
    'income_household_from_business_profit',
]

INC_SENTINEL_COLS = [
    'total_income', 'income_all_members_from_wages', 'income_household_from_rent',
    'income_household_from_self_production', 'income_household_from_private_transfers',
    'income_household_from_business_profit',
]

POI_RENAME = {
    'HH_ID':            'household_id',
    'MEM_ID':           'member_id',
    'STATE':            'state',
    'HR':               'homogeneous_region',
    'DISTRICT':         'district',
    'REGION_TYPE':      'region_type',
    'STRATUM':          'stratum',
    'MONTH_SLOT':       'month_slot',
    'GENDER':           'gender',
    'AGE_YRS':          'age_years',
    'RELATION_WITH_HOH':'relation_with_head_of_household',
    'RELIGION':         'religion',
    'CASTE':            'caste',
    'CASTE_CATEGORY':   'caste_category',
    'LITERACY':         'is_literate',
    'EDU':              'education_level',
    'NATURE_OF_OCCUPATION': 'occupation_type',
    'IS_HEALTHY':           'is_healthy',
    'IS_ON_ORAL_MEDICATION':'is_on_regular_medication',
    'IS_HOSPITALISED':      'is_hospitalised',
    'HAS_BANK_AC':          'has_bank_account',
    'HAS_CREDITCARD':       'has_credit_card',
    'HAS_KISAN_CREDITCARD': 'has_kisan_credit_card',
    'HAS_DEMAT_AC':         'has_demat_account',
    'HAS_PF_AC':            'has_provident_fund_account',
    'HAS_LIC':              'has_life_insurance',
    'HAS_HEALTH_INS':       'has_health_insurance',
    'HAS_MOBILE':           'has_mobile_phone',
    'RESPONSE_STATUS':      'response_status',
}

POI_KEEP = [
    'household_id', 'state', 'homogeneous_region', 'district', 'region_type',
    'stratum', 'month_slot', 'member_id', 'gender', 'age_years',
    'relation_with_head_of_household', 'religion', 'caste', 'caste_category',
    'is_literate', 'education_level', 'occupation_type',
    'is_healthy', 'is_on_regular_medication', 'is_hospitalised',
    'has_bank_account', 'has_credit_card', 'has_kisan_credit_card',
    'has_demat_account', 'has_provident_fund_account', 'has_life_insurance',
    'has_health_insurance', 'has_mobile_phone',
]

POI_BINARY_COLS = [
    'is_literate', 'is_healthy', 'is_on_regular_medication', 'is_hospitalised',
    'has_bank_account', 'has_credit_card', 'has_kisan_credit_card',
    'has_demat_account', 'has_provident_fund_account', 'has_life_insurance',
    'has_health_insurance', 'has_mobile_phone',
]

CON_RENAME = {
    'HH_ID':                            'household_id',
    'STATE':                            'state',
    'HR':                               'homogeneous_region',
    'DISTRICT':                         'district',
    'REGION_TYPE':                      'region_type',
    'STRATUM':                          'stratum',
    'MONTH_SLOT':                       'month_slot',
    'MONTH':                            'reference_month',
    'AGE_GROUP':                        'age_group',
    'OCCUPATION_GROUP':                 'occupation_group',
    'EDU_GROUP':                        'education_group',
    'GENDER_GROUP':                     'gender_group',
    'SIZE_GROUP':                       'household_size_group',
    'ADJ_TOT_EXP':                      'total_expenditure_adjusted',
    'ADJ_M_EXP_FOOD':                   'exp_food_adjusted',
    'ADJ_M_EXP_EDIBLE_OILS':            'exp_edible_oils_adjusted',
    'ADJ_M_EXP_VEGGIE_N_FRUITS':        'exp_vegetables_and_fruits_adjusted',
    'ADJ_M_EXP_VEGGIE_N_WET_SPICES':    'exp_vegetables_and_wet_spices_adjusted',
    'ADJ_M_EXP_FRUITS':                 'exp_fruits_adjusted',
    'ADJ_M_EXP_POTATOES_N_ONIONS':      'exp_potatoes_and_onions_adjusted',
    'ADJ_M_EXP_MILK_N_MILK_PRDS':       'exp_milk_and_milk_products_adjusted',
    'ADJ_M_EXP_BREAD':                  'exp_bread_adjusted',
    'ADJ_M_EXP_BISCUITS':               'exp_biscuits_adjusted',
    'ADJ_M_EXP_SALTY_SNACKS':           'exp_salty_snacks_adjusted',
    'ADJ_M_EXP_CHOCLATE_CAKE_ICECREAM': 'exp_chocolate_cake_icecream_adjusted',
    'ADJ_M_EXP_MEAT_EGGS_N_FISH':       'exp_meat_eggs_and_fish_adjusted',
    'ADJ_M_EXP_BEVERAGES_N_WATER':      'exp_beverages_and_water_adjusted',
    'ADJ_M_EXP_BOTTLED_WATER':          'exp_bottled_water_adjusted',
    'ADJ_M_EXP_INTOXICANTS':            'exp_intoxicants_adjusted',
    'ADJ_M_EXP_CIGARETTES_N_TOBACCO':   'exp_cigarettes_and_tobacco_adjusted',
    'M_EXP_CEREALS_N_PULSES':           'exp_cereals_and_pulses',
    'M_EXP_DRY_SPICES':                 'exp_dry_spices',
    'M_EXP_NOODLES_N_FLAKES':           'exp_noodles_and_flakes',
    'M_EXP_JAM_KETCHUP_PICKLES':        'exp_jam_ketchup_pickles',
    'M_EXP_HEALTH_SUPPLEMENTS':         'exp_health_supplements',
    'M_EXP_READY_TO_EAT_FOOD':          'exp_ready_to_eat_food',
    'M_EXP_TEA_COFFEE':                 'exp_tea_and_coffee',
    'M_EXP_SUGAR_N_OTH_SWEETENERS':     'exp_sugar_and_sweeteners',
    'M_EXP_OTH_FOODS':                  'exp_other_food',
    'M_EXP_LIQUOR':                     'exp_liquor',
    # additional clean-name columns
    'M_EXP_CLOTHING_N_FOOTWEAR':        'exp_clothing_and_footwear',
    'M_EXP_CLOTHING':                   'exp_clothing',
    'M_EXP_FOOTWEAR':                   'exp_footwear',
    'M_EXP_CLOTHING_ACCESSORIES':       'exp_clothing_accessories',
    'M_EXP_COSMETIC_N_TOILETRIES':      'exp_cosmetics_and_toiletries',
    'M_EXP_DENTAL_CARE_PRDS':           'exp_dental_care_products',
    'M_EXP_BATHING_SOAP':               'exp_bathing_soap',
    'M_EXP_COSMETICS':                  'exp_cosmetics',
    'M_EXP_FACE_WASH':                  'exp_face_wash',
    'M_EXP_SHAVING_ARTICLES':           'exp_shaving_articles',
    'M_EXP_HAIR_OIL':                   'exp_hair_oil',
    'M_EXP_SHAMPOO_N_CONDITIONER':      'exp_shampoo_and_conditioner',
    'M_EXP_POWDER':                     'exp_powder',
    'M_EXP_CREAMS':                     'exp_creams',
    'M_EXP_DEODORANTS_N_PERFUMES':      'exp_deodorants_and_perfumes',
    'M_EXP_ALL_TYPES_OF_DETERGENT':     'exp_detergent_all_types',
    'M_EXP_APPLIANCES':                 'exp_appliances',
    'ADJ_M_EXP_RESTAURANTS':            'exp_restaurants_adjusted',
    'M_EXP_RECREATION':                 'exp_recreation',
    'M_EXP_ENTMT':                      'exp_entertainment',
    'M_EXP_TOYS':                       'exp_toys',
    'M_EXP_BILLS_N_RENT':               'exp_bills_and_rent',
    'M_EXP_HOUSE_RENT':                 'exp_house_rent',
    'M_EXP_WATER_CHARGES':              'exp_water_charges',
    'M_EXP_SOCIETY_CHARGES':            'exp_society_charges',
    'M_EXP_OTH_TAXES':                  'exp_other_taxes',
    'M_EXP_COOKING_FUEL':               'exp_cooking_fuel',
    'ADJ_M_EXP_PETROL_N_CNG':           'exp_petrol_and_cng_adjusted',
    'ADJ_M_EXP_DIESEL':                 'exp_diesel_adjusted',
    'M_EXP_ELECTRICITY':                'exp_electricity',
    'ADJ_M_EXP_TRANSPORT':              'exp_transport_adjusted',
    'ADJ_M_EXP_AUTORICKSHAW_N_CAB':     'exp_autorickshaw_and_cab_adjusted',
    'M_EXP_AIRFARE':                    'exp_airfare',
    'M_EXP_COMMUNICATION_N_INFO':       'exp_communication_and_information',
    'M_EXP_CELL_PHONE':                 'exp_mobile_phone',
    'M_EXP_CABLE_TV':                   'exp_cable_tv',
    'M_EXP_INTERNET':                   'exp_internet',
    'M_EXP_NEWSPAPERS_N_MAGAZINES':     'exp_newspapers_and_magazines',
    'M_EXP_EDU':                        'exp_education',
    'M_EXP_SCHOOL_COLLEGE_FEES':        'exp_school_and_college_fees',
    'M_EXP_PVT_TUITION_FEES':           'exp_private_tuition_fees',
    'M_EXP_HOBBY_CLASSES':              'exp_hobby_classes',
    'M_EXP_ADDITIONAL_PROF_EDU':        'exp_additional_professional_education',
    'M_EXP_HEALTH':                     'exp_health',
    'M_EXP_MEDICINES':                  'exp_medicines',
    'M_EXP_HOSPITALISATION_FEES':       'exp_hospitalisation_fees',
    'M_EXP_HEALTH_INS_PREMIUM':         'exp_health_insurance_premium',
    'M_EXP_HEALTH_ENHANCEMENT':         'exp_health_enhancement',
    'M_EXP_ALL_EMIS':                   'exp_all_emis',
    'M_EXP_EMI_FOR_HOUSE':              'exp_emi_house',
    'M_EXP_EMI_FOR_VEHICLE':            'exp_emi_vehicle',
    'M_EXP_MISC':                       'exp_miscellaneous',
    'M_EXP_DOMESTIC_HELP':              'exp_domestic_help',
    'M_EXP_MOTOR_VEHICLE_REPAIRS':      'exp_motor_vehicle_repairs',
    'M_EXP_REMITTANCES_SENT':           'exp_remittances_sent',
    'M_EXP_SOCIAL_OBLIGATIONS':         'exp_social_obligations',
    'M_EXP_RELIGIOUS_OBLIGATIONS':      'exp_religious_obligations',
    'M_EXP_GENERAL_INS':                'exp_general_insurance',
    'M_EXP_VACATION':                   'exp_vacation',
    'M_EXP_FURNITURE_N_FURNISHINGS':    'exp_furniture_and_furnishings',
    'M_EXP_PAINTING_N_RENOVATION':      'exp_painting_and_renovation',
    'RESPONSE_STATUS':                  'response_status',
}


# ─── Helper functions ─────────────────────────────────────────────────────────

def yn_to_int(series: pd.Series) -> pd.Series:
    """Convert Y/N string column to 0/1 Int64, NaN for anything else."""
    s = series.astype('string').str.strip().str.upper()
    out = pd.array([pd.NA] * len(s), dtype='Int64')
    out[s == 'Y'] = 1
    out[s == 'N'] = 0
    return pd.Series(out, index=series.index)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace string sentinel values with NaN in all object columns."""
    str_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in str_cols:
        df[col] = df[col].replace(STRING_NULLS, np.nan)
    return df


def parse_month_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Parse month_slot to datetime."""
    df['month_slot'] = pd.to_datetime(df['month_slot'], format='%b %Y', errors='coerce')
    return df


def list_zip_files(folder: Path, pattern: str) -> list:
    return sorted(folder.glob(pattern))


def read_zip_csv(zip_path: Path) -> pd.DataFrame:
    """Read the first (only) CSV inside a zip file."""
    with zipfile.ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            return pd.read_csv(f, low_memory=False)


# ─── INC cleaning ─────────────────────────────────────────────────────────────

def clean_inc(root: Path) -> None:
    print("\n=== Cleaning Income Pyramid (INC) ===")
    src = root / 'Dataset' / 'Income_Pyramid'
    dst = root / 'processed' / 'income_pyramid'
    dst.mkdir(parents=True, exist_ok=True)

    files = list_zip_files(src, '*.zip')
    print(f"  Found {len(files)} INC zip files")

    for i, zf in enumerate(files):
        out_path = dst / (zf.stem + '.parquet')
        if out_path.exists():
            print(f"  [{i+1}/{len(files)}] SKIP (exists): {zf.name}")
            continue

        print(f"  [{i+1}/{len(files)}] Processing: {zf.name}", end=' ... ', flush=True)
        df = read_zip_csv(zf)

        # Filter accepted responses
        if 'RESPONSE_STATUS' in df.columns:
            df = df[df['RESPONSE_STATUS'] == 'Accepted'].copy()

        # Rename columns (keep only those present)
        rename_map = {k: v for k, v in INC_RENAME.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Sentinel -99 → NaN on income cols
        for col in INC_SENTINEL_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace(-99, np.nan)

        # String nulls → NaN
        df = clean_strings(df)

        # Parse month_slot
        if 'month_slot' in df.columns:
            df = parse_month_slot(df)

        # Keep only desired columns
        keep = [c for c in INC_KEEP if c in df.columns]
        df = df[keep]

        df.to_parquet(out_path, index=False)
        print(f"saved {len(df):,} rows -> {out_path.name}")
        del df
        gc.collect()

    print("  INC cleaning complete.")


# ─── POI cleaning ─────────────────────────────────────────────────────────────

def clean_poi(root: Path) -> None:
    print("\n=== Cleaning People of India (POI) ===")
    src = root / 'Dataset' / 'People_of_India'
    dst = root / 'processed' / 'people_of_india'
    dst.mkdir(parents=True, exist_ok=True)

    files = list_zip_files(src, '*.zip')
    print(f"  Found {len(files)} POI zip files")

    for i, zf in enumerate(files):
        out_path = dst / (zf.stem + '.parquet')
        if out_path.exists():
            print(f"  [{i+1}/{len(files)}] SKIP (exists): {zf.name}")
            continue

        print(f"  [{i+1}/{len(files)}] Processing: {zf.name}", end=' ... ', flush=True)
        df = read_zip_csv(zf)

        # Filter accepted responses
        if 'RESPONSE_STATUS' in df.columns:
            df = df[df['RESPONSE_STATUS'] == 'Accepted'].copy()

        # Rename columns
        rename_map = {k: v for k, v in POI_RENAME.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Sentinel -100 → NaN for age
        if 'age_years' in df.columns:
            df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce').replace(-100, np.nan)

        # String nulls → NaN
        df = clean_strings(df)

        # Parse month_slot
        if 'month_slot' in df.columns:
            df = parse_month_slot(df)

        # Binary Y/N → Int64
        for col in POI_BINARY_COLS:
            if col in df.columns:
                df[col] = yn_to_int(df[col])

        # Education ordinal
        if 'education_level' in df.columns:
            df['education_rank'] = df['education_level'].map(EDU_RANK).astype('Int64')

        # Keep only desired columns
        keep = [c for c in POI_KEEP + ['education_rank'] if c in df.columns]
        df = df[keep]

        df.to_parquet(out_path, index=False)
        print(f"saved {len(df):,} rows -> {out_path.name}")
        del df
        gc.collect()

    print("  POI cleaning complete.")


# ─── CON cleaning ─────────────────────────────────────────────────────────────

def clean_con(root: Path) -> None:
    print("\n=== Cleaning Consumption Pyramid (CON) ===")
    src = root / 'Dataset' / 'Consumption_Pyramid'
    dst = root / 'processed' / 'consumption_pyramid'
    dst.mkdir(parents=True, exist_ok=True)

    files = list_zip_files(src, 'consumption_pyramids_*.zip')
    print(f"  Found {len(files)} CON zip files")

    for i, zf in enumerate(files):
        out_path = dst / (zf.stem + '.parquet')
        if out_path.exists():
            print(f"  [{i+1}/{len(files)}] SKIP (exists): {zf.name}")
            continue

        print(f"  [{i+1}/{len(files)}] Processing: {zf.name}", end=' ... ', flush=True)
        df = read_zip_csv(zf)

        # Filter accepted responses
        if 'RESPONSE_STATUS' in df.columns:
            df = df[df['RESPONSE_STATUS'] == 'Accepted'].copy()

        # Rename columns (keep only those that exist)
        rename_map = {k: v for k, v in CON_RENAME.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Sentinel -99 → NaN on all exp_ columns + total_expenditure_adjusted
        exp_cols = [c for c in df.columns if c.startswith('exp_')]
        if 'total_expenditure_adjusted' in df.columns:
            exp_cols.append('total_expenditure_adjusted')
        for col in exp_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-99, np.nan)

        # String nulls → NaN
        df = clean_strings(df)

        # Parse month_slot
        if 'month_slot' in df.columns:
            df = parse_month_slot(df)

        # Keep only renamed columns (drop raw admin cols)
        keep = [c for c in df.columns if c in set(CON_RENAME.values())]
        keep = [c for c in keep if c != 'response_status']  # drop after filter
        df = df[keep]

        df.to_parquet(out_path, index=False)
        print(f"saved {len(df):,} rows -> {out_path.name}")
        del df
        gc.collect()

    print("  CON cleaning complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Step 1: Clean raw CMIE data')
    parser.add_argument('--root', type=str, default='D:/Project',
                        help='Root project directory')
    args = parser.parse_args()
    root = Path(args.root)

    clean_inc(root)
    clean_poi(root)
    clean_con(root)

    print("\nDone! All datasets cleaned and saved to processed/")


if __name__ == '__main__':
    main()

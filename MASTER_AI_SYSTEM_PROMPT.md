# MASTER AI SYSTEM PROMPT
## Full Pipeline: CMIE Data → Stress Models → Chatbot → Web System
**Version:** 1.0 | **Author:** For AI Agent Use

---

# PART 0 — CONTEXT & MISSION

You are building a complete end-to-end financial stress analysis system for Indian households using CMIE (Centre for Monitoring Indian Economy) survey data. The system has five major components:

1. **Data Preparation** — Clean raw parquet files from three CMIE datasets
2. **Feature Engineering** — Build a unified household-level dataset with stress labels
3. **Model Training** — Train multiple stress prediction models and save them
4. **Chatbot** — A conversational AI advisor using trained models + OpenAI API
5. **Web System** — Full-stack web application with database, API, and frontend

The final product is a web chatbot that:
- Asks users for their financial/household information in a friendly conversation
- Uses trained ML models to assess their financial, food, debt, and health stress
- Uses an LLM (OpenAI) to give personalized, contextual financial advice
- Stores user sessions, assessments, and history in a database

---

# PART 1 — RAW DATA: SOURCE, STRUCTURE & PATH

## 1.1 Dataset Locations (Windows Paths)

```
D:\Project\Dataset\Income_Pyramid\         ← INC raw parquet files
D:\Project\Dataset\People_of_India\        ← POI raw parquet files
D:\Project\Dataset\Consumption_Pyramid\    ← CON raw parquet files
```

Each folder contains many `.parquet` files (partitioned by survey wave/month).

## 1.2 Dataset Descriptions

### A) People of India (POI) — Person-level survey
- **Granularity:** One row per household member per survey wave
- **Purpose:** Demographics, education, health, financial inclusion indicators
- **Key join keys:** `household_id`, `state`, `homogeneous_region`, `district`, `region_type`, `stratum`, `month_slot`

**Columns to KEEP (after cleaning):**
```
household_id, state, homogeneous_region, district, region_type, stratum, month_slot,
member_id, gender, age_years, relation_with_head_of_household,
religion, caste, caste_category, is_literate, education_level, occupation_type,
is_healthy, is_on_regular_medication, is_hospitalised,
has_bank_account, has_credit_card, has_kisan_credit_card, has_demat_account,
has_provident_fund_account, has_life_insurance, has_health_insurance, has_mobile_phone
```

**Binary columns** (Y/N → 1/0):
`is_literate, is_healthy, is_on_regular_medication, is_hospitalised, has_bank_account,
has_credit_card, has_kisan_credit_card, has_demat_account, has_provident_fund_account,
has_life_insurance, has_health_insurance, has_mobile_phone`

### B) Income Pyramid (INC) — Household income survey
- **Granularity:** One row per household per reference month
- **Purpose:** Income levels and sources
- **Key join keys:** `household_id`, `state`, `homogeneous_region`, `district`, `region_type`, `stratum`, `month_slot`, `reference_month`

**Columns to KEEP:**
```
household_id, state, homogeneous_region, district, region_type, stratum,
month_slot, reference_month, age_group, occupation_group, education_group,
gender_group, household_size_group, total_income,
income_all_members_from_wages, income_household_from_rent,
income_household_from_self_production, income_household_from_private_transfers,
income_household_from_business_profit
```

**Sentinel value:** `-99` means missing → replace with `NaN`

### C) Consumption Pyramid (CON) — Household expenditure survey
- **Granularity:** One row per household per reference month
- **Purpose:** Monthly spending across 80+ categories
- **Key join keys:** Same as INC (includes `reference_month`)

**Raw column naming quirk:** Many expense columns are named with prefixes `ADJ_M_EXP_*` and `M_EXP_*`. These must be renamed to clean snake_case names (see rename map below).

**Columns to KEEP and their RENAME MAP:**

| Raw Column Name | Renamed To |
|---|---|
| `ADJ_M_EXP_FOOD` | `exp_food_adjusted` |
| `ADJ_M_EXP_EDIBLE_OILS` | `exp_edible_oils_adjusted` |
| `ADJ_M_EXP_VEGGIE_N_FRUITS` | `exp_vegetables_and_fruits_adjusted` |
| `ADJ_M_EXP_VEGGIE_N_WET_SPICES` | `exp_vegetables_and_wet_spices_adjusted` |
| `ADJ_M_EXP_FRUITS` | `exp_fruits_adjusted` |
| `ADJ_M_EXP_POTATOES_N_ONIONS` | `exp_potatoes_and_onions_adjusted` |
| `ADJ_M_EXP_MILK_N_MILK_PRDS` | `exp_milk_and_milk_products_adjusted` |
| `ADJ_M_EXP_BREAD` | `exp_bread_adjusted` |
| `ADJ_M_EXP_BISCUITS` | `exp_biscuits_adjusted` |
| `ADJ_M_EXP_SALTY_SNACKS` | `exp_salty_snacks_adjusted` |
| `ADJ_M_EXP_CHOCLATE_CAKE_ICECREAM` | `exp_chocolate_cake_icecream_adjusted` |
| `ADJ_M_EXP_MEAT_EGGS_N_FISH` | `exp_meat_eggs_and_fish_adjusted` |
| `ADJ_M_EXP_BEVERAGES_N_WATER` | `exp_beverages_and_water_adjusted` |
| `ADJ_M_EXP_BOTTLED_WATER` | `exp_bottled_water_adjusted` |
| `ADJ_M_EXP_INTOXICANTS` | `exp_intoxicants_adjusted` |
| `ADJ_M_EXP_CIGARETTES_N_TOBACCO` | `exp_cigarettes_and_tobacco_adjusted` |
| `M_EXP_CEREALS_N_PULSES` | `exp_cereals_and_pulses` |
| `M_EXP_DRY_SPICES` | `exp_dry_spices` |
| `M_EXP_NOODLES_N_FLAKES` | `exp_noodles_and_flakes` |
| `M_EXP_JAM_KETCHUP_PICKLES` | `exp_jam_ketchup_pickles` |
| `M_EXP_HEALTH_SUPPLEMENTS` | `exp_health_supplements` |
| `M_EXP_READY_TO_EAT_FOOD` | `exp_ready_to_eat_food` |
| `M_EXP_TEA_COFFEE` | `exp_tea_and_coffee` |
| `M_EXP_SUGAR_N_OTH_SWEETENERS` | `exp_sugar_and_sweeteners` |
| `M_EXP_OTH_FOODS` | `exp_other_food` |
| `M_EXP_LIQUOR` | `exp_liquor` |

**Additional CON columns to KEEP (already clean names):**
```
total_expenditure_adjusted, exp_clothing_and_footwear, exp_clothing, exp_footwear,
exp_clothing_accessories, exp_cosmetics_and_toiletries, exp_dental_care_products,
exp_bathing_soap, exp_cosmetics, exp_face_wash, exp_shaving_articles, exp_hair_oil,
exp_shampoo_and_conditioner, exp_powder, exp_creams, exp_deodorants_and_perfumes,
exp_detergent_all_types, exp_appliances, exp_restaurants_adjusted, exp_recreation,
exp_entertainment, exp_toys, exp_bills_and_rent, exp_house_rent, exp_water_charges,
exp_society_charges, exp_other_taxes, exp_cooking_fuel, exp_petrol_and_cng_adjusted,
exp_diesel_adjusted, exp_electricity, exp_transport_adjusted,
exp_autorickshaw_and_cab_adjusted, exp_airfare, exp_communication_and_information,
exp_mobile_phone, exp_cable_tv, exp_internet, exp_newspapers_and_magazines,
exp_education, exp_school_and_college_fees, exp_private_tuition_fees,
exp_hobby_classes, exp_additional_professional_education, exp_health, exp_medicines,
exp_hospitalisation_fees, exp_health_insurance_premium, exp_health_enhancement,
exp_all_emis, exp_emi_house, exp_emi_vehicle, exp_miscellaneous, exp_domestic_help,
exp_motor_vehicle_repairs, exp_remittances_sent, exp_social_obligations,
exp_religious_obligations, exp_general_insurance, exp_vacation,
exp_furniture_and_furnishings, exp_painting_and_renovation
```

---

# PART 2 — DATA CLEANING RULES (Apply to ALL datasets)

## Step 1: Drop Non-Response rows
```python
# Filter: keep only accepted responses
df = df[df['response_status'] == 'Accepted']
```

## Step 2: Sentinel values → NaN
```python
# INC: income columns: -99 → NaN
inc_sentinel_cols = ['total_income', 'income_all_members_from_wages', ...]
df[inc_sentinel_cols] = df[inc_sentinel_cols].replace(-99, np.nan)

# CON: all exp_* columns + total_expenditure_adjusted: -99 → NaN
exp_cols = [c for c in df.columns if c.startswith('exp_')] + ['total_expenditure_adjusted']
df[exp_cols] = df[exp_cols].replace(-99, np.nan)

# POI: age_years, age_months: -100 → NaN
df['age_years'] = df['age_years'].replace(-100, np.nan)
```

## Step 3: String null values → NaN
```python
STRING_NULLS = ['Data Not Available', 'Not Applicable', 'Not applicable',
                'data not available', 'not applicable', 'DK']
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].replace(STRING_NULLS, np.nan)
```

## Step 4: Parse month_slot to datetime
```python
df['month_slot'] = pd.to_datetime(df['month_slot'], format='%b %Y', errors='coerce')
```

## Step 5: Binary Y/N → Int64
```python
# For POI binary columns
def yn_to_int(series):
    s = series.astype('string').str.strip().str.upper()
    out = pd.Series(pd.array([pd.NA] * len(s), dtype='Int64'), index=series.index)
    out[s == 'Y'] = 1
    out[s == 'N'] = 0
    return out
```

## Step 6: Drop admin/weight columns
These are survey design columns not needed for modelling:
```
response_status, non_response_reason, primary_sampling_unit_id,
household_weight_monthly_survey, raw_household_weight_monthly_survey,
household_weight_country_monthly_survey, raw_household_weight_country_monthly_survey,
household_weight_state_monthly_survey, raw_household_weight_state_monthly_survey,
household_non_response_weight_*, member_weight_*, raw_member_weight_*,
member_age15plus_weight_*, member_non_response_weight_*,
wave_number, member_status, age_months, state_of_origin
```

---

# PART 3 — FEATURE ENGINEERING: HOUSEHOLD-LEVEL AGGREGATION

## 3.1 POI: Person → Household Aggregation

Since POI is person-level, aggregate to household level using `JOIN_KEYS_COMMON`:
`['household_id', 'state', 'homogeneous_region', 'district', 'region_type', 'stratum', 'month_slot']`

**Aggregation rules:**

| Input Column | Output Column | Aggregation |
|---|---|---|
| `is_healthy` | `is_healthy_hh_min` | min (all healthy = 1, any unhealthy = 0) |
| `is_hospitalised` | `is_hospitalised_hh_any` | max (any hospitalised = 1) |
| `is_on_regular_medication` | `is_on_regular_medication_hh_any` | max |
| `has_bank_account` | `has_bank_account_hh_any` | max |
| `has_health_insurance` | `has_health_insurance_hh_any` | max |
| `has_life_insurance` | `has_life_insurance_hh_any` | max |
| `has_provident_fund_account` | `has_provident_fund_account_hh_any` | max |
| `has_credit_card` | `has_credit_card_hh_any` | max |
| `has_demat_account` | `has_demat_account_hh_any` | max |
| `has_mobile_phone` | `has_mobile_phone_hh_any` | max |
| `age_years` | `age_years_hh_mean` | mean |
| `education_level` | `education_rank_hh_max` | max of ordinal rank (see below) |
| `gender` | `gender_hh_mode` | mode |
| `occupation_type` | `occupation_type_hh_mode` | mode |
| `education_level` | `education_level_hh_mode` | mode |

**Education ordinal ranking:**
```python
EDU_ORDER = ['Illiterate', 'Literate but no formal schooling', 'Below Primary',
             'Primary', 'Middle', 'Secondary', 'Higher Secondary',
             'Diploma', 'Graduate', 'Post Graduate', 'Doctorate']
```

## 3.2 Dataset Join Strategy

```
INC (household × reference_month)
  JOIN CON (household × reference_month)
    ON: household_id + state + homogeneous_region + district +
        region_type + stratum + month_slot + reference_month
    TYPE: INNER JOIN

  LEFT JOIN POI_aggregated (household)
    ON: household_id + state + homogeneous_region + district +
        region_type + stratum + month_slot
    TYPE: LEFT JOIN (not all households have POI data)
```

---

# PART 4 — STRESS LABELS (Target Variables)

These four binary labels are the MODEL TARGETS. Compute them AFTER the join.

## Label 1: financial_stress (binary 0/1)
```python
eps = 1e-9
ratio_gap = (total_expenditure_adjusted - total_income) / (total_income + eps)
financial_stress = (ratio_gap > 0.1).astype(int)
# i.e., spending more than 110% of income → stressed
```

## Label 2: food_stress (binary 0/1)
```python
food_stress = (exp_food_adjusted / (total_expenditure_adjusted + eps) > 0.5).astype(int)
# i.e., >50% of spending goes to food → stressed
```

## Label 3: debt_stress (binary 0/1)
```python
debt_stress = (exp_all_emis / (total_income + eps) > 0.3).astype(int)
# i.e., EMIs are >30% of income → stressed
```

## Label 4: health_stress (binary 0/1)
```python
health_stress = ((is_hospitalised_hh_any > 0) | (is_on_regular_medication_hh_any > 0)).astype(int)
# i.e., any member hospitalised or on regular medication → stressed
```

## Composite Score (0–4)
```python
composite_stress_score = financial_stress + food_stress + debt_stress + health_stress
# 0 = no stress, 4 = maximum stress
```

---

# PART 5 — DERIVED RATIO FEATURES (Input to Models)

Compute these BEFORE dropping income/expenditure columns:

```python
eps = 1e-9
emi_to_income_ratio          = exp_all_emis / (total_income + eps)
food_to_expense_ratio        = exp_food_adjusted / (total_expenditure_adjusted + eps)
health_to_expense_ratio      = exp_health / (total_expenditure_adjusted + eps)
education_to_expense_ratio   = exp_education / (total_expenditure_adjusted + eps)
discretionary_to_expense_ratio = (exp_recreation + exp_vacation + exp_restaurants_adjusted) / (total_expenditure_adjusted + eps)
savings_proxy                = total_income - total_expenditure_adjusted

# Income diversity: count of non-zero income sources
income_sources = ['income_all_members_from_wages', 'income_household_from_rent',
                  'income_household_from_self_production',
                  'income_household_from_private_transfers',
                  'income_household_from_business_profit']
income_diversity_score = (df[income_sources].fillna(0) > 0).sum(axis=1)
```

**After computing labels and ratios, DROP these leakage columns from model input:**
`total_income`, `total_expenditure_adjusted`

---

# PART 6 — OUTPUT FILE STRUCTURE

```
D:\Project\
├── Dataset\
│   ├── Income_Pyramid\          ← raw input
│   ├── People_of_India\         ← raw input
│   └── Consumption_Pyramid\     ← raw input
│
├── processed\                   ← Step 1 output: cleaned per-dataset parquets
│   ├── income_pyramid\
│   ├── people_of_india\
│   └── consumption_pyramid\
│
├── processed2\                  ← Step 2 output: column-selected parquets
│   ├── income_pyramid\          ← *_inc.parquet files
│   ├── people_of_india\         ← *_poi.parquet files
│   ├── consumption_pyramid\     ← *_con.parquet files
│   └── household_joined\
│       ├── poi_household_agg.parquet      ← POI aggregated cache
│       └── household_stress_dataset.parquet  ← FINAL joined+labeled dataset
│
├── models\                      ← Trained model files (.pkl / .joblib)
│   ├── financial_stress_model.pkl
│   ├── food_stress_model.pkl
│   ├── debt_stress_model.pkl
│   ├── health_stress_model.pkl
│   ├── composite_stress_model.pkl
│   └── feature_metadata.json    ← feature names, encoders, scaler info
│
├── web\                         ← Web application
│   ├── backend\                 ← FastAPI or Flask backend
│   ├── frontend\                ← React or HTML/CSS/JS frontend
│   └── database\                ← PostgreSQL schema + migrations
│
└── scripts\
    ├── 01_clean_data.py
    ├── 02_build_household_dataset.py
    ├── 03_train_models.py
    ├── 04_evaluate_models.py
    └── 05_run_chatbot.py
```

---

# PART 7 — MODEL TRAINING SPECIFICATION

## 7.1 Models to Train

Train ONE model per stress label (4 binary classifiers + 1 composite regressor/classifier):

| Model Name | Target | Type | Recommended Algorithm |
|---|---|---|---|
| `financial_stress_model` | `financial_stress` | Binary classifier | XGBoost or LightGBM |
| `food_stress_model` | `food_stress` | Binary classifier | XGBoost or RandomForest |
| `debt_stress_model` | `debt_stress` | Binary classifier | XGBoost or LightGBM |
| `health_stress_model` | `health_stress` | Binary classifier | XGBoost or LightGBM |
| `composite_stress_model` | `composite_stress_score` (0-4) | Multiclass or Regression | XGBoost Regressor → round |

## 7.2 Input Features for Models

**Numeric features (scale with StandardScaler or use tree models that don't need scaling):**
```
emi_to_income_ratio, food_to_expense_ratio, health_to_expense_ratio,
education_to_expense_ratio, discretionary_to_expense_ratio, savings_proxy,
income_diversity_score, age_years_hh_mean, education_rank_hh_max,
exp_all_emis, exp_food_adjusted, exp_health, exp_education,
exp_recreation, exp_vacation, exp_restaurants_adjusted,
income_all_members_from_wages, income_household_from_rent,
income_household_from_self_production, income_household_from_private_transfers,
income_household_from_business_profit,
is_healthy_hh_min, is_hospitalised_hh_any, is_on_regular_medication_hh_any,
has_bank_account_hh_any, has_health_insurance_hh_any, has_life_insurance_hh_any,
has_provident_fund_account_hh_any, has_credit_card_hh_any,
has_demat_account_hh_any, has_mobile_phone_hh_any
```

**Categorical features (one-hot encode or use LabelEncoder):**
```
state, region_type, age_group, occupation_group, education_group,
gender_group, household_size_group, gender_hh_mode, occupation_type_hh_mode
```

## 7.3 Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## 7.4 Class Imbalance Handling
- Use `scale_pos_weight` in XGBoost or `class_weight='balanced'` in sklearn models
- Alternatively use SMOTE for oversampling

## 7.5 Evaluation Metrics
- **Primary:** F1-Score (macro), AUC-ROC
- **Secondary:** Precision, Recall, Confusion Matrix
- Save evaluation report to `models/evaluation_report.json`

## 7.6 Model Persistence
```python
import joblib
# Save
joblib.dump(model, 'models/financial_stress_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')  # scaler + encoders

# Save feature metadata
import json
metadata = {
    'numeric_features': [...],
    'categorical_features': [...],
    'feature_order': [...],  # exact order for inference
    'label_encoders': {...},  # category → int mappings
    'model_version': '1.0',
    'training_date': '...',
}
with open('models/feature_metadata.json', 'w') as f:
    json.dump(metadata, f)
```

---

# PART 8 — CHATBOT SPECIFICATION

## 8.1 Architecture
- **LLM:** OpenAI GPT-4o-mini (or GPT-4o for better quality)
- **Stress Prediction:** Local trained ML models (loaded via joblib)
- **Memory:** Conversation history stored in PostgreSQL
- **Framework:** LangChain (optional) or direct OpenAI API calls

## 8.2 Conversation Flow

The chatbot collects information in a structured multi-step conversation:

### Step 1: Greeting & Profile
```
Bot: "Namaste! I'm your financial wellness advisor. I'll help assess your household's 
financial health and give you personalized advice. Let's start with some basics.

What state do you live in? (e.g., Maharashtra, Gujarat, Tamil Nadu...)"
```

### Step 2: Household Information
```
Collect:
- Number of family members
- Region type (urban/rural)
- Primary earner's occupation
- Education level of head of household
- Age of head of household
```

### Step 3: Income Information
```
Collect:
- Total monthly household income
- Income from salary/wages
- Income from rent (if any)
- Income from business (if any)
- Other income sources
```

### Step 4: Expense Information
```
Collect:
- Total monthly spending (approximate)
- Monthly food expenses
- Monthly EMI/loan payments
- Monthly health expenses
- Monthly education expenses
```

### Step 5: Health Information
```
Collect:
- Is any family member currently hospitalised?
- Is any member on regular long-term medication?
- Does the family have health insurance?
```

### Step 6: Assets & Savings
```
Collect:
- Does anyone in family have a bank account?
- Does anyone have life insurance?
- Does anyone have any investments (provident fund, demat account)?
```

## 8.3 Prediction & Advice Generation

After collecting all information:

```python
# 1. Map user inputs to model features
user_features = map_chat_inputs_to_features(user_responses)

# 2. Run ML models
predictions = {
    'financial_stress': financial_model.predict_proba(user_features)[0][1],
    'food_stress': food_model.predict_proba(user_features)[0][1],
    'debt_stress': debt_model.predict_proba(user_features)[0][1],
    'health_stress': health_model.predict_proba(user_features)[0][1],
}

# 3. Generate advice via OpenAI
system_prompt = """You are a compassionate Indian household financial advisor. 
You have access to ML model predictions about a family's stress levels.
Give specific, actionable advice in simple language (mix Hindi terms naturally).
Always be encouraging, never alarming. Suggest government schemes when relevant.
Format advice in clear sections."""

user_message = f"""
Family profile: {user_profile_summary}
Stress predictions: {predictions}
Financial ratios: {computed_ratios}

Give personalized financial health advice covering:
1. Overall financial health status
2. Key areas of concern (if any)
3. Specific actionable recommendations
4. Relevant government schemes they may qualify for
5. Emergency fund and savings advice
"""
```

## 8.4 System Prompt for OpenAI

```python
ADVISOR_SYSTEM_PROMPT = """
You are 'ArthSaathi' (Financial Friend), a compassionate AI financial wellness 
advisor for Indian households. You have deep knowledge of:

- Indian household economics and budgeting
- Government schemes: PM Jan Dhan, PMJJBY, PMSBY, PM Kisan, Ayushman Bharat, 
  NPS, EPF, PPF, Sukanya Samriddhi, Atal Pension Yojana
- CMIE-based household stress patterns across Indian states
- RBI guidelines on household debt
- Financial inclusion best practices

Your personality:
- Warm, respectful, uses "aap" form of address
- Speaks simply, avoids jargon
- Gives SPECIFIC numbers and percentages in advice
- Always ends with an encouraging, hopeful note
- Never judges the household's financial situation

When you receive ML model stress scores (0 to 1 probability):
- 0.0-0.3: Low risk — affirm positive behaviors
- 0.3-0.6: Moderate risk — gentle alerts + specific fixes
- 0.6-1.0: High risk — clear action items + urgent recommendations

Structure your advice as:
🏠 Financial Health Overview
📊 Your Numbers (key ratios)
⚠️ Areas Needing Attention (if any)  
✅ Action Plan (3-5 specific steps)
🇮🇳 Government Schemes You May Qualify For
💪 Encouragement
"""
```

---

# PART 9 — DATABASE SCHEMA (PostgreSQL)

```sql
-- Users table
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100),
    email       VARCHAR(150) UNIQUE,
    phone       VARCHAR(20),
    state       VARCHAR(50),
    region_type VARCHAR(20),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Conversations table
CREATE TABLE conversations (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id),
    session_id      VARCHAR(100) UNIQUE,
    started_at      TIMESTAMP DEFAULT NOW(),
    completed       BOOLEAN DEFAULT FALSE,
    profile_json    JSONB,    -- collected user profile
    prediction_json JSONB,    -- ML model predictions
    advice_text     TEXT      -- generated advice
);

-- Messages table (full conversation history)
CREATE TABLE messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id),
    role            VARCHAR(10),  -- 'user' or 'assistant'
    content         TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Assessments table (summary of stress scores)
CREATE TABLE assessments (
    id                    SERIAL PRIMARY KEY,
    conversation_id       INTEGER REFERENCES conversations(id),
    financial_stress_prob FLOAT,
    food_stress_prob      FLOAT,
    debt_stress_prob      FLOAT,
    health_stress_prob    FLOAT,
    composite_score       FLOAT,
    assessed_at           TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_conversations_user    ON conversations(user_id);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_assessments_conv      ON assessments(conversation_id);
```

---

# PART 10 — WEB SYSTEM SPECIFICATION

## 10.1 Backend (FastAPI recommended)

```
web/backend/
├── main.py                  ← FastAPI app entry
├── routers/
│   ├── chat.py              ← POST /chat/message, GET /chat/history
│   ├── assessment.py        ← POST /assess, GET /assessments/{user_id}
│   └── users.py             ← POST /users, GET /users/{id}
├── models/
│   ├── predict.py           ← load joblib models, run inference
│   └── schemas.py           ← Pydantic request/response schemas
├── services/
│   ├── openai_service.py    ← OpenAI API calls
│   ├── chat_service.py      ← conversation logic
│   └── db_service.py        ← database CRUD
├── database.py              ← SQLAlchemy setup
└── config.py                ← env vars (OPENAI_API_KEY, DATABASE_URL, etc.)
```

**Key API endpoints:**
```
POST /api/chat/start          → create session, return session_id
POST /api/chat/message        → send message, get bot reply
GET  /api/chat/{session_id}   → get full conversation history
POST /api/assess              → run ML models on profile, store result
GET  /api/assessments/{user_id} → get user's assessment history
```

## 10.2 Frontend (React recommended)

```
web/frontend/
├── src/
│   ├── components/
│   │   ├── ChatWindow.jsx       ← main chat interface
│   │   ├── MessageBubble.jsx    ← individual message
│   │   ├── StressGauge.jsx      ← visual stress score display
│   │   ├── AdviceCard.jsx       ← formatted advice display
│   │   └── ProgressBar.jsx      ← conversation step tracker
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Chat.jsx
│   │   └── History.jsx          ← past assessments
│   ├── services/
│   │   └── api.js               ← axios calls to backend
│   └── App.jsx
```

## 10.3 Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://postgres:password@localhost:5432/arthsaathi
MODEL_DIR=D:/Project/models
SECRET_KEY=your-secret-key-here
```

---

# PART 11 — EXECUTION ORDER

Run scripts in this exact order:

```bash
# Step 1: Clean raw data
python scripts/01_clean_data.py --root D:/Project

# Step 2: Build household dataset (join + labels + features)
python scripts/02_build_household_dataset.py --root D:/Project

# Step 3: Train all models
python scripts/03_train_models.py --root D:/Project --data D:/Project/processed2/household_joined/household_stress_dataset.parquet

# Step 4: Evaluate models
python scripts/04_evaluate_models.py --root D:/Project

# Step 5: Start web server (after setting .env)
cd web/backend && uvicorn main:app --reload

# Step 6: Start frontend
cd web/frontend && npm start
```

---

# PART 12 — IMPORTANT IMPLEMENTATION NOTES

1. **Memory:** The full dataset is large (millions of rows). Read parquet files in chunks or use Dask if RAM is limited. For model training, a sample of 500K–1M rows is sufficient.

2. **Missing data strategy:** Use median imputation for numeric features, mode imputation for categoricals. Fit imputer on training set only, apply to test.

3. **Feature validation at inference:** The chatbot collects partial information. Map missing fields to NaN and let the model handle them — tree-based models handle NaN natively.

4. **Model input mapping:** Create a function `map_chat_inputs_to_features(user_dict)` that maps chatbot-collected fields (e.g., `"monthly_income": 25000`) to the exact feature vector expected by models (e.g., `total_income`, ratio features, etc.).

5. **Security:** Never expose ML model internals or raw data through the API. Only return predictions and advice.

6. **Conversation state:** Store the current conversation step and collected data in the `conversations.profile_json` column. On each message, update this JSON.

7. **Fallback advice:** If ML model prediction fails (missing data), use rule-based thresholds for advice and still call OpenAI with available information.

8. **Indian context:** Always consider Indian financial context — income in INR, EMIs, Indian government schemes, regional economic differences (e.g., rural Maharashtra vs urban Tamil Nadu have different benchmarks).

---

# PART 13 — QUICK REFERENCE: KEY COLUMN NAMES

```python
JOIN_KEYS_COMMON = ['household_id', 'state', 'homogeneous_region', 'district', 
                    'region_type', 'stratum', 'month_slot']

JOIN_KEYS_WITH_MONTH = JOIN_KEYS_COMMON + ['reference_month']

STRESS_LABELS = ['financial_stress', 'food_stress', 'debt_stress', 'health_stress']

RATIO_FEATURES = ['emi_to_income_ratio', 'food_to_expense_ratio', 
                  'health_to_expense_ratio', 'education_to_expense_ratio',
                  'discretionary_to_expense_ratio', 'savings_proxy',
                  'income_diversity_score']

POI_AGG_FEATURES = ['is_healthy_hh_min', 'is_hospitalised_hh_any',
                    'is_on_regular_medication_hh_any', 'has_bank_account_hh_any',
                    'has_health_insurance_hh_any', 'has_life_insurance_hh_any',
                    'has_provident_fund_account_hh_any', 'has_credit_card_hh_any',
                    'has_demat_account_hh_any', 'has_mobile_phone_hh_any',
                    'age_years_hh_mean', 'education_rank_hh_max',
                    'gender_hh_mode', 'occupation_type_hh_mode', 'education_level_hh_mode']
```

---

*This prompt was generated from analysis of: preprocess_inc.py, preprocess_poi.py, preprocess_con.py, preprocess_utils.py, build_household_dataset.py, preprocess_data_v3.py, 03_preprocessing_task2.py, 00_preprocess_all.py, 06_parquet_to_postgres.ipynb*

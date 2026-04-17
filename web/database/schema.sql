-- =============================================================
-- ArthSaathi Database Schema (PostgreSQL)
-- =============================================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100),
    email       VARCHAR(150) UNIQUE,
    phone       VARCHAR(20),
    state       VARCHAR(50),
    region_type VARCHAR(20),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    session_id      VARCHAR(100) UNIQUE NOT NULL,
    started_at      TIMESTAMP DEFAULT NOW(),
    completed       BOOLEAN DEFAULT FALSE,
    profile_json    JSONB,
    prediction_json JSONB,
    advice_text     TEXT
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(10) NOT NULL,   -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Assessments table
CREATE TABLE IF NOT EXISTS assessments (
    id                    SERIAL PRIMARY KEY,
    conversation_id       INTEGER REFERENCES conversations(id) ON DELETE CASCADE UNIQUE,
    financial_stress_prob FLOAT,
    food_stress_prob      FLOAT,
    debt_stress_prob      FLOAT,
    health_stress_prob    FLOAT,
    composite_score       FLOAT,
    -- v4 fields
    is_stressed           BOOLEAN,
    stress_level          SMALLINT,          -- 0=none 1=mild 2=moderate 3=severe
    stressed_domains      TEXT[],            -- e.g. {financial_stress, food_stress}
    input_warnings        TEXT[],            -- out-of-range chatbot inputs
    -- v5 fields
    model_version         VARCHAR(10) DEFAULT 'v5',
    prediction_confidence FLOAT,             -- bootstrap std; >0.15 = uncertain
    shap_top_reasons      JSONB,             -- {"financial_stress": ["reason1", ...]}
    ab_group              VARCHAR(5),        -- 'v4' | 'v5' for A/B tracking
    assessed_at           TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_conversations_user    ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_assessments_conv      ON assessments(conversation_id);
CREATE INDEX IF NOT EXISTS idx_assessments_stressed  ON assessments(is_stressed);

-- v4 migration: add columns if upgrading from v3 schema
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS is_stressed      BOOLEAN;
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS stress_level     SMALLINT;
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS stressed_domains TEXT[];
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS input_warnings   TEXT[];

-- v5 migration: add columns if upgrading from v4 schema
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS model_version         VARCHAR(10) DEFAULT 'v5';
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT;
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS shap_top_reasons      JSONB;
ALTER TABLE assessments ADD COLUMN IF NOT EXISTS ab_group              VARCHAR(5);


"""
database.py — SQLAlchemy async engine + session factory for ArthSaathi.
"""
from sqlalchemy import (
    create_engine, Column, Integer, SmallInteger, String,
    Boolean, Float, Text, DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── ORM Models ───────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(150), unique=True, nullable=True)
    phone = Column(String(20), nullable=True)
    state = Column(String(50), nullable=True)
    region_type = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    conversations = relationship('Conversation', back_populates='user')


class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    session_id = Column(String(100), unique=True, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed = Column(Boolean, default=False)
    profile_json = Column(JSONB, nullable=True)
    prediction_json = Column(JSONB, nullable=True)
    advice_text = Column(Text, nullable=True)
    user = relationship('User', back_populates='conversations')
    messages = relationship('Message', back_populates='conversation', order_by='Message.id')
    assessment = relationship('Assessment', back_populates='conversation', uselist=False)


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(10))         # 'user' | 'assistant'
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    conversation = relationship('Conversation', back_populates='messages')


class Assessment(Base):
    __tablename__ = 'assessments'
    id                    = Column(Integer, primary_key=True, index=True)
    conversation_id       = Column(Integer, ForeignKey('conversations.id'), unique=True)
    financial_stress_prob = Column(Float, nullable=True)
    food_stress_prob      = Column(Float, nullable=True)
    debt_stress_prob      = Column(Float, nullable=True)
    health_stress_prob    = Column(Float, nullable=True)
    composite_score       = Column(Float, nullable=True)
    # v4 fields
    is_stressed           = Column(Boolean,      nullable=True)
    stress_level          = Column(SmallInteger, nullable=True)
    stressed_domains      = Column(ARRAY(String), nullable=True)
    input_warnings        = Column(ARRAY(String), nullable=True)
    # v5 fields
    model_version         = Column(String(10), nullable=True, default='v5')
    prediction_confidence = Column(Float, nullable=True)     # bootstrap std
    shap_top_reasons      = Column(JSONB,  nullable=True)    # {label: [reasons]}
    ab_group              = Column(String(5), nullable=True) # 'v4'|'v5'
    assessed_at           = Column(DateTime(timezone=True), server_default=func.now())
    conversation          = relationship('Conversation', back_populates='assessment')


# ─── DB dependency ────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)

"""
schemas.py — Pydantic request/response schemas for ArthSaathi API.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr


# ─── Users ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    state: Optional[str] = None
    region_type: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    state: Optional[str] = None
    region_type: Optional[str] = None

    model_config = {"from_attributes": True}


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatStartRequest(BaseModel):
    user_id: Optional[int] = None
    user_name: Optional[str] = "Guest"


class ChatStartResponse(BaseModel):
    session_id: str
    message: str                  # first bot greeting


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    session_id: str
    reply: str
    step: int                     # current conversation step (1-15)
    is_complete: bool             # True when advice has been given
    predictions: Optional[Dict[str, Any]] = None   # v4: mixed types (float, bool, int, list)
    advice: Optional[str] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str

    model_config = {"from_attributes": True}


class ConversationHistory(BaseModel):
    session_id: str
    messages: List[MessageOut]
    profile: Optional[Dict[str, Any]] = None
    predictions: Optional[Dict[str, Any]] = None   # v4: mixed types


# ─── Assessment ───────────────────────────────────────────────────────────────

class AssessmentRequest(BaseModel):
    session_id: str
    profile: Dict[str, Any]


class AssessmentResponse(BaseModel):
    session_id: str
    financial_stress_prob: Optional[float]
    food_stress_prob: Optional[float]
    debt_stress_prob: Optional[float]
    health_stress_prob: Optional[float]
    composite_score: Optional[float]
    advice: Optional[str] = None

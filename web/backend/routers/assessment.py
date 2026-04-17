"""
assessment.py — Assessment router.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.schemas import AssessmentRequest, AssessmentResponse
import services.db_service as db_svc
from models.predict import predict
from services.openai_service import generate_advice
from config import MODEL_DIR

router = APIRouter(prefix="/api/assessments", tags=["assessments"])


@router.post("", response_model=AssessmentResponse)
def run_assessment(payload: AssessmentRequest, db: Session = Depends(get_db)):
    """Run ML models on a given profile and store the assessment."""
    conv = db_svc.get_conversation(db, payload.session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    predictions = predict(payload.profile, MODEL_DIR)
    advice = generate_advice(payload.profile, predictions)

    db_svc.create_or_update_assessment(db, conv.id, predictions)
    db_svc.update_conversation_predictions(db, payload.session_id, predictions, advice)

    return AssessmentResponse(
        session_id=payload.session_id,
        financial_stress_prob=predictions.get('financial_stress'),
        food_stress_prob=predictions.get('food_stress'),
        debt_stress_prob=predictions.get('debt_stress'),
        health_stress_prob=predictions.get('health_stress'),
        composite_score=predictions.get('composite_stress_score'),
        advice=advice,
    )


@router.get("/user/{user_id}")
def get_assessments(user_id: int, db: Session = Depends(get_db)):
    """Get all assessments for a user."""
    assessments = db_svc.get_assessments_by_user(db, user_id)
    return [
        {
            "id": a.id,
            "conversation_id": a.conversation_id,
            "financial_stress_prob": a.financial_stress_prob,
            "food_stress_prob": a.food_stress_prob,
            "debt_stress_prob": a.debt_stress_prob,
            "health_stress_prob": a.health_stress_prob,
            "composite_score": a.composite_score,
            "assessed_at": a.assessed_at,
        }
        for a in assessments
    ]

@router.get("/global")
def get_global_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get all past assessments unconditionally (since there is no user login yet)."""
    assessments = db_svc.get_global_assessments(db, limit)
    return [
        {
            "id": a.id,
            "conversation_id": a.conversation.session_id if a.conversation else None,
            "financial_stress_prob": a.financial_stress_prob,
            "food_stress_prob": a.food_stress_prob,
            "debt_stress_prob": a.debt_stress_prob,
            "health_stress_prob": a.health_stress_prob,
            "composite_score": a.composite_score,
            "assessed_at": a.assessed_at,
        }
        for a in assessments
    ]

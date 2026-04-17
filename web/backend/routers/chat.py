"""
chat.py — Chat conversation router.
Manages multi-step info collection and final advice generation.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.schemas import (
    ChatStartRequest, ChatStartResponse,
    ChatMessageRequest, ChatMessageResponse,
    ConversationHistory, MessageOut,
)
import services.db_service as db_svc
from services.chat_service import (
    TOTAL_STEPS, process_user_response,
    get_next_step_prompt, format_profile_summary,
)
from services.openai_service import generate_advice
from models.predict import predict
from config import MODEL_DIR

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/start", response_model=ChatStartResponse)
def start_chat(payload: ChatStartRequest, db: Session = Depends(get_db)):
    """Create a new conversation session and return the first greeting."""
    conv = db_svc.create_conversation(db, user_id=payload.user_id)
    
    # Initialize profile with step counter
    profile = {"_step": 1, "_user_name": payload.user_name or "Guest"}
    db_svc.update_conversation_profile(db, conv.session_id, profile)

    greeting = get_next_step_prompt(1, profile)
    db_svc.add_message(db, conv.id, "assistant", greeting)

    return ChatStartResponse(session_id=conv.session_id, message=greeting)


@router.post("/message", response_model=ChatMessageResponse)
def send_message(payload: ChatMessageRequest, db: Session = Depends(get_db)):
    """Process a user message and return bot reply."""
    conv = db_svc.get_conversation(db, payload.session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new chat.")

    if conv.completed:
        return ChatMessageResponse(
            session_id=payload.session_id,
            reply="Your assessment is already complete. Start a new chat to get a fresh assessment.",
            step=TOTAL_STEPS,
            is_complete=True,
            predictions=conv.prediction_json,
            advice=conv.advice_text,
        )

    # Store user message
    db_svc.add_message(db, conv.id, "user", payload.message)

    # Load current profile and step
    profile = conv.profile_json or {"_step": 1}
    current_step = profile.get("_step", 1)

    # Parse the user's answer for the current step
    profile = process_user_response(current_step, payload.message, profile)
    next_step = current_step + 1
    profile["_step"] = next_step

    # Save updated profile
    db_svc.update_conversation_profile(db, payload.session_id, profile)

    # If all questions answered → run models + generate advice
    if next_step > TOTAL_STEPS:
        try:
            predictions = predict(profile, MODEL_DIR)
        except Exception as e:
            predictions = {
                'financial_stress': 0.5,
                'food_stress': 0.5,
                'debt_stress': 0.5,
                'health_stress': 0.5,
                'composite_stress_score': 2.0,
            }

        advice = generate_advice(profile, predictions)

        db_svc.update_conversation_predictions(db, payload.session_id, predictions, advice)
        db_svc.create_or_update_assessment(db, conv.id, predictions)

        summary = format_profile_summary(profile)
        reply = (
            f"Thank you! 🙏 Here is your **ArthSaathi Assessment**:\n\n"
            f"**Your Profile:**\n{summary}\n\n"
            f"---\n\n{advice}"
        )
        db_svc.add_message(db, conv.id, "assistant", reply)

        return ChatMessageResponse(
            session_id=payload.session_id,
            reply=reply,
            step=TOTAL_STEPS,
            is_complete=True,
            predictions=predictions,
            advice=advice,
        )

    # Next question
    bot_reply = get_next_step_prompt(next_step, profile)
    db_svc.add_message(db, conv.id, "assistant", bot_reply)

    return ChatMessageResponse(
        session_id=payload.session_id,
        reply=bot_reply,
        step=next_step,
        is_complete=False,
    )


@router.get("/{session_id}", response_model=ConversationHistory)
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve full conversation history."""
    conv = db_svc.get_conversation(db, session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db_svc.get_messages(db, conv.id)
    return ConversationHistory(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in messages],
        profile=conv.profile_json,
        predictions=conv.prediction_json,
    )

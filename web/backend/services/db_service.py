"""
db_service.py — Database CRUD operations for ArthSaathi.
"""
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session, joinedload

from database import User, Conversation, Message, Assessment


# ─── Users ────────────────────────────────────────────────────────────────────

def create_user(db: Session, name: str, email: Optional[str] = None,
                phone: Optional[str] = None, state: Optional[str] = None,
                region_type: Optional[str] = None) -> User:
    user = User(name=name, email=email, phone=phone,
                state=state, region_type=region_type)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


# ─── Conversations ────────────────────────────────────────────────────────────

def create_conversation(db: Session, user_id: Optional[int] = None) -> Conversation:
    session_id = str(uuid.uuid4())
    conv = Conversation(user_id=user_id, session_id=session_id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(db: Session, session_id: str) -> Optional[Conversation]:
    return db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()


from sqlalchemy.orm.attributes import flag_modified

def update_conversation_profile(db: Session, session_id: str,
                                 profile: Dict[str, Any]) -> Optional[Conversation]:
    conv = get_conversation(db, session_id)
    if conv:
        conv.profile_json = profile
        flag_modified(conv, "profile_json")
        db.commit()
        db.refresh(conv)
    return conv


def update_conversation_predictions(db: Session, session_id: str,
                                     predictions: Dict[str, float],
                                     advice: str) -> Optional[Conversation]:
    conv = get_conversation(db, session_id)
    if conv:
        conv.prediction_json = predictions
        conv.advice_text = advice
        conv.completed = True
        flag_modified(conv, "prediction_json")
        db.commit()
        db.refresh(conv)
    return conv


def get_user_conversations(db: Session, user_id: int) -> List[Conversation]:
    return (db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.started_at.desc())
            .all())


# ─── Messages ─────────────────────────────────────────────────────────────────

def add_message(db: Session, conversation_id: int, role: str,
                content: str) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages(db: Session, conversation_id: int) -> List[Message]:
    return (db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .all())


# ─── Assessments ──────────────────────────────────────────────────────────────

def create_or_update_assessment(db: Session, conversation_id: int,
                                 predictions: Dict[str, Any]) -> Assessment:
    # v4 fields
    is_stressed      = predictions.get('is_stressed')
    stress_level     = predictions.get('stress_level')
    stressed_domains = predictions.get('stressed_domains') or []
    input_warnings   = predictions.get('input_warnings') or []
    # v5 fields
    model_version    = predictions.get('model_version', 'v5')
    shap_top_reasons = predictions.get('shap_reasons') or {}
    ab_group         = predictions.get('ab_group')
    confidence_info  = predictions.get('confidence') or {}
    stds = [v.get('std') for v in confidence_info.values()
            if isinstance(v, dict) and v.get('std') is not None]
    pred_confidence  = float(sum(stds) / len(stds)) if stds else None

    def _fill(obj):
        obj.financial_stress_prob = predictions.get('financial_stress')
        obj.food_stress_prob      = predictions.get('food_stress')
        obj.debt_stress_prob      = predictions.get('debt_stress')
        obj.health_stress_prob    = predictions.get('health_stress')
        obj.composite_score       = predictions.get('composite_stress_score')
        obj.is_stressed           = bool(is_stressed) if is_stressed is not None else None
        obj.stress_level          = int(stress_level) if stress_level is not None else None
        obj.stressed_domains      = stressed_domains
        obj.input_warnings        = input_warnings
        obj.model_version         = model_version
        obj.prediction_confidence = pred_confidence
        obj.shap_top_reasons      = shap_top_reasons if shap_top_reasons else None
        obj.ab_group              = ab_group

    existing = (db.query(Assessment)
                .filter(Assessment.conversation_id == conversation_id)
                .first())
    if existing:
        _fill(existing)
        db.commit()
        db.refresh(existing)
        return existing

    assessment = Assessment(conversation_id=conversation_id)
    _fill(assessment)
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment



def get_assessments_by_user(db: Session, user_id: int) -> List[Assessment]:
    convs = get_user_conversations(db, user_id)
    conv_ids = [c.id for c in convs]
    return (db.query(Assessment)
            .filter(Assessment.conversation_id.in_(conv_ids))
            .order_by(Assessment.assessed_at.desc())
            .all())

def get_global_assessments(db: Session, limit: int = 50) -> List[Assessment]:
    """Get all assessments regardless of user (used for local anon sessions)."""
    return (db.query(Assessment)
            .options(joinedload(Assessment.conversation))
            .filter(Assessment.prediction_confidence.isnot(None)) # must have predictions
            .order_by(Assessment.assessed_at.desc())
            .limit(limit)
            .all())

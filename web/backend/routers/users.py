"""
users.py — User management router.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.schemas import UserCreate, UserResponse
import services.db_service as db_svc

router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.email:
        existing = db_svc.get_user_by_email(db, payload.email)
        if existing:
            return existing
    return db_svc.create_user(
        db,
        name=payload.name,
        email=payload.email,
        phone=payload.phone,
        state=payload.state,
        region_type=payload.region_type,
    )


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db_svc.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

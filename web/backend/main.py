"""
main.py — FastAPI application entry point for ArthSaathi.
"""
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from database import create_tables
from routers import chat, assessment, users

# ── Prometheus metrics (pip install prometheus-client) ────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    PREDICTION_COUNTER = Counter(
        "arthsaathi_predictions_total",
        "Total ML predictions made",
        ["stress_type"],
    )
    PREDICTION_LATENCY = Histogram(
        "arthsaathi_prediction_latency_seconds",
        "End-to-end /api/chat/message latency",
    )
    HIGH_STRESS_COUNTER = Counter(
        "arthsaathi_high_stress_total",
        "Number of high-stress flags raised",
        ["domain"],
    )
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
    print("  [INFO] prometheus_client not installed — /metrics endpoint disabled")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ArthSaathi API",
    description="Financial Stress Analysis & Advisory API for Indian Households",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(chat.router)
app.include_router(assessment.router)


@app.on_event("startup")
def startup_event():
    """Initialize DB tables on startup."""
    try:
        create_tables()
        print("✓ Database tables created/verified")
    except Exception as e:
        print(f"⚠ DB init warning: {e}")


@app.get("/")
def root():
    return {
        "app": "ArthSaathi",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics" if PROMETHEUS_ENABLED else "disabled (install prometheus-client)",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint. Install: pip install prometheus-client"""
    if not PROMETHEUS_ENABLED:
        return Response("# prometheus_client not installed\n", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


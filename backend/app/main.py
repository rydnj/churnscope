"""ChurnScope FastAPI application.

Serves all analytics data to the React dashboard.

Usage:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.routers import kpis, segments, churn, forecast

app = FastAPI(
    title="ChurnScope API",
    description="Customer Churn Prediction & Revenue Analytics",
    version="1.0.0",
)

# CORS — allow the React dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(kpis.router, prefix=settings.api_prefix)
app.include_router(segments.router, prefix=settings.api_prefix)
app.include_router(churn.router, prefix=settings.api_prefix)
app.include_router(forecast.router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {"status": "ok", "app": "ChurnScope API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}
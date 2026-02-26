"""Churn endpoints: risk list, model metrics, feature importance."""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.database import get_engine
from backend.app.schemas import (
    ChurnRiskCustomer, PaginatedChurnRisk, ModelMetrics, FeatureImportance,
)

router = APIRouter(prefix="/churn", tags=["Churn"])

REPORTS_DIR = Path("reports/churn")


@router.get("/risk", response_model=PaginatedChurnRisk)
def get_churn_risk(
    risk_tier: str | None = Query(None, regex="^(high|medium|low)$"),
    segment: str | None = None,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    engine: Engine = Depends(get_engine),
):
    offset = (page - 1) * limit

    where_clauses = ["a.churn_probability IS NOT NULL"]
    params = {"limit": limit, "offset": offset}

    if risk_tier:
        if risk_tier == "high":
            where_clauses.append("a.churn_probability >= 0.7")
        elif risk_tier == "medium":
            where_clauses.append("a.churn_probability >= 0.4 AND a.churn_probability < 0.7")
        else:
            where_clauses.append("a.churn_probability < 0.4")

    if segment:
        where_clauses.append("a.segment_name = :segment")
        params["segment"] = segment

    where_sql = " AND ".join(where_clauses)

    with engine.connect() as conn:
        count = conn.execute(text(
            f"SELECT COUNT(*) FROM agg_customer_features a WHERE {where_sql}"
        ), params).scalar()

        rows = conn.execute(text(f"""
            SELECT
                a.customer_id, a.churn_probability,
                CASE
                    WHEN a.churn_probability >= 0.7 THEN 'high'
                    WHEN a.churn_probability >= 0.4 THEN 'medium'
                    ELSE 'low'
                END AS risk_tier,
                a.segment_name, a.recency_days, a.frequency, a.monetary
            FROM agg_customer_features a
            WHERE {where_sql}
            ORDER BY a.churn_probability DESC
            LIMIT :limit OFFSET :offset
        """), params).fetchall()

    items = [
        ChurnRiskCustomer(
            customer_id=r[0], churn_probability=round(float(r[1]), 4),
            risk_tier=r[2], segment_name=r[3],
            recency_days=r[4], frequency=r[5], monetary=round(float(r[6]), 2),
        )
        for r in rows
    ]

    return PaginatedChurnRisk(items=items, total=count, page=page, page_size=limit)


@router.get("/model-metrics", response_model=list[ModelMetrics])
def get_model_metrics():
    csv_path = REPORTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return []

    import pandas as pd
    df = pd.read_csv(csv_path)
    return [
        ModelMetrics(
            model_name=row["model"], precision=row["precision"],
            recall=row["recall"], f1=row["f1"], auc_roc=row["auc_roc"],
        )
        for _, row in df.iterrows()
    ]


@router.get("/feature-importance", response_model=list[FeatureImportance])
def get_feature_importance():
    csv_path = REPORTS_DIR / "feature_importance.csv"
    if not csv_path.exists():
        return []

    import pandas as pd
    df = pd.read_csv(csv_path)
    return [
        FeatureImportance(feature=row["feature"], importance=round(row["importance"], 4))
        for _, row in df.head(15).iterrows()
    ]
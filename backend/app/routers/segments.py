"""Segment endpoints: customer segment profiles and details."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.database import get_engine
from backend.app.schemas import CustomerSegment

router = APIRouter(prefix="/segments", tags=["Segments"])


@router.get("", response_model=list[CustomerSegment])
def get_segments(engine: Engine = Depends(get_engine)):
    sql = """
    SELECT
        cluster_id,
        segment_name,
        COUNT(*) AS customer_count,
        ROUND(AVG(recency_days)::NUMERIC, 2) AS avg_recency,
        ROUND(AVG(frequency)::NUMERIC, 2) AS avg_frequency,
        ROUND(AVG(monetary)::NUMERIC, 2) AS avg_monetary,
        ROUND(
            COUNT(CASE WHEN churn_label = TRUE THEN 1 END)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100, 2
        ) AS churn_rate
    FROM agg_customer_features
    WHERE segment_name IS NOT NULL
    GROUP BY cluster_id, segment_name
    ORDER BY avg_monetary DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    return [
        CustomerSegment(
            cluster_id=r[0], segment_name=r[1], customer_count=r[2],
            avg_recency=float(r[3]), avg_frequency=float(r[4]),
            avg_monetary=float(r[5]), churn_rate=float(r[6]) if r[6] else None,
        )
        for r in rows
    ]
"""KPI endpoints: executive summary and revenue trends."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.database import get_engine
from backend.app.schemas import KPISummary, RevenueTrendPoint

router = APIRouter(prefix="/kpis", tags=["KPIs"])


@router.get("/summary", response_model=KPISummary)
def get_kpi_summary(engine: Engine = Depends(get_engine)):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END) AS total_revenue,
                COUNT(DISTINCT f.customer_id) AS active_customers,
                COUNT(DISTINCT f.invoice_no) AS total_orders,
                SUM(CASE WHEN f.is_return = FALSE THEN f.total_amount ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT CASE WHEN f.is_return = FALSE THEN f.invoice_no END), 0)
                    AS avg_order_value
            FROM fact_transactions f
        """)).fetchone()

        churn = conn.execute(text("""
            SELECT
                COUNT(CASE WHEN churn_label = TRUE THEN 1 END)::FLOAT
                / NULLIF(COUNT(*), 0) AS churn_rate
            FROM agg_customer_features
            WHERE churn_label IS NOT NULL
        """)).fetchone()

    return KPISummary(
        total_revenue=round(float(row[0]), 2),
        active_customers=row[1],
        total_orders=row[2],
        avg_order_value=round(float(row[3]), 2),
        churn_rate=round(float(churn[0]) * 100, 2) if churn[0] else 0,
    )


@router.get("/revenue-trend", response_model=list[RevenueTrendPoint])
def get_revenue_trend(
    granularity: str = Query("monthly", regex="^(monthly|quarterly)$"),
    engine: Engine = Depends(get_engine),
):
    if granularity == "monthly":
        sql = """
        WITH monthly AS (
            SELECT d.year, d.month,
                SUM(f.total_amount) AS revenue,
                COUNT(DISTINCT f.customer_id) AS active_customers,
                COUNT(DISTINCT f.invoice_no) AS order_count
            FROM fact_transactions f
            JOIN dim_dates d ON f.date_id = d.date_id
            WHERE f.is_return = FALSE
            GROUP BY d.year, d.month
        )
        SELECT year, month, revenue, active_customers, order_count,
            ROUND((revenue - LAG(revenue) OVER (ORDER BY year, month))
                / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0) * 100, 2)
                AS mom_growth_pct
        FROM monthly ORDER BY year, month
        """
    else:
        sql = """
        WITH quarterly AS (
            SELECT d.year, d.quarter AS month,
                SUM(f.total_amount) AS revenue,
                COUNT(DISTINCT f.customer_id) AS active_customers,
                COUNT(DISTINCT f.invoice_no) AS order_count
            FROM fact_transactions f
            JOIN dim_dates d ON f.date_id = d.date_id
            WHERE f.is_return = FALSE
            GROUP BY d.year, d.quarter
        )
        SELECT year, month, revenue, active_customers, order_count,
            ROUND((revenue - LAG(revenue) OVER (ORDER BY year, month))
                / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0) * 100, 2)
                AS mom_growth_pct
        FROM quarterly ORDER BY year, month
        """

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    return [
        RevenueTrendPoint(
            year=r[0], month=r[1], revenue=round(float(r[2]), 2),
            active_customers=r[3], order_count=r[4],
            mom_growth_pct=round(float(r[5]), 2) if r[5] is not None else None,
        )
        for r in rows
    ]
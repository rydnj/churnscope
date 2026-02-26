"""Forecast endpoints: historical + predicted revenue time series."""

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.app.database import get_engine
from backend.app.schemas import ForecastPoint, RevenueTimeSeries

router = APIRouter(prefix="/forecast", tags=["Forecast"])

REPORTS_DIR = Path("reports/forecast")


@router.get("", response_model=RevenueTimeSeries)
def get_forecast(engine: Engine = Depends(get_engine)):
    # Historical monthly revenue
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                TO_CHAR(DATE_TRUNC('month', d.full_date), 'YYYY-MM') AS date,
                SUM(f.total_amount) AS revenue
            FROM fact_transactions f
            JOIN dim_dates d ON f.date_id = d.date_id
            WHERE f.is_return = FALSE
            GROUP BY DATE_TRUNC('month', d.full_date)
            ORDER BY date
        """)).fetchall()

    historical = [
        ForecastPoint(date=r[0], revenue=round(float(r[1]), 2))
        for r in rows
    ]

    # Forecast from CSV
    forecast_csv = REPORTS_DIR / "forecast.csv"
    model_csv = REPORTS_DIR / "model_comparison.csv"

    forecast_points = []
    model_name = "ARIMA(1,1,1)"
    mape = 0.0

    if forecast_csv.exists():
        import pandas as pd
        df = pd.read_csv(forecast_csv)
        forecast_points = [
            ForecastPoint(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m"),
                forecast=round(row["forecast"], 2),
                lower_bound=round(row["lower_bound"], 2),
                upper_bound=round(row["upper_bound"], 2),
            )
            for _, row in df.iterrows()
        ]

    if model_csv.exists():
        import pandas as pd
        models = pd.read_csv(model_csv)
        best = models.iloc[0]
        model_name = best["model"]
        mape = best["mape"]

    return RevenueTimeSeries(
        historical=historical,
        forecast=forecast_points,
        model_name=model_name,
        mape=mape,
    )
"""Pydantic response models. These define the exact JSON shape
the frontend receives. Frontend TypeScript types mirror these."""

from pydantic import BaseModel


class KPISummary(BaseModel):
    total_revenue: float
    active_customers: int
    churn_rate: float
    avg_order_value: float
    total_orders: int
    forecasted_revenue_next_quarter: float | None = None


class RevenueTrendPoint(BaseModel):
    year: int
    month: int
    revenue: float
    active_customers: int
    order_count: int
    mom_growth_pct: float | None = None


class CustomerSegment(BaseModel):
    cluster_id: int
    segment_name: str
    customer_count: int
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    churn_rate: float | None = None


class ChurnRiskCustomer(BaseModel):
    customer_id: int
    churn_probability: float
    risk_tier: str
    segment_name: str | None = None
    recency_days: int
    frequency: int
    monetary: float


class PaginatedChurnRisk(BaseModel):
    items: list[ChurnRiskCustomer]
    total: int
    page: int
    page_size: int


class ModelMetrics(BaseModel):
    model_name: str
    precision: float
    recall: float
    f1: float
    auc_roc: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ForecastPoint(BaseModel):
    date: str
    revenue: float | None = None
    forecast: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


class RevenueTimeSeries(BaseModel):
    historical: list[ForecastPoint]
    forecast: list[ForecastPoint]
    model_name: str
    mape: float
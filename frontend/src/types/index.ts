// Mirrors backend Pydantic schemas exactly

export interface KPISummary {
  total_revenue: number;
  active_customers: number;
  churn_rate: number;
  avg_order_value: number;
  total_orders: number;
  forecasted_revenue_next_quarter: number | null;
}

export interface RevenueTrendPoint {
  year: number;
  month: number;
  revenue: number;
  active_customers: number;
  order_count: number;
  mom_growth_pct: number | null;
}

export interface CustomerSegment {
  cluster_id: number;
  segment_name: string;
  customer_count: number;
  avg_recency: number;
  avg_frequency: number;
  avg_monetary: number;
  churn_rate: number | null;
}

export interface ChurnRiskCustomer {
  customer_id: number;
  churn_probability: number;
  risk_tier: string;
  segment_name: string | null;
  recency_days: number;
  frequency: number;
  monetary: number;
}

export interface PaginatedChurnRisk {
  items: ChurnRiskCustomer[];
  total: number;
  page: number;
  page_size: number;
}

export interface ModelMetrics {
  model_name: string;
  precision: number;
  recall: number;
  f1: number;
  auc_roc: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ForecastPoint {
  date: string;
  revenue: number | null;
  forecast: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
}

export interface RevenueTimeSeries {
  historical: ForecastPoint[];
  forecast: ForecastPoint[];
  model_name: string;
  mape: number;
}
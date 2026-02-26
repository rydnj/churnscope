import type {
  KPISummary, RevenueTrendPoint, CustomerSegment,
  PaginatedChurnRisk, ModelMetrics, FeatureImportance, RevenueTimeSeries,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8001/api/v1";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export const fetchKPISummary = () =>
  fetchJSON<KPISummary>("/kpis/summary");

export const fetchRevenueTrend = (granularity = "monthly") =>
  fetchJSON<RevenueTrendPoint[]>(`/kpis/revenue-trend?granularity=${granularity}`);

export const fetchSegments = () =>
  fetchJSON<CustomerSegment[]>("/segments");

export const fetchChurnRisk = (params: { risk_tier?: string; page?: number; limit?: number } = {}) => {
  const query = new URLSearchParams();
  if (params.risk_tier) query.set("risk_tier", params.risk_tier);
  if (params.page) query.set("page", String(params.page));
  if (params.limit) query.set("limit", String(params.limit));
  return fetchJSON<PaginatedChurnRisk>(`/churn/risk?${query}`);
};

export const fetchModelMetrics = () =>
  fetchJSON<ModelMetrics[]>("/churn/model-metrics");

export const fetchFeatureImportance = () =>
  fetchJSON<FeatureImportance[]>("/churn/feature-importance");

export const fetchForecast = () =>
  fetchJSON<RevenueTimeSeries>("/forecast");
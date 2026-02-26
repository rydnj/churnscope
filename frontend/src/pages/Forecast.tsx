import { useEffect, useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { fetchForecast } from "../api/client";
import type { RevenueTimeSeries } from "../types";
import KPICard from "../components/KPICard";

export default function Forecast() {
  const [data, setData] = useState<RevenueTimeSeries | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchForecast()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40 }}>Loading...</div>;
  if (!data) return <div style={{ padding: 40 }}>Failed to load forecast</div>;

  // Combine historical and forecast into one chart dataset
  const chartData = [
    ...data.historical.map(h => ({
      date: h.date,
      revenue: h.revenue,
      forecast: null as number | null,
      lower: null as number | null,
      upper: null as number | null,
    })),
    ...data.forecast.map(f => ({
      date: f.date,
      revenue: null as number | null,
      forecast: f.forecast,
      lower: f.lower_bound,
      upper: f.upper_bound,
    })),
  ];

  const totalForecast = data.forecast.reduce((s, f) => s + (f.forecast || 0), 0);
  const avgForecast = totalForecast / data.forecast.length;

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>Revenue Forecast</h1>

      <div style={{ display: "flex", gap: 16, marginBottom: 32, flexWrap: "wrap" }}>
        <KPICard
          title="Forecast Model"
          value={data.model_name}
          subtitle={`MAPE: ${data.mape}%`}
          color="#3498db"
        />
        <KPICard
          title="6-Month Forecast"
          value={`£${(totalForecast / 1e6).toFixed(2)}M`}
          subtitle={`Avg £${(avgForecast / 1000).toFixed(0)}k/month`}
          color="#2ecc71"
        />
      </div>

      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}>
        <h2 style={{ marginBottom: 16 }}>Historical + Forecast</h2>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`} />
            <Tooltip
              formatter={(value, name) => {
                if (value === null || value === undefined) return ["–", name];
                return [`£${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, name];
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="revenue" stroke="#2c3e50" strokeWidth={2} dot={{ r: 3 }} name="Historical" connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#e74c3c" strokeWidth={2} strokeDasharray="8 4" dot={{ r: 4 }} name="Forecast" connectNulls={false} />
            <Area type="monotone" dataKey="upper" stroke="none" fill="#e74c3c" fillOpacity={0.1} name="Upper Bound" />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#e74c3c" fillOpacity={0.1} name="Lower Bound" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Forecast table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", marginTop: 24, overflowX: "auto" }}>
        <h2 style={{ marginBottom: 16 }}>Forecast Detail</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
              <th style={{ padding: "12px 16px" }}>Month</th>
              <th style={{ padding: "12px 16px" }}>Forecast</th>
              <th style={{ padding: "12px 16px" }}>Lower Bound</th>
              <th style={{ padding: "12px 16px" }}>Upper Bound</th>
            </tr>
          </thead>
          <tbody>
            {data.forecast.map(f => (
              <tr key={f.date} style={{ borderBottom: "1px solid #ecf0f1" }}>
                <td style={{ padding: "12px 16px", fontWeight: 600 }}>{f.date}</td>
                <td style={{ padding: "12px 16px" }}>£{f.forecast?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td style={{ padding: "12px 16px", color: "#95a5a6" }}>£{f.lower_bound?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td style={{ padding: "12px 16px", color: "#95a5a6" }}>£{f.upper_bound?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
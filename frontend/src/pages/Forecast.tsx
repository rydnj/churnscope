import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceArea,
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

  if (loading) return <div style={{ padding: 40, color: "#7f8c8d" }}>Loading forecast...</div>;
  if (!data) return <div style={{ padding: 40, color: "#e74c3c" }}>Failed to load forecast</div>;

  // Exclude truncated last month (Dec 2011 only has 9 days)
  const avgHistorical = data.historical.reduce((s, h) => s + (h.revenue || 0), 0) / data.historical.length;
  const historical = data.historical.filter((h, i) => {
    if (i === data.historical.length - 1 && (h.revenue || 0) < avgHistorical * 0.5) return false;
    return true;
  });

  const lastHistorical = historical[historical.length - 1];

  // Build unified chart data
  const chartData = [
    ...historical.map(h => ({
      date: h.date,
      revenue: h.revenue,
      forecast: null as number | null,
      lower: null as number | null,
      upper: null as number | null,
    })),
    // Bridge: last historical point also starts the forecast line
    ...(lastHistorical ? [{
      date: lastHistorical.date,
      revenue: lastHistorical.revenue,
      forecast: lastHistorical.revenue,
      lower: null as number | null,
      upper: null as number | null,
    }] : []),
    ...data.forecast.map(f => ({
      date: f.date,
      revenue: null as number | null,
      forecast: f.forecast,
      lower: f.lower_bound,
      upper: f.upper_bound,
    })),
  ];

  // Deduplicate: if bridge date matches last historical, merge them
  const deduped: typeof chartData = [];
  for (const d of chartData) {
    const existing = deduped.find(x => x.date === d.date);
    if (existing && d.forecast !== null) {
      existing.forecast = d.forecast;
    } else if (!existing) {
      deduped.push({ ...d });
    }
  }

  const totalForecast = data.forecast.reduce((s, f) => s + (f.forecast || 0), 0);
  const avgForecast = totalForecast / data.forecast.length;

  // Get forecast dates for ReferenceArea shading
  const forecastDates = data.forecast.map(f => f.date);
  const firstForecast = forecastDates[0];
  const lastForecast = forecastDates[forecastDates.length - 1];

  return (
    <div>
      <h1 style={{ marginBottom: 24, color: "#2c3e50" }}>Revenue Forecast</h1>

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
        <KPICard
          title="Forecast Accuracy"
          value={`${(100 - data.mape).toFixed(1)}%`}
          subtitle="Based on walk-forward validation"
          color={data.mape < 15 ? "#2ecc71" : "#f39c12"}
        />
      </div>

      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", marginBottom: 24 }}>
        <h2 style={{ marginBottom: 4, color: "#2c3e50" }}>Historical + Forecast</h2>
        <p style={{ color: "#7f8c8d", fontSize: 13, marginBottom: 16 }}>
          Shaded region shows the forecast period. Dashed line = predicted revenue.
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={deduped} margin={{ bottom: 20, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
              angle={-45}
              textAnchor="end"
              height={70}
              interval={0}
            />
            <YAxis
              tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`}
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
              domain={["auto", "auto"]}
            />
            <Tooltip
              formatter={(value, name) => {
                if (value === null || value === undefined) return ["–", ""];
                const v = Number(value);
                const labels: Record<string, string> = {
                  revenue: "Historical",
                  forecast: "Forecast",
                  upper: "Upper Bound (95%)",
                  lower: "Lower Bound (95%)",
                };
                return [`£${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, labels[name as string] || name];
              }}
              contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
            />

            {/* Shaded forecast region */}
            {firstForecast && lastForecast && (
              <ReferenceArea
                x1={firstForecast}
                x2={lastForecast}
                fill="#e74c3c"
                fillOpacity={0.06}
                stroke="#e74c3c"
                strokeOpacity={0.15}
              />
            )}

            {/* Historical line */}
            <Line
              type="monotone"
              dataKey="revenue"
              stroke="#2c3e50"
              strokeWidth={2.5}
              dot={{ r: 3, fill: "#2c3e50" }}
              connectNulls={false}
              name="revenue"
            />

            {/* Forecast line */}
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#e74c3c"
              strokeWidth={2.5}
              strokeDasharray="8 4"
              dot={{ r: 4, fill: "#e74c3c", stroke: "#fff", strokeWidth: 2 }}
              connectNulls={false}
              name="forecast"
            />

            {/* Confidence bounds as lighter lines */}
            <Line
              type="monotone"
              dataKey="upper"
              stroke="#e74c3c"
              strokeWidth={1}
              strokeDasharray="4 4"
              strokeOpacity={0.4}
              dot={false}
              connectNulls={false}
              name="upper"
            />
            <Line
              type="monotone"
              dataKey="lower"
              stroke="#e74c3c"
              strokeWidth={1}
              strokeDasharray="4 4"
              strokeOpacity={0.4}
              dot={false}
              connectNulls={false}
              name="lower"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Forecast table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", overflowX: "auto" }}>
        <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Forecast Detail</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
              {["Month", "Forecast", "Lower Bound (95%)", "Upper Bound (95%)"].map(h => (
                <th key={h} style={{ padding: "12px 16px", color: "#7f8c8d", fontSize: 12, textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.forecast.map(f => (
              <tr key={f.date} style={{ borderBottom: "1px solid #f5f6fa" }}>
                <td style={{ padding: "14px 16px", fontWeight: 600, color: "#2c3e50" }}>{f.date}</td>
                <td style={{ padding: "14px 16px", fontWeight: 600, color: "#e74c3c" }}>
                  £{f.forecast?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
                <td style={{ padding: "14px 16px", color: "#95a5a6" }}>
                  £{f.lower_bound?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
                <td style={{ padding: "14px 16px", color: "#95a5a6" }}>
                  £{f.upper_bound?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
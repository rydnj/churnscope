import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  Line, ComposedChart,
} from "recharts";
import { fetchKPISummary, fetchRevenueTrend } from "../api/client";
import type { KPISummary, RevenueTrendPoint } from "../types";
import KPICard from "../components/KPICard";

export default function Dashboard() {
  const [kpis, setKpis] = useState<KPISummary | null>(null);
  const [trend, setTrend] = useState<RevenueTrendPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchKPISummary(), fetchRevenueTrend()])
      .then(([k, t]) => { setKpis(k); setTrend(t); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: "#7f8c8d" }}>Loading dashboard...</div>;
  if (!kpis) return <div style={{ padding: 40, color: "#e74c3c" }}>Failed to load data</div>;

  const chartData = trend.map(t => ({
    label: `${t.year}-${String(t.month).padStart(2, "0")}`,
    revenue: Math.round(t.revenue),
    customers: t.active_customers,
    growth: t.mom_growth_pct,
  }));

  return (
    <div>
      <h1 style={{ marginBottom: 24, color: "#2c3e50" }}>Executive Dashboard</h1>

      <div style={{ display: "flex", gap: 16, marginBottom: 32, flexWrap: "wrap" }}>
        <KPICard
          title="Total Revenue"
          value={`£${(kpis.total_revenue / 1e6).toFixed(1)}M`}
          subtitle={`${kpis.total_orders.toLocaleString()} orders`}
          color="#2ecc71"
        />
        <KPICard
          title="Active Customers"
          value={kpis.active_customers.toLocaleString()}
          color="#3498db"
        />
        <KPICard
          title="Churn Rate"
          value={`${kpis.churn_rate}%`}
          subtitle="Last 90 days"
          color={kpis.churn_rate > 40 ? "#e74c3c" : "#f39c12"}
        />
        <KPICard
          title="Avg Order Value"
          value={`£${kpis.avg_order_value.toFixed(0)}`}
          color="#9b59b6"
        />
      </div>

      {/* Revenue chart with trend line */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", marginBottom: 24 }}>
        <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Monthly Revenue</h2>
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={chartData} margin={{ bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
              angle={-45}
              textAnchor="end"
              height={70}
              interval={0}
            />
            <YAxis
              tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`}
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
            />
            <Tooltip
              formatter={(value, name) => {
                if (name === "revenue") return [`£${Number(value).toLocaleString()}`, "Revenue"];
                return [value, name];
              }}
              labelStyle={{ fontWeight: 600, color: "#2c3e50" }}
              contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
            />
            <Bar dataKey="revenue" fill="#3498db" radius={[4, 4, 0, 0]} opacity={0.85} />
            <Line type="monotone" dataKey="revenue" stroke="#2c3e50" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Active customers chart */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}>
        <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Monthly Active Customers</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
              angle={-45}
              textAnchor="end"
              height={70}
              interval={0}
            />
            <YAxis tick={{ fontSize: 11, fill: "#7f8c8d" }} />
            <Tooltip
              formatter={(value) => [Number(value).toLocaleString(), "Customers"]}
              contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
            />
            <Bar dataKey="customers" fill="#2ecc71" radius={[4, 4, 0, 0]} opacity={0.85} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
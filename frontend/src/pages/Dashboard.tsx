import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
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

  if (loading) return <div style={{ padding: 40 }}>Loading...</div>;
  if (!kpis) return <div style={{ padding: 40 }}>Failed to load data</div>;

  const chartData = trend.map(t => ({
    label: `${t.year}-${String(t.month).padStart(2, "0")}`,
    revenue: Math.round(t.revenue),
    customers: t.active_customers,
  }));

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>Executive Dashboard</h1>

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

      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}>
        <h2 style={{ marginBottom: 16 }}>Monthly Revenue</h2>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`} />
            <Tooltip
              formatter={(value) => [`£${Number(value).toLocaleString()}`, "Revenue"]}
              labelStyle={{ fontWeight: 600 }}
            />
            <Bar dataKey="revenue" fill="#3498db" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
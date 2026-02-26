import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell,
} from "recharts";
import { fetchSegments } from "../api/client";
import type { CustomerSegment } from "../types";

const SEGMENT_COLORS: Record<string, string> = {
  "Champions": "#2ecc71",
  "Promising New": "#3498db",
  "At-Risk Big Spenders": "#f39c12",
  "Lost Low-Value": "#e74c3c",
};

const DEFAULT_COLOR = "#95a5a6";

export default function Segments() {
  const [segments, setSegments] = useState<CustomerSegment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSegments()
      .then(setSegments)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: "#7f8c8d" }}>Loading segments...</div>;

  const pieData = segments.map(s => ({
    name: s.segment_name,
    value: s.customer_count,
    color: SEGMENT_COLORS[s.segment_name] || DEFAULT_COLOR,
  }));

  const barData = [...segments].sort((a, b) => a.avg_monetary - b.avg_monetary);

  return (
    <div>
      <h1 style={{ marginBottom: 24, color: "#2c3e50" }}>Customer Segments</h1>

      <div style={{ display: "flex", gap: 24, marginBottom: 32, flexWrap: "wrap" }}>
        {/* Pie chart */}
        <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", flex: 1, minWidth: 340 }}>
          <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Segment Distribution</h2>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={110}
                innerRadius={50}
                paddingAngle={2}
                label={({ name, percent }) => `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`}
                labelLine={{ stroke: "#bdc3c7" }}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => [Number(value).toLocaleString(), "Customers"]}
                contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Horizontal bar chart */}
        <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", flex: 1, minWidth: 340 }}>
          <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Average Revenue per Segment</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={barData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
              <XAxis
                type="number"
                tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`}
                tick={{ fontSize: 11, fill: "#7f8c8d" }}
              />
              <YAxis
                type="category"
                dataKey="segment_name"
                width={170}
                tick={{ fontSize: 12, fill: "#2c3e50" }}
              />
              <Tooltip
                formatter={(value) => [`£${Number(value).toLocaleString()}`, "Avg Revenue"]}
                contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
              />
              <Bar dataKey="avg_monetary" radius={[0, 6, 6, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={SEGMENT_COLORS[entry.segment_name] || DEFAULT_COLOR} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment cards */}
      <div style={{ display: "flex", gap: 16, marginBottom: 32, flexWrap: "wrap" }}>
        {segments.map(s => (
          <div key={s.cluster_id} style={{
            background: "#fff", borderRadius: 12, padding: 20,
            borderLeft: `5px solid ${SEGMENT_COLORS[s.segment_name] || DEFAULT_COLOR}`,
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)", flex: 1, minWidth: 220,
          }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#2c3e50", marginBottom: 12 }}>
              {s.segment_name}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 13, color: "#555" }}>
              <div><strong>{s.customer_count.toLocaleString()}</strong> customers</div>
              <div>Avg Revenue: <strong>£{s.avg_monetary.toLocaleString()}</strong></div>
              <div>Avg Frequency: <strong>{s.avg_frequency.toFixed(1)}</strong> orders</div>
              <div>Avg Recency: <strong>{s.avg_recency.toFixed(0)}</strong> days</div>
              <div>Churn Rate: <strong style={{
                color: s.churn_rate === null ? "#95a5a6" :
                  s.churn_rate > 50 ? "#e74c3c" : s.churn_rate > 25 ? "#f39c12" : "#2ecc71"
              }}>
                {s.churn_rate !== null ? `${s.churn_rate}%` : "0%"}
              </strong></div>
            </div>
          </div>
        ))}
      </div>

      {/* Detail table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", overflowX: "auto" }}>
        <h2 style={{ marginBottom: 16, color: "#2c3e50" }}>Segment Profiles</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
              {["Segment", "Customers", "Avg Recency", "Avg Frequency", "Avg Revenue", "Churn Rate"].map(h => (
                <th key={h} style={{ padding: "12px 16px", color: "#7f8c8d", fontSize: 12, textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {segments.map(s => (
              <tr key={s.cluster_id} style={{ borderBottom: "1px solid #f5f6fa" }}>
                <td style={{ padding: "14px 16px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: SEGMENT_COLORS[s.segment_name] || DEFAULT_COLOR }} />
                    <span style={{ fontWeight: 600, color: "#2c3e50" }}>{s.segment_name}</span>
                  </div>
                </td>
                <td style={{ padding: "14px 16px" }}>{s.customer_count.toLocaleString()}</td>
                <td style={{ padding: "14px 16px" }}>{s.avg_recency.toFixed(0)} days</td>
                <td style={{ padding: "14px 16px" }}>{s.avg_frequency.toFixed(1)}</td>
                <td style={{ padding: "14px 16px" }}>£{s.avg_monetary.toLocaleString()}</td>
                <td style={{ padding: "14px 16px" }}>
                  <span style={{
                    fontWeight: 600,
                    color: s.churn_rate === null ? "#2ecc71" :
                      s.churn_rate > 50 ? "#e74c3c" : s.churn_rate > 25 ? "#f39c12" : "#2ecc71"
                  }}>
                    {s.churn_rate !== null ? `${s.churn_rate}%` : "0%"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
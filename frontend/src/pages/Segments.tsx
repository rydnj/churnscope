import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Legend,
} from "recharts";
import { fetchSegments } from "../api/client";
import type { CustomerSegment } from "../types";

export default function Segments() {
  const [segments, setSegments] = useState<CustomerSegment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSegments()
      .then(setSegments)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40 }}>Loading...</div>;

  const pieData = segments.map(s => ({
    name: s.segment_name,
    value: s.customer_count,
  }));

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>Customer Segments</h1>

      <div style={{ display: "flex", gap: 24, marginBottom: 32, flexWrap: "wrap" }}>
        {/* Pie chart: segment sizes */}
        <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", flex: 1, minWidth: 300 }}>
          <h2 style={{ marginBottom: 16 }}>Segment Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} fill="#3498db"
                label={({ name, percent }) => `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`}>
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar chart: avg monetary per segment */}
        <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", flex: 1, minWidth: 300 }}>
          <h2 style={{ marginBottom: 16 }}>Average Revenue per Segment</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={segments} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis type="number" tickFormatter={(v: number) => `£${(v / 1000).toFixed(0)}k`} />
              <YAxis type="category" dataKey="segment_name" width={160} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value) => [`£${Number(value).toLocaleString()}`, "Avg Revenue"]} />
              <Bar dataKey="avg_monetary" fill="#3498db" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment detail table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", overflowX: "auto" }}>
        <h2 style={{ marginBottom: 16 }}>Segment Profiles</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
              <th style={{ padding: "12px 16px" }}>Segment</th>
              <th style={{ padding: "12px 16px" }}>Customers</th>
              <th style={{ padding: "12px 16px" }}>Avg Recency</th>
              <th style={{ padding: "12px 16px" }}>Avg Frequency</th>
              <th style={{ padding: "12px 16px" }}>Avg Revenue</th>
              <th style={{ padding: "12px 16px" }}>Churn Rate</th>
            </tr>
          </thead>
          <tbody>
            {segments.map(s => (
              <tr key={s.cluster_id} style={{ borderBottom: "1px solid #ecf0f1" }}>
                <td style={{ padding: "12px 16px", fontWeight: 600 }}>{s.segment_name}</td>
                <td style={{ padding: "12px 16px" }}>{s.customer_count.toLocaleString()}</td>
                <td style={{ padding: "12px 16px" }}>{s.avg_recency.toFixed(0)} days</td>
                <td style={{ padding: "12px 16px" }}>{s.avg_frequency.toFixed(1)}</td>
                <td style={{ padding: "12px 16px" }}>£{s.avg_monetary.toLocaleString()}</td>
                <td style={{ padding: "12px 16px" }}>
                  {s.churn_rate !== null ? (
                    <span style={{ color: s.churn_rate > 50 ? "#e74c3c" : s.churn_rate > 25 ? "#f39c12" : "#2ecc71", fontWeight: 600 }}>
                      {s.churn_rate}%
                    </span>
                  ) : "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
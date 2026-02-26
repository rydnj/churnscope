import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { fetchChurnRisk, fetchFeatureImportance } from "../api/client";
import type { ChurnRiskCustomer, FeatureImportance } from "../types";

const TIER_COLORS: Record<string, string> = {
  high: "#e74c3c",
  medium: "#f39c12",
  low: "#2ecc71",
};

export default function ChurnRisk() {
  const [customers, setCustomers] = useState<ChurnRiskCustomer[]>([]);
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [total, setTotal] = useState(0);
  const [tier, setTier] = useState<string>("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const pageSize = 20;

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchChurnRisk({ risk_tier: tier || undefined, page, limit: pageSize }),
      fetchFeatureImportance(),
    ]).then(([risk, fi]) => {
      setCustomers(risk.items);
      setTotal(risk.total);
      setFeatures(fi);
    }).finally(() => setLoading(false));
  }, [tier, page]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div>
      <h1 style={{ marginBottom: 24 }}>Churn Risk</h1>

      {/* Feature importance chart */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", marginBottom: 24 }}>
        <h2 style={{ marginBottom: 16 }}>What Drives Churn?</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={features.slice(0, 10)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={180} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => [Number(v).toFixed(4), "Importance"]} />
            <Bar dataKey="importance" fill="#e74c3c" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Filters and table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", overflowX: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
          <h2 style={{ margin: 0 }}>At-Risk Customers ({total.toLocaleString()})</h2>
          <div style={{ display: "flex", gap: 8 }}>
            {["", "high", "medium", "low"].map(t => (
              <button
                key={t}
                onClick={() => { setTier(t); setPage(1); }}
                style={{
                  padding: "8px 16px", border: "none", borderRadius: 6, cursor: "pointer",
                  background: tier === t ? "#2c3e50" : "#ecf0f1",
                  color: tier === t ? "#fff" : "#2c3e50",
                  fontWeight: 600, fontSize: 13,
                }}
              >
                {t || "All"}
              </button>
            ))}
          </div>
        </div>

        {loading ? <div>Loading...</div> : (
          <>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
                  <th style={{ padding: "12px 16px" }}>Customer ID</th>
                  <th style={{ padding: "12px 16px" }}>Churn Probability</th>
                  <th style={{ padding: "12px 16px" }}>Risk</th>
                  <th style={{ padding: "12px 16px" }}>Segment</th>
                  <th style={{ padding: "12px 16px" }}>Recency</th>
                  <th style={{ padding: "12px 16px" }}>Frequency</th>
                  <th style={{ padding: "12px 16px" }}>Revenue</th>
                </tr>
              </thead>
              <tbody>
                {customers.map(c => (
                  <tr key={c.customer_id} style={{ borderBottom: "1px solid #ecf0f1" }}>
                    <td style={{ padding: "12px 16px" }}>{c.customer_id}</td>
                    <td style={{ padding: "12px 16px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 60, height: 8, background: "#ecf0f1", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ width: `${c.churn_probability * 100}%`, height: "100%", background: TIER_COLORS[c.risk_tier] || "#999", borderRadius: 4 }} />
                        </div>
                        <span>{(c.churn_probability * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      <span style={{ padding: "4px 10px", borderRadius: 12, fontSize: 12, fontWeight: 600, color: "#fff", background: TIER_COLORS[c.risk_tier] || "#999" }}>
                        {c.risk_tier}
                      </span>
                    </td>
                    <td style={{ padding: "12px 16px" }}>{c.segment_name || "–"}</td>
                    <td style={{ padding: "12px 16px" }}>{c.recency_days}d</td>
                    <td style={{ padding: "12px 16px" }}>{c.frequency}</td>
                    <td style={{ padding: "12px 16px" }}>£{c.monetary.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 16 }}>
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                style={{ padding: "8px 16px", border: "1px solid #ddd", borderRadius: 6, cursor: "pointer", background: "#fff" }}>
                ← Prev
              </button>
              <span style={{ padding: "8px 16px", lineHeight: "24px" }}>
                Page {page} of {totalPages}
              </span>
              <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                style={{ padding: "8px 16px", border: "1px solid #ddd", borderRadius: 6, cursor: "pointer", background: "#fff" }}>
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
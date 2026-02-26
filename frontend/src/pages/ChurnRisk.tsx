import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { fetchChurnRisk, fetchFeatureImportance, fetchModelMetrics } from "../api/client";
import type { ChurnRiskCustomer, FeatureImportance, ModelMetrics } from "../types";
import KPICard from "../components/KPICard";

const TIER_COLORS: Record<string, string> = {
  high: "#e74c3c",
  medium: "#f39c12",
  low: "#2ecc71",
};

// Clean up encoded feature names for display
function cleanFeatureName(name: string): string {
  return name
    .replace(/_enc$/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

export default function ChurnRisk() {
  const [customers, setCustomers] = useState<ChurnRiskCustomer[]>([]);
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
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
      fetchModelMetrics(),
    ]).then(([risk, fi, m]) => {
      setCustomers(risk.items);
      setTotal(risk.total);
      setFeatures(fi);
      setMetrics(m);
    }).finally(() => setLoading(false));
  }, [tier, page]);

  const totalPages = Math.ceil(total / pageSize);

  // Clean feature names and limit to top 8 for readability
  const chartFeatures = features.slice(0, 8).map(f => ({
    ...f,
    name: cleanFeatureName(f.feature),
  })).reverse(); // Reverse so highest is at top in horizontal bar

  const bestModel = metrics.length > 0 ? metrics[0] : null;

  return (
    <div>
      <h1 style={{ marginBottom: 24, color: "#2c3e50" }}>Churn Risk</h1>

      {/* Model KPIs */}
      {bestModel && (
        <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
          <KPICard
            title="Best Model"
            value={bestModel.model_name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
            subtitle={`AUC-ROC: ${bestModel.auc_roc.toFixed(4)}`}
            color="#3498db"
          />
          <KPICard
            title="Precision"
            value={`${(bestModel.precision * 100).toFixed(1)}%`}
            color="#2ecc71"
          />
          <KPICard
            title="Recall"
            value={`${(bestModel.recall * 100).toFixed(1)}%`}
            color="#f39c12"
          />
          <KPICard
            title="F1 Score"
            value={`${(bestModel.f1 * 100).toFixed(1)}%`}
            color="#9b59b6"
          />
        </div>
      )}

      {/* Feature importance chart */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", marginBottom: 24 }}>
        <h2 style={{ marginBottom: 4, color: "#2c3e50" }}>What Drives Churn?</h2>
        <p style={{ color: "#7f8c8d", fontSize: 13, marginBottom: 16 }}>
          Feature importance from the {bestModel?.model_name.replace(/_/g, " ") || "best"} model
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartFeatures} layout="vertical" margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#7f8c8d" }}
              domain={[0, "auto"]}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tick={{ fontSize: 12, fill: "#2c3e50" }}
            />
            <Tooltip
              formatter={(v) => [Number(v).toFixed(4), "Importance"]}
              contentStyle={{ borderRadius: 8, border: "1px solid #ecf0f1" }}
            />
            <Bar dataKey="importance" fill="#e74c3c" radius={[0, 6, 6, 0]} barSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Customer risk table */}
      <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 2px 8px rgba(0,0,0,0.08)", overflowX: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
          <h2 style={{ margin: 0, color: "#2c3e50" }}>
            At-Risk Customers
            <span style={{ fontSize: 14, fontWeight: 400, color: "#7f8c8d", marginLeft: 8 }}>
              ({total.toLocaleString()} total)
            </span>
          </h2>
          <div style={{ display: "flex", gap: 6 }}>
            {[
              { key: "", label: "All" },
              { key: "high", label: "High" },
              { key: "medium", label: "Medium" },
              { key: "low", label: "Low" },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => { setTier(t.key); setPage(1); }}
                style={{
                  padding: "8px 16px", border: "none", borderRadius: 6, cursor: "pointer",
                  background: tier === t.key ? (TIER_COLORS[t.key] || "#2c3e50") : "#f5f6fa",
                  color: tier === t.key ? "#fff" : "#2c3e50",
                  fontWeight: 600, fontSize: 13, transition: "all 0.15s",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: "center", color: "#7f8c8d" }}>Loading...</div>
        ) : (
          <>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #ecf0f1", textAlign: "left" }}>
                  {["Customer ID", "Churn Probability", "Risk", "Segment", "Recency", "Frequency", "Revenue"].map(h => (
                    <th key={h} style={{ padding: "12px 16px", color: "#7f8c8d", fontSize: 12, textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {customers.map(c => (
                  <tr key={c.customer_id} style={{ borderBottom: "1px solid #f5f6fa" }}>
                    <td style={{ padding: "12px 16px", fontWeight: 500 }}>{c.customer_id}</td>
                    <td style={{ padding: "12px 16px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 80, height: 8, background: "#f0f0f0", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{
                            width: `${c.churn_probability * 100}%`,
                            height: "100%",
                            background: TIER_COLORS[c.risk_tier] || "#999",
                            borderRadius: 4,
                            transition: "width 0.3s",
                          }} />
                        </div>
                        <span style={{ fontSize: 13, fontWeight: 500, minWidth: 45 }}>
                          {(c.churn_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      <span style={{
                        padding: "4px 12px", borderRadius: 12, fontSize: 11,
                        fontWeight: 700, color: "#fff", textTransform: "uppercase",
                        background: TIER_COLORS[c.risk_tier] || "#999",
                        letterSpacing: 0.5,
                      }}>
                        {c.risk_tier}
                      </span>
                    </td>
                    <td style={{ padding: "12px 16px", fontSize: 13 }}>{c.segment_name || "–"}</td>
                    <td style={{ padding: "12px 16px", fontSize: 13 }}>{c.recency_days}d</td>
                    <td style={{ padding: "12px 16px", fontSize: 13 }}>{c.frequency}</td>
                    <td style={{ padding: "12px 16px", fontSize: 13 }}>£{c.monetary.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 12, marginTop: 20 }}>
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                style={{
                  padding: "8px 20px", border: "1px solid #ddd", borderRadius: 6,
                  cursor: page === 1 ? "default" : "pointer",
                  background: "#fff", color: page === 1 ? "#ccc" : "#2c3e50",
                  fontWeight: 500,
                }}
              >
                ← Prev
              </button>
              <span style={{ fontSize: 13, color: "#7f8c8d" }}>
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                style={{
                  padding: "8px 20px", border: "1px solid #ddd", borderRadius: 6,
                  cursor: page === totalPages ? "default" : "pointer",
                  background: "#fff", color: page === totalPages ? "#ccc" : "#2c3e50",
                  fontWeight: 500,
                }}
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
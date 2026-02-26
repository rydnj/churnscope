interface KPICardProps {
  title: string;
  value: string;
  subtitle?: string;
  color?: string;
}

export default function KPICard({ title, value, subtitle, color = "#3498db" }: KPICardProps) {
  return (
    <div style={{
      background: "#fff",
      borderRadius: 12,
      padding: "24px",
      borderTop: `4px solid ${color}`,
      boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
      minWidth: 200,
      flex: 1,
    }}>
      <div style={{ fontSize: 13, color: "#7f8c8d", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
        {title}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#2c3e50" }}>
        {value}
      </div>
      {subtitle && (
        <div style={{ fontSize: 13, color: "#95a5a6", marginTop: 4 }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}
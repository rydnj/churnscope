import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Segments from "./pages/Segments";
import ChurnRisk from "./pages/ChurnRisk";
import Forecast from "./pages/Forecast";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard" },
  { path: "/segments", label: "Segments" },
  { path: "/churn", label: "Churn Risk" },
  { path: "/forecast", label: "Forecast" },
];

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: "flex", minHeight: "100vh", background: "#f5f6fa" }}>
        {/* Sidebar */}
        <nav style={{
          width: 220, background: "#2c3e50", padding: "24px 0",
          display: "flex", flexDirection: "column",
          position: "fixed", top: 0, left: 0, bottom: 0,
        }}>
          <div style={{ padding: "0 24px 24px", borderBottom: "1px solid #34495e" }}>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#fff" }}>ChurnScope</div>
            <div style={{ fontSize: 12, color: "#7f8c8d", marginTop: 4 }}>Analytics Platform</div>
          </div>

          <div style={{ marginTop: 24, display: "flex", flexDirection: "column", gap: 4 }}>
            {NAV_ITEMS.map(item => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/"}
                style={({ isActive }) => ({
                  padding: "12px 24px",
                  color: isActive ? "#fff" : "#bdc3c7",
                  background: isActive ? "#34495e" : "transparent",
                  textDecoration: "none",
                  fontSize: 14,
                  fontWeight: isActive ? 600 : 400,
                  borderLeft: isActive ? "3px solid #3498db" : "3px solid transparent",
                })}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Main content */}
        <main style={{ marginLeft: 220, flex: 1, padding: 32 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/segments" element={<Segments />} />
            <Route path="/churn" element={<ChurnRisk />} />
            <Route path="/forecast" element={<Forecast />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
import type { CSSProperties } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { Logo } from "./Logo";

function initials(fullName: string): string {
  return fullName
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]!.toUpperCase())
    .join("");
}

export function TopBar({ active }: { active: "analysis" | "history" | "none" }) {
  const { user, logout } = useAuth();

  const tabStyle = (isActive: boolean): CSSProperties => ({
    padding: "9px 16px",
    borderRadius: 9,
    fontSize: 14,
    fontWeight: 600,
    whiteSpace: "nowrap",
    textDecoration: "none",
    background: isActive ? "var(--hv-brand-soft)" : "transparent",
    color: isActive ? "var(--hv-brand-dark)" : "var(--hv-text-muted)",
  });

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "16px 48px",
        background: "#fff",
        borderBottom: "1px solid var(--hv-border)",
        boxShadow: "var(--hv-shadow-topbar)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 36 }}>
        <Link to="/analysis">
          <Logo height={26} />
        </Link>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <Link to="/analysis" style={tabStyle(active === "analysis")}>
            Анализ препарата
          </Link>
          <Link to="/history" style={tabStyle(active === "history")}>
            История случаев
          </Link>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, whiteSpace: "nowrap" }}>
        <div
          style={{
            width: 38,
            height: 38,
            borderRadius: 11,
            background: "var(--hv-brand-gradient)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 13,
            fontWeight: 700,
            color: "#fff",
            flex: "none",
          }}
        >
          {user ? initials(user.full_name) : "?"}
        </div>
        <div>
          <div style={{ fontSize: 13.5, fontWeight: 700, lineHeight: 1.25, fontFamily: "var(--hv-font-display)" }}>
            {user ? `Др. ${user.full_name}` : ""}
          </div>
          <div style={{ fontSize: 11.5, color: "var(--hv-text-faint)", lineHeight: 1.25 }}>{user?.role}</div>
        </div>
        <button
          onClick={logout}
          title="Выйти из системы"
          style={{
            marginLeft: 4, padding: "8px 10px", borderRadius: 9, border: "1px solid var(--hv-border)",
            background: "transparent", color: "var(--hv-text-muted)", fontSize: 12.5, fontWeight: 600, cursor: "pointer",
          }}
        >
          Выйти
        </button>
      </div>
    </div>
  );
}

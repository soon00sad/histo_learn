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
    padding: "9px 14px",
    fontSize: 12.5,
    fontWeight: 700,
    letterSpacing: "0.04em",
    textTransform: "uppercase",
    whiteSpace: "nowrap",
    textDecoration: "none",
    borderBottom: isActive ? "2px solid var(--hv-accent-green)" : "2px solid transparent",
    color: isActive ? "var(--hv-text)" : "var(--hv-text-muted)",
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
            marginLeft: 4, padding: "8px 12px", borderRadius: 7, border: "1px solid var(--hv-border)",
            background: "transparent", color: "var(--hv-text-muted)", fontSize: 11.5, fontWeight: 700,
            letterSpacing: "0.04em", textTransform: "uppercase", cursor: "pointer",
          }}
        >
          Выйти
        </button>
      </div>
    </div>
  );
}

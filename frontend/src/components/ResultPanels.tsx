import type { RegionOut } from "../api/types";

export interface VerdictLike {
  verdict_label: string;
  is_malignant: boolean;
  confidence: number;
  malignant_probability: number;
  benign_probability: number;
  top_regions: RegionOut[];
}

export function VerdictCard({ result }: { result: VerdictLike }) {
  const malignantColor = "var(--hv-malignant)";
  const benignColor = "var(--hv-benign)";
  return (
    <div style={{ background: "#fff", borderRadius: "var(--hv-radius-lg)", padding: "26px 28px", boxShadow: "var(--hv-shadow-card)" }}>
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.06em", color: "var(--hv-text-faint)", textTransform: "uppercase", marginBottom: 10 }}>
        Вердикт модели
      </div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 14 }}>
        <div style={{ fontSize: 25, fontWeight: 700, color: result.is_malignant ? malignantColor : benignColor, letterSpacing: "-0.01em", whiteSpace: "nowrap", fontFamily: "var(--hv-font-display)" }}>
          {result.verdict_label}
        </div>
        <div style={{ fontSize: 29, fontWeight: 700, color: "var(--hv-text)", whiteSpace: "nowrap", fontFamily: "var(--hv-font-display)" }}>
          {(result.confidence * 100).toFixed(1)}%
        </div>
      </div>
      <div style={{ fontSize: 12.5, color: "var(--hv-text-faint)", marginTop: 2 }}>уверенность модели</div>

      <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 10 }}>
        <ProbabilityBar label="Злокачественная" value={result.malignant_probability} color={malignantColor} />
        <ProbabilityBar label="Доброкачественная" value={result.benign_probability} color={benignColor} />
      </div>
    </div>
  );
}

export function ProbabilityBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 5 }}>
        <span style={{ color: "oklch(0.35 0.04 264)", fontWeight: 600 }}>{label}</span>
        <span style={{ fontWeight: 700, color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 8, borderRadius: 5, background: "oklch(0.93 0.008 264)", overflow: "hidden" }}>
        <div style={{ width: `${value * 100}%`, height: "100%", background: color }} />
      </div>
    </div>
  );
}

export function TopZonesCard({ regions }: { regions: RegionOut[] }) {
  return (
    <div style={{ background: "#fff", borderRadius: "var(--hv-radius-lg)", padding: "22px 24px", boxShadow: "var(--hv-shadow-card)" }}>
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.06em", color: "var(--hv-text-faint)", textTransform: "uppercase", marginBottom: 14 }}>
        Топ-{regions.length} зоны внимания
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {regions.map((region, i) => (
          <div
            key={region.rank}
            style={{
              display: "flex", alignItems: "center", gap: 12, padding: "9px 0",
              borderBottom: i < regions.length - 1 ? "1px solid var(--hv-border-light)" : "none",
            }}
          >
            <div style={{ width: 24, height: 24, borderRadius: 7, background: "var(--hv-malignant-soft)", color: "var(--hv-malignant)", fontSize: 12, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flex: "none" }}>
              {region.rank}
            </div>
            <div style={{ flex: 1, fontSize: 13, color: "oklch(0.3 0.03 264)" }}>
              X: {region.x}&nbsp;&nbsp;Y: {region.y}
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--hv-malignant)" }}>
              {(region.score * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

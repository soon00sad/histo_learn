import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { TopBar } from "../components/TopBar";
import { api, ApiError, parseUtc } from "../api/client";
import type { CaseSummary } from "../api/types";

type VerdictFilter = "all" | "malignant" | "benign";
type StatusFilter = "all" | "pending" | "confirmed" | "rejected";
type SortMode = "priority" | "date";

const FILTERS: { label: string; status: StatusFilter; verdict: VerdictFilter }[] = [
  { label: "Все случаи", status: "all", verdict: "all" },
  { label: "Подтверждено врачом", status: "confirmed", verdict: "all" },
  { label: "На рассмотрении", status: "pending", verdict: "all" },
  { label: "Отклонено врачом", status: "rejected", verdict: "all" },
  { label: "Злокачественные", status: "all", verdict: "malignant" },
  { label: "Доброкачественные", status: "all", verdict: "benign" },
];

const STATUS_LABEL: Record<CaseSummary["status"], string> = {
  confirmed: "Подтверждён",
  pending: "На рассмотрении",
  rejected: "Отклонён врачом",
};
const STATUS_COLOR: Record<CaseSummary["status"], string> = {
  confirmed: "var(--hv-benign)",
  pending: "var(--hv-pending)",
  rejected: "var(--hv-malignant)",
};
const STATUS_DOT: Record<CaseSummary["status"], string> = {
  confirmed: "var(--hv-benign)",
  pending: "var(--hv-pending-dot)",
  rejected: "var(--hv-malignant)",
};

export function HistoryPage() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [activeFilter, setActiveFilter] = useState(0);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortMode>("priority");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const filter = FILTERS[activeFilter];
    const params: { status?: string; verdict?: string; search?: string; sort: SortMode } = { sort };
    if (filter.status !== "all") params.status = filter.status;
    if (filter.verdict !== "all") params.verdict = filter.verdict;
    if (search.trim()) params.search = search.trim();

    api
      .listCases(params)
      .then(setCases)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось загрузить список случаев"));
  }, [activeFilter, search, sort]);

  return (
    <div style={{ minHeight: "100vh", background: "var(--hv-bg)", fontFamily: "var(--hv-font-body)" }}>
      <TopBar active="history" />

      <div style={{ padding: "36px 48px 20px", display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 18 }}>
        <div style={{ fontSize: 32, fontWeight: 600, letterSpacing: "-0.015em", fontFamily: "var(--hv-font-display)" }}>
          История случаев
        </div>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Поиск по ID случая…"
          style={{ width: 240, padding: "10px 14px", borderRadius: 10, border: "1.5px solid oklch(0.88 0.008 264)", fontSize: 13.5, background: "#fff" }}
        />
      </div>

      <div style={{ padding: "0 48px 12px", display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {FILTERS.map((filter, i) => (
            <button
              key={filter.label}
              onClick={() => setActiveFilter(i)}
              style={{
                padding: "8px 15px", borderRadius: 9, whiteSpace: "nowrap", fontSize: 13, fontWeight: 700,
                border: activeFilter === i ? "none" : "1px solid var(--hv-border)",
                background: activeFilter === i ? "var(--hv-brand)" : "#fff",
                color: activeFilter === i ? "#fff" : "oklch(0.4 0.02 264)",
                cursor: "pointer",
              }}
            >
              {filter.label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "var(--hv-text-muted)" }}>
          Сортировка:
          {(["priority", "date"] as SortMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setSort(mode)}
              style={{
                padding: "6px 12px", borderRadius: 8, fontSize: 12.5, fontWeight: 700, cursor: "pointer",
                border: sort === mode ? "none" : "1px solid var(--hv-border)",
                background: sort === mode ? "oklch(0.4 0.02 264)" : "#fff",
                color: sort === mode ? "#fff" : "oklch(0.4 0.02 264)",
              }}
            >
              {mode === "priority" ? "по приоритету" : "по дате"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "0 48px 48px" }}>
        <div style={{ background: "#fff", borderRadius: 18, boxShadow: "var(--hv-shadow-card)", overflow: "hidden" }}>
          <div style={headerRowStyle}>
            <div>ID случая</div>
            <div>Дата</div>
            <div>Вердикт</div>
            <div>Доля опухоли</div>
            <div>Статус</div>
            <div />
          </div>

          {error && <div style={{ padding: 24, color: "var(--hv-malignant)" }}>{error}</div>}
          {!error && cases.length === 0 && (
            <div style={{ padding: 24, color: "var(--hv-text-muted)" }}>Случаи не найдены.</div>
          )}

          {cases.map((c, i) => (
            <Link
              key={c.id}
              to={`/cases/${c.id}`}
              style={{
                textDecoration: "none", color: "inherit", display: "grid",
                gridTemplateColumns: "1.3fr 1.1fr 1.3fr 0.9fr 1.3fr 0.4fr", alignItems: "center",
                padding: "15px 24px", borderBottom: i < cases.length - 1 ? "1px solid oklch(0.95 0.004 264)" : "none",
              }}
            >
              <div style={{ fontSize: 13.5, fontWeight: 700 }}>{c.id}</div>
              <div style={{ fontSize: 13, color: "oklch(0.45 0.02 264)" }}>
                {parseUtc(c.created_at).toLocaleDateString("ru-RU")}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                <span
                  style={{
                    padding: "4px 10px", borderRadius: 7, fontSize: 12.5, fontWeight: 700,
                    background: c.is_malignant ? "var(--hv-malignant-soft)" : "var(--hv-benign-soft)",
                    color: c.is_malignant ? "var(--hv-malignant)" : "var(--hv-benign)",
                  }}
                >
                  {c.verdict_label}
                </span>
                {c.mask_source !== "model" && (
                  <span
                    title="Демонстрация на эталонной маске BCSS, не вывод обученной модели"
                    style={{ padding: "3px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600, background: "oklch(0.95 0.01 264)", color: "oklch(0.6 0.02 264)" }}
                  >
                    эталон
                  </span>
                )}
              </div>
              <div style={{ fontSize: 13.5, fontWeight: 700 }}>{(c.tumor_area_fraction * 100).toFixed(1)}%</div>
              <div>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12.5, fontWeight: 600, color: STATUS_COLOR[c.status] }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: STATUS_DOT[c.status] }} />
                  {STATUS_LABEL[c.status]}
                </span>
              </div>
              <div style={{ textAlign: "right", color: "oklch(0.7 0.01 264)" }}>›</div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

const headerRowStyle = {
  display: "grid",
  gridTemplateColumns: "1.3fr 1.1fr 1.3fr 0.9fr 1.3fr 0.4fr",
  padding: "14px 24px",
  fontSize: 11.5,
  fontWeight: 700,
  letterSpacing: "0.05em",
  textTransform: "uppercase" as const,
  color: "var(--hv-text-faint)",
  borderBottom: "1px solid var(--hv-border-light)",
};

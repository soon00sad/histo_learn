import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { JobStatusOut } from "../api/types";

const POLL_INTERVAL_MS = 1500;

// The design mock shows 4 illustrative steps in a fixed order; the actual
// pipeline (src/inference/wsi_aggregator.py) tiles first, then normalizes
// + classifies each tile together, then builds the heatmap — so the
// "normalization" stage signal is folded into the middle step here rather
// than shown as its own row, to stay honest about real execution order.
const STAGE_RANK: Record<string, number> = { tiling: 0, normalization: 1, inference: 1, heatmap: 2 };
const STEPS = [
  { rank: 0, title: "Нарезка на фрагменты", detail: "Препарат разбивается на фрагменты, фон отсеивается" },
  { rank: 1, title: "Нормализация окраски и анализ нейросетью", detail: "Каждый фрагмент нормализуется и классифицируется" },
  { rank: 2, title: "Построение тепловой карты", detail: "Результаты фрагментов собираются в общую карту" },
];

function overallProgress(job: JobStatusOut): number {
  if (job.status === "done") return 1;
  const rank = STAGE_RANK[job.stage] ?? 0;
  const local = job.stage === "tiling" || job.stage === "heatmap" ? job.progress : job.stage === "inference" ? job.progress : 0;
  return Math.min(1, rank / 3 + local / 3);
}

export function ProcessingPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<JobStatusOut | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const status = await api.getJob(jobId);
        setJob(status);
        if (status.status === "done" && status.case_id) {
          if (intervalRef.current) window.clearInterval(intervalRef.current);
          navigate(`/cases/${status.case_id}`);
        } else if (status.status === "failed") {
          if (intervalRef.current) window.clearInterval(intervalRef.current);
        }
      } catch {
        setPollError("Не удалось получить статус обработки. Проверьте соединение с сервером.");
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [jobId, navigate]);

  const percent = job ? Math.round(overallProgress(job) * 100) : 0;
  const currentRank = job ? STAGE_RANK[job.stage] ?? 0 : 0;

  return (
    <div style={{ minHeight: "100vh", background: "var(--hv-bg)", fontFamily: "var(--hv-font-body)", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "16px 48px", background: "#fff", borderBottom: "1px solid var(--hv-border)" }}>
        <img src="/logo.png" alt="HistoVision" style={{ height: 24, width: "auto" }} />
      </div>

      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
        <div style={{ width: "100%", maxWidth: 560, background: "#fff", borderRadius: 22, padding: "44px 44px 38px", boxShadow: "var(--hv-shadow-card)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 18, marginBottom: 8 }}>
            <ProgressRing percent={percent} failed={job?.status === "failed"} />
            <div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "var(--hv-font-display)", letterSpacing: "-0.01em" }}>
                Анализ препарата
              </div>
              <div style={{ fontSize: 12.5, color: "var(--hv-text-muted)", marginTop: 3 }}>
                {job?.message ?? "Подготовка…"} · не закрывайте страницу
              </div>
            </div>
          </div>

          {pollError && <div style={{ marginTop: 16, color: "var(--hv-malignant)", fontSize: 13 }}>{pollError}</div>}

          {job?.status === "failed" ? (
            <div style={{ marginTop: 24 }}>
              <div style={{ color: "var(--hv-malignant)", fontWeight: 600, marginBottom: 12 }}>
                {job.error ?? "Анализ не удался."}
              </div>
              <Link to="/analysis" style={{ fontWeight: 700 }}>
                ← Вернуться и попробовать снова
              </Link>
            </div>
          ) : (
            <>
              <div style={{ marginTop: 30, display: "flex", flexDirection: "column", gap: 4 }}>
                {STEPS.map((step, i) => (
                  <StepRow
                    key={step.rank}
                    title={step.title}
                    detail={step.detail}
                    state={currentRank > step.rank ? "done" : currentRank === step.rank ? "active" : "pending"}
                    showConnector={i < STEPS.length - 1}
                  />
                ))}
              </div>

              <div style={{ marginTop: 26, height: 8, borderRadius: 5, background: "oklch(0.93 0.008 264)", overflow: "hidden" }}>
                <div style={{ width: `${percent}%`, height: "100%", background: "var(--hv-brand-gradient)", transition: "width 0.4s ease" }} />
              </div>
              <div style={{ marginTop: 10, fontSize: 12, color: "var(--hv-text-muted)", textAlign: "center" }}>{percent}%</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ProgressRing({ percent, failed }: { percent: number; failed: boolean }) {
  const radius = 27;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - percent / 100);
  return (
    <div style={{ position: "relative", width: 64, height: 64, flex: "none" }}>
      <svg width={64} height={64} viewBox="0 0 64 64" style={{ position: "absolute", inset: 0 }}>
        <circle cx={32} cy={32} r={radius} fill="none" stroke="oklch(0.92 0.008 264)" strokeWidth={6} />
        <circle
          cx={32} cy={32} r={radius} fill="none"
          stroke={failed ? "var(--hv-malignant)" : "var(--hv-brand)"}
          strokeWidth={6} strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.4s ease" }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, fontFamily: "var(--hv-font-display)" }}>
        {percent}%
      </div>
    </div>
  );
}

function StepRow({ title, detail, state, showConnector }: { title: string; detail: string; state: "done" | "active" | "pending"; showConnector: boolean }) {
  return (
    <>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14, padding: "13px 0", opacity: state === "pending" ? 0.5 : 1 }}>
        <div
          style={{
            width: 26, height: 26, borderRadius: "50%", flex: "none", marginTop: 1,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: state === "done" ? "oklch(0.6 0.15 150 / 15%)" : state === "active" ? "var(--hv-brand-soft)" : "oklch(0.9 0.006 264)",
          }}
        >
          {state === "done" && <span style={{ color: "oklch(0.45 0.15 150)", fontSize: 13 }}>✓</span>}
          {state === "active" && <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--hv-brand)" }} />}
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700 }}>{title}</div>
          <div style={{ fontSize: 12.5, color: "var(--hv-text-faint)", marginTop: 2 }}>{detail}</div>
        </div>
      </div>
      {showConnector && <div style={{ width: 1.5, height: 14, background: "var(--hv-border)", marginLeft: 12.5 }} />}
    </>
  );
}

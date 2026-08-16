import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { TopBar } from "../components/TopBar";
import { Disclaimer } from "../components/Disclaimer";
import { VerdictCard, TopZonesCard } from "../components/ResultPanels";
import { api, ApiError, fetchAuthenticatedBlobUrl } from "../api/client";
import type { AnalysisResult, IhcMarkersInput } from "../api/types";

type Mode = "patch" | "wsi";

const TISSUE_TYPE = "Молочная железа, биопсия";
const NOT_SPECIFIED = "Не указан";

export function AnalysisPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState<Mode>("patch");
  const [file, setFile] = useState<File | null>(null);
  const [ki67, setKi67] = useState(0);
  const [erStatus, setErStatus] = useState(NOT_SPECIFIED);
  const [prStatus, setPrStatus] = useState(NOT_SPECIFIED);
  const [her2, setHer2] = useState(NOT_SPECIFIED);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!result) return;
    let cancelled = false;
    Promise.all([
      fetchAuthenticatedBlobUrl(api.caseImageUrl(result.case_id)),
      fetchAuthenticatedBlobUrl(api.caseHeatmapUrl(result.case_id)),
    ]).then(([src, heat]) => {
      if (cancelled) return;
      setSourceUrl(src);
      setHeatmapUrl(heat);
    });
    return () => {
      cancelled = true;
    };
  }, [result]);

  const buildIhc = (): IhcMarkersInput => ({
    ki67: ki67 > 0 ? ki67 : undefined,
    er_status: erStatus !== NOT_SPECIFIED ? erStatus : undefined,
    pr_status: prStatus !== NOT_SPECIFIED ? prStatus : undefined,
    her2_status: her2 !== NOT_SPECIFIED ? her2 : undefined,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setIsSubmitting(true);
    setError(null);
    try {
      if (mode === "patch") {
        const analysis = await api.analyzePatch(file, TISSUE_TYPE, buildIhc());
        setResult(analysis);
      } else {
        const job = await api.analyzeWsi(file, TISSUE_TYPE, buildIhc());
        navigate(`/processing/${job.job_id}`);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось выполнить анализ");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGenerateReport = () => {
    if (!result) return;
    navigate(`/cases/${result.case_id}/report`);
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--hv-bg)", fontFamily: "var(--hv-font-body)" }}>
      <TopBar active="analysis" />

      <div style={{ padding: "36px 48px 8px", display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 20 }}>
        <div>
          <div style={{ fontSize: 32, fontWeight: 600, letterSpacing: "-0.015em", lineHeight: 1.1, fontFamily: "var(--hv-font-display)" }}>
            Анализ препарата
          </div>
          {result && (
            <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 14, fontSize: 13, color: "var(--hv-text-muted)" }}>
              <span>
                Случай <b style={{ color: "oklch(0.3 0.05 264)", fontWeight: 700 }}>{result.case_id}</b>
              </span>
              <Dot />
              <span>{TISSUE_TYPE}</span>
            </div>
          )}
        </div>

        <div style={{ display: "flex", background: "oklch(0.94 0.008 264)", padding: 4, borderRadius: 12, gap: 2 }}>
          {(["patch", "wsi"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              disabled={Boolean(result) || isSubmitting}
              style={{
                padding: "9px 18px",
                borderRadius: 9,
                border: "none",
                cursor: result ? "default" : "pointer",
                background: mode === m ? "#fff" : "transparent",
                color: mode === m ? "oklch(0.22 0.05 264)" : "var(--hv-text-muted)",
                fontSize: 13.5,
                fontWeight: mode === m ? 700 : 600,
                whiteSpace: "nowrap",
                boxShadow: mode === m ? "0 2px 8px -2px oklch(0.3 0.1 264 / 25%)" : "none",
              }}
            >
              {m === "patch" ? "Живой анализ" : "Полный препарат"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "28px 48px 16px", display: "grid", gridTemplateColumns: "1.05fr 0.85fr", gap: 28, alignItems: "start" }}>
        <div style={{ background: "#fff", borderRadius: "var(--hv-radius-lg)", padding: 24, boxShadow: "var(--hv-shadow-card)" }}>
          <UploadRow file={file} onPick={() => fileInputRef.current?.click()} mode={mode} />
          <input
            ref={fileInputRef}
            type="file"
            accept={mode === "patch" ? "image/png,image/jpeg" : ".svs,.tif,.tiff,.ndpi,.mrxs,.vms,.vmu,.scn"}
            style={{ display: "none" }}
            onChange={(e) => {
              setResult(null);
              setFile(e.target.files?.[0] ?? null);
            }}
          />

          {!result ? (
            <IhcForm
              ki67={ki67}
              setKi67={setKi67}
              erStatus={erStatus}
              setErStatus={setErStatus}
              prStatus={prStatus}
              setPrStatus={setPrStatus}
              her2={her2}
              setHer2={setHer2}
            />
          ) : (
            <>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", margin: "16px 0" }}>
                <div style={{ fontSize: 14.5, fontWeight: 700, fontFamily: "var(--hv-font-display)" }}>Изображение препарата</div>
                <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                  <span style={{ fontSize: 12.5, color: "var(--hv-text-muted)", fontWeight: 600 }}>Тепловая карта</span>
                  <span
                    onClick={() => setShowHeatmap((v) => !v)}
                    style={{
                      width: 38, height: 22, borderRadius: 20, background: "var(--hv-brand-dark)", padding: 2,
                      display: "flex", alignItems: "center", justifyContent: showHeatmap ? "flex-end" : "flex-start",
                    }}
                  >
                    <span style={{ width: 18, height: 18, borderRadius: "50%", background: "#fff" }} />
                  </span>
                </label>
              </div>
              <div style={{ position: "relative", borderRadius: 14, overflow: "hidden", aspectRatio: "4/3.1", background: "#eee" }}>
                {(showHeatmap ? heatmapUrl : sourceUrl) && (
                  <img
                    src={(showHeatmap ? heatmapUrl : sourceUrl)!}
                    alt="Препарат"
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  />
                )}
              </div>
            </>
          )}

          {error && <div style={{ marginTop: 14, fontSize: 13, color: "var(--hv-malignant)", fontWeight: 600 }}>{error}</div>}

          {!result && (
            <button
              onClick={handleAnalyze}
              disabled={!file || isSubmitting}
              style={{
                marginTop: 18, width: "100%", textAlign: "center", border: "none",
                cursor: file && !isSubmitting ? "pointer" : "default",
                padding: "14px 22px", borderRadius: 13, background: "var(--hv-brand-gradient)",
                color: "#fff", fontSize: 14.5, fontWeight: 700, fontFamily: "var(--hv-font-display)",
                opacity: file && !isSubmitting ? 1 : 0.55,
              }}
            >
              {isSubmitting ? "Анализируем…" : "Анализировать"}
            </button>
          )}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {result ? (
            <>
              <VerdictCard result={result} />
              <TopZonesCard regions={result.top_regions} />
              <button
                onClick={handleGenerateReport}
                style={{
                  textAlign: "center", border: "none", cursor: "pointer", padding: "16px 22px",
                  borderRadius: 13, background: "var(--hv-brand-gradient)", color: "#fff", fontSize: 14.5,
                  fontWeight: 700, fontFamily: "var(--hv-font-display)", display: "flex", alignItems: "center",
                  justifyContent: "center", gap: 8, boxShadow: "0 16px 32px -14px oklch(0.5 0.22 296 / 55%)",
                }}
              >
                Сформировать PDF-отчёт
              </button>
            </>
          ) : (
            <div style={{ background: "#fff", borderRadius: "var(--hv-radius-lg)", padding: 26, boxShadow: "var(--hv-shadow-card)", color: "var(--hv-text-muted)", fontSize: 13.5 }}>
              Загрузите изображение препарата и нажмите «Анализировать» — результат появится здесь.
            </div>
          )}
        </div>
      </div>

      <Disclaimer />
    </div>
  );
}

function Dot() {
  return <span style={{ width: 3, height: 3, borderRadius: "50%", background: "oklch(0.75 0.01 264)" }} />;
}

function UploadRow({ file, onPick, mode }: { file: File | null; onPick: () => void; mode: Mode }) {
  return (
    <div
      style={{
        display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, marginBottom: 18,
        padding: "12px 16px", border: "1.5px dashed oklch(0.8 0.02 264)", borderRadius: 12, background: "oklch(0.98 0.004 264)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
        <div style={{ width: 34, height: 34, borderRadius: 9, background: "var(--hv-brand-soft)", display: "flex", alignItems: "center", justifyContent: "center", flex: "none" }}>
          📄
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            {file ? file.name : mode === "patch" ? "Файл не выбран (PNG/JPEG)" : "Файл не выбран (SVS/TIFF/NDPI…)"}
          </div>
          {file && (
            <div style={{ fontSize: 11.5, color: "var(--hv-text-faint)" }}>{(file.size / 1e6).toFixed(1)} МБ</div>
          )}
        </div>
      </div>
      <div onClick={onPick} style={{ fontSize: 12.5, fontWeight: 700, color: "var(--hv-brand)", whiteSpace: "nowrap", cursor: "pointer" }}>
        {file ? "Заменить файл" : "Выбрать файл"}
      </div>
    </div>
  );
}

function IhcForm(props: {
  ki67: number;
  setKi67: (v: number) => void;
  erStatus: string;
  setErStatus: (v: string) => void;
  prStatus: string;
  setPrStatus: (v: string) => void;
  her2: string;
  setHer2: (v: string) => void;
}) {
  const statusOptions = [NOT_SPECIFIED, "Позитивный", "Негативный"];
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--hv-font-display)", margin: "4px 0 12px" }}>
        ИГХ-маркеры (опционально)
      </div>
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, color: "var(--hv-text-muted)", marginBottom: 6 }}>
          <span>Ki-67</span>
          <span>{props.ki67}%</span>
        </div>
        <input
          type="range" min={0} max={100} value={props.ki67}
          onChange={(e) => props.setKi67(Number(e.target.value))}
          style={{ width: "100%" }}
        />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Select label="ER-статус" value={props.erStatus} onChange={props.setErStatus} options={statusOptions} />
        <Select label="PR-статус" value={props.prStatus} onChange={props.setPrStatus} options={statusOptions} />
        <Select label="HER-2" value={props.her2} onChange={props.setHer2} options={[NOT_SPECIFIED, "0", "1+", "2+", "3+"]} />
      </div>
    </div>
  );
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <label>
      <div style={{ fontSize: 12, color: "var(--hv-text-muted)", marginBottom: 4 }}>{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: "100%", padding: "9px 10px", borderRadius: 8, border: "1.5px solid oklch(0.88 0.008 264)", fontSize: 13, background: "#fff" }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </label>
  );
}


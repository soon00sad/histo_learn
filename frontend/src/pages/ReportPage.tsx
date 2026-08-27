import { useEffect, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { api, ApiError, fetchAuthenticatedBlobUrl, parseUtc } from "../api/client";
import { SparkIcon } from "../components/SparkIcon";
import type { CaseDetail } from "../api/types";

export function ReportPage() {
  const { caseId } = useParams<{ caseId: string }>();
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    if (!caseId) return;
    api
      .getCase(caseId)
      .then(async (detail) => {
        setCaseDetail(detail);
        const [src, mask] = await Promise.all([
          fetchAuthenticatedBlobUrl(api.caseImageUrl(caseId)),
          fetchAuthenticatedBlobUrl(api.caseMaskUrl(caseId)),
        ]);
        setSourceUrl(src);
        setMaskUrl(mask);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось загрузить случай"));
  }, [caseId]);

  const handleDownload = async () => {
    if (!caseId) return;
    setIsDownloading(true);
    try {
      const blobUrl = await fetchAuthenticatedBlobUrl(api.caseReportUrl(caseId));
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `HistoVision_${caseId}.pdf`;
      link.click();
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сформировать отчёт");
    } finally {
      setIsDownloading(false);
    }
  };

  if (error) return <div style={{ padding: 48, color: "var(--hv-malignant)" }}>{error}</div>;
  if (!caseDetail) return <div style={{ padding: 48, color: "var(--hv-text-muted)" }}>Загрузка отчёта…</div>;

  return (
    <div style={{ minHeight: "100vh", background: "#f5f5f4", fontFamily: "var(--hv-font-body)", color: "var(--hv-text)", padding: "32px 24px" }}>
      <div style={{ maxWidth: 780, margin: "0 auto 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Link to={`/cases/${caseDetail.id}`} style={{ fontSize: 13, fontWeight: 700 }}>
          ← Назад к случаю
        </Link>
        <button
          onClick={handleDownload}
          disabled={isDownloading}
          style={{
            padding: "11px 20px", borderRadius: "var(--hv-radius-btn)", border: "none", cursor: isDownloading ? "default" : "pointer",
            background: "var(--hv-brand-gradient)", color: "#fff", fontWeight: 700, fontSize: 13.5,
            opacity: isDownloading ? 0.7 : 1,
            display: "flex", alignItems: "center", gap: 8,
          }}
        >
          {isDownloading ? "Формируем PDF…" : "Скачать PDF"}
          {!isDownloading && <SparkIcon size={10} />}
        </button>
      </div>

      {/* On-screen preview of the printed report layout (see src/report/pdf_report.py
          for the authoritative PDF version generated server-side). */}
      <div style={{ maxWidth: 780, margin: "0 auto", background: "#fff", borderRadius: 7, boxShadow: "0 2px 10px rgba(20,20,19,0.12)", padding: "0.55in 0.6in" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "2px solid var(--hv-text)", paddingBottom: 14 }}>
          <Logo height={26} />
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "var(--hv-font-display)" }}>Заключение по анализу препарата</div>
            <div style={{ fontSize: 11, color: "var(--hv-text-muted)" }}>
              Случай {caseDetail.id} · {parseUtc(caseDetail.created_at).toLocaleString("ru-RU")}
            </div>
            {caseDetail.mask_source !== "model" && (
              <div
                title="Демонстрация на эталонной маске BCSS (разметка патологов), не вывод обученной модели"
                style={{ marginTop: 3, fontSize: 9, fontWeight: 500, color: "oklch(0.65 0.01 264)" }}
              >
                эталонные данные
              </div>
            )}
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 22, marginTop: 18 }}>
          <div>
            <SectionLabel>Изображение препарата с маской сегментации</SectionLabel>
            <div style={{ position: "relative", borderRadius: 10, overflow: "hidden", aspectRatio: "4/3.3", background: "#eee", border: "1px solid var(--hv-border)" }}>
              {sourceUrl && (
                <img src={sourceUrl} alt="Препарат" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
              )}
              {maskUrl && (
                <img
                  src={maskUrl} alt="Маска сегментации"
                  style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", mixBlendMode: "multiply", opacity: 0.55 }}
                />
              )}
            </div>

            <div style={{ marginTop: 16 }}>
              <SectionLabel>Классы тканей</SectionLabel>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--hv-border)" }}>
                    <th style={thStyle}>Класс</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Доля площади</th>
                  </tr>
                </thead>
                <tbody>
                  {caseDetail.class_areas.map((c) => (
                    <tr key={c.name_en} style={{ borderBottom: "1px solid var(--hv-border-light)" }}>
                      <td style={{ padding: "6px 0", display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 9, height: 9, borderRadius: 3, background: c.color, flex: "none" }} />
                        {c.name_ru}
                      </td>
                      <td style={{ textAlign: "right", fontWeight: 700 }}>{(c.fraction * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <div style={{ background: "oklch(0.98 0.004 264)", border: "1px solid var(--hv-border)", borderRadius: 12, padding: "18px 20px" }}>
              <SectionLabel>Вердикт модели</SectionLabel>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "var(--hv-font-display)", color: caseDetail.is_malignant ? "var(--hv-malignant)" : "var(--hv-benign)" }}>
                {caseDetail.verdict_label}
              </div>
              <div style={{ fontSize: 12, color: "var(--hv-text-muted)", marginTop: 2 }}>
                доля опухолевой ткани <b>{(caseDetail.tumor_area_fraction * 100).toFixed(1)}%</b>
              </div>
            </div>

            <div style={{ marginTop: 14, fontSize: 10, lineHeight: 1.5, color: "var(--hv-text-muted)", borderLeft: "2px solid oklch(0.7 0.01 264)", paddingLeft: 10 }}>
              Система поддержки принятия решений. Окончательное заключение формулирует врач. Результат сформирован моделью HistoVision и подлежит проверке специалистом.
            </div>

            <div style={{ marginTop: 18 }}>
              <SectionLabel>Сведения о случае</SectionLabel>
              <InfoRow label="Ткань" value={caseDetail.tissue_type} />
              <InfoRow label="Файл препарата" value={caseDetail.source_filename} />
              <InfoRow label="Режим анализа" value={caseDetail.analysis_mode === "wsi" ? "Полный препарат" : "Живой анализ"} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--hv-text-muted)", marginBottom: 8 }}>
      {children}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, lineHeight: 1.9 }}>
      <span style={{ color: "var(--hv-text-muted)" }}>{label}</span>
      <span style={{ fontWeight: 600 }}>{value}</span>
    </div>
  );
}

const thStyle = { textAlign: "left" as const, padding: "5px 0", fontWeight: 700, color: "oklch(0.4 0.02 264)" };

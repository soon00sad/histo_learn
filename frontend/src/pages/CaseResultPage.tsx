import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { TopBar } from "../components/TopBar";
import { Disclaimer } from "../components/Disclaimer";
import { SegmentationViewer } from "../components/SegmentationViewer";
import { api, ApiError, fetchAuthenticatedBlobUrl, parseUtc } from "../api/client";
import type { CaseDetail } from "../api/types";

export function CaseResultPage() {
  const { caseId } = useParams<{ caseId: string }>();
  const navigate = useNavigate();

  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [showDisagreeForm, setShowDisagreeForm] = useState(false);
  const [disagreeComment, setDisagreeComment] = useState("");
  const [correctedVerdict, setCorrectedVerdict] = useState("");

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

  const handleAgree = async () => {
    if (!caseDetail) return;
    try {
      const updated = await api.reviewCase(caseDetail.id, { agreed: true });
      setCaseDetail({ ...caseDetail, status: updated.status });
      setShowDisagreeForm(false);
    } catch (err) {
      alert(err instanceof ApiError ? err.message : "Не удалось сохранить решение врача");
    }
  };

  const handleDisagreeSubmit = async () => {
    if (!caseDetail) return;
    try {
      const updated = await api.reviewCase(caseDetail.id, {
        agreed: false,
        comment: disagreeComment.trim() || undefined,
        corrected_verdict_label: correctedVerdict.trim() || undefined,
      });
      setCaseDetail({ ...caseDetail, status: updated.status });
      setShowDisagreeForm(false);
      setDisagreeComment("");
      setCorrectedVerdict("");
    } catch (err) {
      alert(err instanceof ApiError ? err.message : "Не удалось сохранить решение врача");
    }
  };

  if (error) {
    return (
      <div style={{ minHeight: "100vh", background: "var(--hv-bg)" }}>
        <TopBar active="none" />
        <div style={{ padding: 48, color: "var(--hv-malignant)" }}>{error}</div>
      </div>
    );
  }

  if (!caseDetail) {
    return (
      <div style={{ minHeight: "100vh", background: "var(--hv-bg)" }}>
        <TopBar active="none" />
        <div style={{ padding: 48, color: "var(--hv-text-muted)" }}>Загрузка случая…</div>
      </div>
    );
  }

  const isReviewed = caseDetail.status !== "pending";

  return (
    <div style={{ minHeight: "100vh", background: "var(--hv-bg)", fontFamily: "var(--hv-font-body)" }}>
      <TopBar active="none" />

      <div style={{ padding: "36px 48px 8px", display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 20 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ fontSize: 32, fontWeight: 600, letterSpacing: "-0.015em", lineHeight: 1.1, fontFamily: "var(--hv-font-display)" }}>
              Случай {caseDetail.id}
            </div>
            {caseDetail.mask_source !== "model" && (
              <span
                title="Демонстрация на эталонной маске BCSS (разметка патологов), не вывод обученной модели"
                style={{ padding: "3px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600, background: "oklch(0.95 0.01 264)", color: "oklch(0.6 0.02 264)" }}
              >
                эталон
              </span>
            )}
          </div>
          <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 14, fontSize: 13, color: "var(--hv-text-muted)" }}>
            <span>{parseUtc(caseDetail.created_at).toLocaleString("ru-RU")}</span>
            <span style={{ width: 3, height: 3, borderRadius: "50%", background: "oklch(0.75 0.01 264)" }} />
            <span>{caseDetail.tissue_type}</span>
            <span style={{ width: 3, height: 3, borderRadius: "50%", background: "oklch(0.75 0.01 264)" }} />
            <span>{caseDetail.analysis_mode === "wsi" ? "Полный препарат" : "Живой анализ"}</span>
          </div>
        </div>
        <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
          <div style={{ display: "flex", gap: 10 }}>
            {/* Review is a one-time decision (POST /cases/{id}/review 409s once
                the case has left "pending") — both buttons lock together as
                soon as either one has been used, not just the one clicked. */}
            <button
              onClick={handleAgree}
              disabled={isReviewed}
              style={{
                padding: "10px 18px", borderRadius: 10, border: "1.5px solid var(--hv-benign)",
                background: caseDetail.status === "confirmed" ? "var(--hv-benign-soft)" : "#fff",
                color: "var(--hv-benign)", fontWeight: 700, fontSize: 13.5,
                opacity: isReviewed && caseDetail.status !== "confirmed" ? 0.5 : 1,
                cursor: isReviewed ? "default" : "pointer",
              }}
            >
              {caseDetail.status === "confirmed" ? "Подтверждён врачом" : "Согласен"}
            </button>
            <button
              onClick={() => setShowDisagreeForm((v) => !v)}
              disabled={isReviewed}
              style={{
                padding: "10px 18px", borderRadius: 10, border: "1.5px solid var(--hv-malignant)",
                background: caseDetail.status === "rejected" ? "var(--hv-malignant-soft)" : "#fff",
                color: "var(--hv-malignant)", fontWeight: 700, fontSize: 13.5,
                opacity: isReviewed && caseDetail.status !== "rejected" ? 0.5 : 1,
                cursor: isReviewed ? "default" : "pointer",
              }}
            >
              {caseDetail.status === "rejected" ? "Отклонён врачом" : "Не согласен"}
            </button>
          </div>

          {showDisagreeForm && !isReviewed && (
            <div
              style={{
                position: "absolute", top: "calc(100% + 8px)", right: 0, zIndex: 20,
                width: 360, padding: 14, borderRadius: 12, background: "#fff",
                boxShadow: "0 16px 40px -12px oklch(0.2 0.05 268 / 35%)", border: "1px solid var(--hv-border)",
                display: "flex", flexDirection: "column", gap: 8,
              }}
            >
              <textarea
                value={disagreeComment}
                onChange={(e) => setDisagreeComment(e.target.value)}
                placeholder="Комментарий: в чём вы не согласны с заключением системы?"
                rows={3}
                style={{ resize: "vertical", padding: 8, borderRadius: 8, border: "1.5px solid var(--hv-border)", fontSize: 13, fontFamily: "inherit" }}
              />
              <input
                type="text"
                value={correctedVerdict}
                onChange={(e) => setCorrectedVerdict(e.target.value)}
                placeholder="Правильный вердикт (необязательно)"
                style={{ padding: 8, borderRadius: 8, border: "1.5px solid var(--hv-border)", fontSize: 13 }}
              />
              <button
                onClick={handleDisagreeSubmit}
                style={{
                  alignSelf: "flex-end", padding: "8px 16px", borderRadius: 9, border: "none",
                  background: "var(--hv-malignant)", color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer",
                }}
              >
                Отправить
              </button>
            </div>
          )}
        </div>
      </div>

      <div style={{ padding: "28px 48px 16px", display: "flex", flexDirection: "column", gap: 18, maxWidth: 900 }}>
        <SegmentationViewer
          tissueImageUrl={sourceUrl}
          maskImageUrl={maskUrl}
          classAreas={caseDetail.class_areas}
          verdictLabel={caseDetail.verdict_label}
          isMalignant={caseDetail.is_malignant}
          tumorAreaFraction={caseDetail.tumor_area_fraction}
        />
        <button
          onClick={() => navigate(`/cases/${caseDetail.id}/report`)}
          style={{
            alignSelf: "flex-start", textAlign: "center", border: "none", cursor: "pointer", padding: "14px 22px",
            borderRadius: "var(--hv-radius-btn)", background: "var(--hv-brand-gradient)", color: "#fff", fontSize: 14.5,
            fontWeight: 700, fontFamily: "var(--hv-font-display)", display: "flex", alignItems: "center",
            justifyContent: "center", gap: 8, boxShadow: "var(--hv-shadow-btn)",
          }}
        >
          {caseDetail.report_available ? "Открыть PDF-отчёт" : "Сформировать PDF-отчёт"}
        </button>
      </div>

      <Disclaimer />
    </div>
  );
}

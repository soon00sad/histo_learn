import { useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import type { ClassAreaOut } from "../api/types";

/**
 * Pixel-wise segmentation result viewer: pan/zoom slide area with a
 * mask overlay (opacity/visibility controls), minimap, tissue-class
 * legend, and a verdict/class-breakdown summary bar. Used by both
 * AnalysisPage (a just-finished analysis) and CaseResultPage (a saved
 * case) — same result shape either way (src/api/schemas.py's
 * AnalysisResult/CaseDetail both carry class_areas + tumor_area_fraction).
 *
 * Adapted from the original static /preview/analysis-v2 design mockup,
 * but driven entirely by real props: no fixed aspect ratio (measured from
 * the actual loaded image, since real slides/patches vary in size) and no
 * hardcoded class list (legend is built from whatever classes the model
 * actually found — classAreas only contains classes present in the mask).
 */

interface SegmentationViewerProps {
  tissueImageUrl: string | null;
  maskImageUrl: string | null;
  classAreas: ClassAreaOut[];
  verdictLabel: string;
  isMalignant: boolean;
  tumorAreaFraction: number;
}

function pct(fraction: number): string {
  return (fraction * 100).toFixed(1) + "%";
}

function clampNum(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function SegmentationViewer({
  tissueImageUrl, maskImageUrl, classAreas, verdictLabel, isMalignant, tumorAreaFraction,
}: SegmentationViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacityState] = useState(55);
  const [zoom, setZoomState] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [vw, setVw] = useState(0);
  const [vh, setVh] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [slideAr, setSlideAr] = useState(4 / 3);

  useEffect(() => {
    const measure = () => {
      const el = viewerRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setVw(r.width);
      setVh(r.height);
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Reset pan/zoom whenever a new image loads (different case/analysis).
  useEffect(() => {
    setZoomState(1);
    setTx(0);
    setTy(0);
  }, [tissueImageUrl]);

  const fit = () => {
    if (!vw || !vh) return { w: 0, h: 0 };
    return vw / vh > slideAr ? { w: vh * slideAr, h: vh } : { w: vw, h: vw / slideAr };
  };

  const clamp = (nextTx: number, nextTy: number, nextZoom: number) => {
    const f = fit();
    const maxX = Math.max(0, (f.w * nextZoom - vw) / 2);
    const maxY = Math.max(0, (f.h * nextZoom - vh) / 2);
    return { tx: clampNum(nextTx, -maxX, maxX), ty: clampNum(nextTy, -maxY, maxY) };
  };

  const setZoom = (z: number) => {
    const nextZoom = +clampNum(z, 1, 6).toFixed(2);
    const k = nextZoom / zoom;
    const c = clamp(tx * k, ty * k, nextZoom);
    setZoomState(nextZoom);
    setTx(c.tx);
    setTy(c.ty);
  };

  const f = fit();
  const dispW = f.w * zoom;
  const dispH = f.h * zoom;
  const fw = dispW ? Math.min(1, vw / dispW) : 1;
  const fh = dispH ? Math.min(1, vh / dispH) : 1;
  const cx = dispW ? 0.5 - tx / dispW : 0.5;
  const cy = dispH ? 0.5 - ty / dispH : 0.5;
  const left = clampNum(cx - fw / 2, 0, 1 - fw);
  const top = clampNum(cy - fh / 2, 0, 1 - fh);

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (zoom <= 1) return;
    dragRef.current = { x: e.clientX, y: e.clientY, tx, ty };
    setDragging(true);
  };
  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    const c = clamp(dragRef.current.tx + (e.clientX - dragRef.current.x), dragRef.current.ty + (e.clientY - dragRef.current.y), zoom);
    setTx(c.tx);
    setTy(c.ty);
  };
  const onUp = () => {
    if (dragRef.current) {
      dragRef.current = null;
      setDragging(false);
    }
  };
  // Scroll-to-zoom needs e.preventDefault() so the page doesn't scroll behind
  // the viewer while zooming — but React attaches onWheel as a passive
  // listener, where preventDefault() is silently ignored (logs a console
  // warning, does nothing). A native listener registered with
  // {passive: false} is the only way to actually block page scroll here.
  // The ref-to-latest-closure indirection lets the effect register the
  // listener once instead of re-attaching on every zoom/pan change.
  const onWheelRef = useRef<(e: WheelEvent) => void>(() => {});
  onWheelRef.current = (e: WheelEvent) => {
    e.preventDefault();
    setZoom(zoom + (e.deltaY < 0 ? 0.25 : -0.25));
  };

  useEffect(() => {
    const el = viewerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => onWheelRef.current(e);
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [tissueImageUrl]);

  const zoomLabel = zoom.toFixed(1).replace(/\.0$/, "") + "×";
  const verdictColor = isMalignant ? "var(--hv-malignant)" : "var(--hv-benign)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div
        style={{
          background: "oklch(0.17 0.04 268)", borderRadius: "var(--hv-radius-lg)",
          boxShadow: "var(--hv-shadow-card)", overflow: "hidden",
        }}
      >
        {/* controls bar */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "12px 18px", borderBottom: "1px solid oklch(1 0 0 / 8%)" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#fff" }}>Сегментация тканей</div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "6px 12px", borderRadius: 9, background: "oklch(1 0 0 / 6%)" }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "oklch(0.8 0.015 268)" }}>Маска</span>
              <div
                onClick={() => setShowMask((v) => !v)}
                style={{
                  width: 34, height: 19, borderRadius: 20, padding: 2, display: "flex", alignItems: "center", cursor: "pointer",
                  background: showMask ? "oklch(0.55 0.22 296)" : "oklch(0.4 0.02 268)", transition: "background .18s ease",
                }}
              >
                <div style={{ width: 15, height: 15, borderRadius: "50%", background: "#fff", transform: `translateX(${showMask ? 15 : 0}px)`, transition: "transform .18s ease" }} />
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "6px 12px", borderRadius: 9, background: "oklch(1 0 0 / 6%)" }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "oklch(0.8 0.015 268)" }}>Прозрачность</span>
              <input
                type="range" min={10} max={90} value={opacity}
                onChange={(e) => {
                  setOpacityState(Number(e.target.value));
                  setShowMask(true);
                }}
                style={{ width: 80 }}
              />
              <span style={{ fontSize: 11.5, fontWeight: 700, color: "oklch(0.72 0.14 296)", width: 30, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{opacity}%</span>
            </div>
          </div>
        </div>

        {/* viewer */}
        <div style={{ position: "relative", height: 420, background: "oklch(0.93 0.006 268)" }}>
          {!tissueImageUrl ? (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--hv-text-muted)", fontSize: 13 }}>
              Изображение недоступно
            </div>
          ) : (
            <div
              ref={viewerRef}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerLeave={onUp}
              style={{
                position: "absolute", inset: 0, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center",
                touchAction: "none", cursor: zoom > 1 ? (dragging ? "grabbing" : "grab") : "default",
              }}
            >
              <div
                style={{
                  position: "relative", width: "100%", height: "100%", flex: "none",
                  transform: `translate(${tx}px,${ty}px) scale(${zoom})`,
                  transition: dragging ? "none" : "transform .22s ease",
                }}
              >
                <img
                  src={tissueImageUrl} alt="Препарат" draggable={false}
                  onLoad={(e) => {
                    const { naturalWidth, naturalHeight } = e.currentTarget;
                    if (naturalWidth && naturalHeight) setSlideAr(naturalWidth / naturalHeight);
                  }}
                  style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", userSelect: "none", WebkitUserDrag: "none" } as CSSProperties}
                />
                {maskImageUrl && (
                  <img
                    src={maskImageUrl} alt="Сегментация тканей" draggable={false}
                    style={{
                      position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated",
                      mixBlendMode: "multiply", pointerEvents: "none", userSelect: "none",
                      opacity: showMask ? opacity / 100 : 0, transition: "opacity .18s ease",
                    }}
                  />
                )}
              </div>
            </div>
          )}

          {tissueImageUrl && (
            <>
              {/* zoom controls */}
              <div style={{ position: "absolute", right: 14, top: 14, display: "flex", flexDirection: "column", gap: 1, borderRadius: 10, overflow: "hidden", background: "oklch(0.88 0.008 268)", boxShadow: "0 8px 24px -10px oklch(0.2 0.05 268 / 40%)" }}>
                <div onClick={() => setZoom(zoom + 0.5)} style={{ width: 32, height: 32, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", position: "relative" }}>
                  <span style={{ position: "absolute", width: 12, height: 1.6, background: "oklch(0.3 0.03 268)" }} />
                  <span style={{ position: "absolute", width: 1.6, height: 12, background: "oklch(0.3 0.03 268)" }} />
                </div>
                <div onClick={() => setZoom(zoom - 0.5)} style={{ width: 32, height: 32, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>
                  <span style={{ width: 12, height: 1.6, background: "oklch(0.3 0.03 268)" }} />
                </div>
                <div
                  onClick={() => { setZoomState(1); setTx(0); setTy(0); }}
                  style={{ width: 32, height: 26, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: 10, fontWeight: 700, color: "oklch(0.45 0.02 268)" }}
                >
                  {zoomLabel}
                </div>
              </div>

              {/* minimap */}
              <div style={{ position: "absolute", right: 14, bottom: 14, width: 100, borderRadius: 8, overflow: "hidden", background: "#fff", padding: 4, boxShadow: "0 8px 24px -10px oklch(0.2 0.05 268 / 40%)" }}>
                <div style={{ position: "relative" }}>
                  <img src={tissueImageUrl} alt="" style={{ display: "block", width: "100%", height: "auto", borderRadius: 4 }} />
                  <div
                    style={{
                      position: "absolute", left: `${(left * 100).toFixed(1)}%`, top: `${(top * 100).toFixed(1)}%`,
                      width: `${(fw * 100).toFixed(1)}%`, height: `${(fh * 100).toFixed(1)}%`,
                      border: "1.5px solid rgb(150,90,240)", background: "rgba(150,90,240,0.14)", borderRadius: 3, boxSizing: "border-box",
                    }}
                  />
                </div>
              </div>

              {/* legend — only classes the model actually found */}
              {classAreas.length > 0 && (
                <div style={{ position: "absolute", left: 14, top: 14, background: "oklch(1 0 0 / 92%)", backdropFilter: "blur(8px)", borderRadius: 10, padding: "11px 13px", boxShadow: "0 8px 28px -12px oklch(0.2 0.05 268 / 40%)", maxWidth: 200 }}>
                  <div style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", color: "oklch(0.5 0.02 268)", marginBottom: 8 }}>Классы тканей</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {classAreas.map((c) => (
                      <div key={c.name_en} style={{ display: "flex", alignItems: "center", gap: 8, whiteSpace: "nowrap" }}>
                        <span style={{ width: 10, height: 10, borderRadius: 3, flex: "none", background: c.color }} />
                        <span style={{ fontSize: 12, fontWeight: 600, color: "oklch(0.25 0.03 268)", overflow: "hidden", textOverflow: "ellipsis" }}>{c.name_ru}</span>
                        <span style={{ fontSize: 11.5, color: "oklch(0.5 0.02 268)", marginLeft: "auto", paddingLeft: 10, fontVariantNumeric: "tabular-nums" }}>{pct(c.fraction)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* summary bar */}
        <div style={{ padding: "14px 18px", borderTop: "1px solid oklch(1 0 0 / 8%)" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: classAreas.length ? 10 : 0 }}>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "oklch(0.6 0.02 268)" }}>Вердикт</span>
            <span style={{ fontSize: 16, fontWeight: 700, fontFamily: "var(--hv-font-display)", color: verdictColor }}>
              {verdictLabel}
              <span style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.7 0.02 268)", marginLeft: 8, fontVariantNumeric: "tabular-nums" }}>
                · доля опухолевой ткани {pct(tumorAreaFraction)}
              </span>
            </span>
          </div>
          {classAreas.length > 0 && (
            <>
              <div style={{ display: "flex", height: 7, borderRadius: 4, overflow: "hidden" }}>
                {classAreas.map((c) => (
                  <div key={c.name_en} style={{ width: pct(c.fraction), background: c.color }} />
                ))}
              </div>
              <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 8 }}>
                {classAreas.map((c) => (
                  <span key={c.name_en} style={{ fontSize: 11, color: "oklch(0.72 0.02 268)", whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" }}>
                    {c.name_ru} <b style={{ color: "#fff", fontWeight: 700 }}>{pct(c.fraction)}</b>
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

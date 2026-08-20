import { useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from "react";
import { Link } from "react-router-dom";

/**
 * Static design preview of the "Анализ препарата v2" segmentation viewer
 * (imported from the Claude Design project). Not wired to the API — the
 * backend currently produces only a binary verdict + GradCAM++ regions,
 * no tissue-class segmentation mask. Case id, verdict, and class shares
 * below are fixed sample data from the mockup, same as tissue.jpg/mask.png
 * (generated placeholders standing in for real slide imagery).
 */

const SLIDE_AR = 1280 / 960;

interface TissueClass {
  name: string;
  color: string;
  share: number;
}

const CLASSES: TissueClass[] = [
  { name: "Опухоль", color: "rgb(214,58,44)", share: 0.21 },
  { name: "Строма", color: "rgb(62,116,214)", share: 0.52 },
  { name: "Лимфоциты", color: "rgb(36,168,132)", share: 0.16 },
  { name: "Доброкачественная", color: "rgb(136,96,206)", share: 0.09 },
  { name: "Некроз", color: "rgb(196,142,32)", share: 0.02 },
];

function pct(share: number): string {
  return Math.round(share * 100) + "%";
}

function clampNum(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function AnalysisV2Preview() {
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

  const fit = () => {
    if (!vw || !vh) return { w: 0, h: 0 };
    return vw / vh > SLIDE_AR ? { w: vh * SLIDE_AR, h: vh } : { w: vw, h: vw / SLIDE_AR };
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
  const onWheel = (e: ReactWheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    setZoom(zoom + (e.deltaY < 0 ? 0.25 : -0.25));
  };

  const zoomLabel = zoom.toFixed(1).replace(/\.0$/, "") + "×";

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "oklch(0.17 0.04 268)", fontFamily: "Manrope, system-ui, sans-serif", color: "oklch(0.95 0.01 268)", overflow: "hidden" }}>
      {/* thin top bar */}
      <div style={{ flex: "none", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 24, padding: "12px 24px", borderBottom: "1px solid oklch(1 0 0 / 8%)", background: "oklch(0.2 0.042 268)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 22, minWidth: 0 }}>
          <Link to="/analysis">
            <img src="/logo.png" alt="HistoVision" style={{ height: 20, width: "auto", display: "block", filter: "brightness(0) invert(1)", opacity: 0.92 }} />
          </Link>
          <div style={{ width: 1, height: 22, background: "oklch(1 0 0 / 10%)", flex: "none" }} />
          <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0, fontVariantNumeric: "tabular-nums" }}>
            <span style={{ fontSize: 13, fontWeight: 700, whiteSpace: "nowrap" }}>HV-2026-04831</span>
            <span style={{ fontSize: 12, color: "oklch(0.65 0.02 268)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>биопсия молочной железы · H&amp;E · 40×</span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 14, flex: "none" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "7px 12px", borderRadius: 9, background: "oklch(1 0 0 / 6%)" }}>
            <span style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.8 0.015 268)", whiteSpace: "nowrap" }}>Сегментация</span>
            <div
              onClick={() => setShowMask((v) => !v)}
              style={{
                width: 36, height: 20, borderRadius: 20, padding: 2, display: "flex", alignItems: "center", cursor: "pointer",
                background: showMask ? "oklch(0.55 0.22 296)" : "oklch(0.4 0.02 268)", transition: "background .18s ease",
              }}
            >
              <div style={{ width: 16, height: 16, borderRadius: "50%", background: "#fff", transform: `translateX(${showMask ? 16 : 0}px)`, transition: "transform .18s ease" }} />
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 14px", borderRadius: 9, background: "oklch(1 0 0 / 6%)" }}>
            <span style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.8 0.015 268)", whiteSpace: "nowrap" }}>Прозрачность</span>
            <input
              type="range" min={10} max={90} value={opacity}
              onChange={(e) => {
                setOpacityState(Number(e.target.value));
                setShowMask(true);
              }}
              style={rangeStyle}
            />
            <span style={{ fontSize: 12, fontWeight: 700, color: "oklch(0.72 0.14 296)", width: 34, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{opacity}%</span>
          </div>
          <button
            style={{
              border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 8, padding: "9px 16px", borderRadius: 10,
              background: "linear-gradient(135deg, oklch(0.52 0.22 296), oklch(0.44 0.2 288))", color: "#fff", fontSize: 13, fontWeight: 700,
              fontFamily: "Sora, sans-serif", whiteSpace: "nowrap", boxShadow: "0 10px 22px -10px oklch(0.52 0.22 296 / 60%)",
            }}
          >
            <span style={{ display: "inline-block", width: 11, height: 9, border: "1.6px solid #fff", borderRadius: 2, position: "relative" }}>
              <span style={{ position: "absolute", left: "50%", top: -6, transform: "translateX(-50%)", width: 0, height: 0, borderLeft: "3.5px solid transparent", borderRight: "3.5px solid transparent", borderBottom: "4.5px solid #fff" }} />
            </span>
            Загрузить препарат
          </button>
        </div>
      </div>

      {/* viewer */}
      <div style={{ flex: 1, minHeight: 0, position: "relative", background: "oklch(0.93 0.006 268)", overflow: "hidden" }}>
        <div
          ref={viewerRef}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={onUp}
          onWheel={onWheel}
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
            <img src="/design-preview/slide-tissue.jpg" alt="Гистологический препарат" draggable={false} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", userSelect: "none", WebkitUserDrag: "none" } as CSSProperties} />
            <img
              src="/design-preview/slide-mask.png" alt="Сегментация тканей" draggable={false}
              style={{
                position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", imageRendering: "pixelated",
                mixBlendMode: "multiply", pointerEvents: "none", userSelect: "none",
                opacity: showMask ? opacity / 100 : 0, transition: "opacity .18s ease",
              }}
            />
          </div>
        </div>

        {/* zoom controls */}
        <div style={{ position: "absolute", right: 18, top: 18, display: "flex", flexDirection: "column", gap: 1, borderRadius: 11, overflow: "hidden", background: "oklch(0.88 0.008 268)", boxShadow: "0 8px 24px -10px oklch(0.2 0.05 268 / 40%)" }}>
          <div onClick={() => setZoom(zoom + 0.5)} style={{ width: 38, height: 38, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", position: "relative" }}>
            <span style={{ position: "absolute", width: 14, height: 1.8, background: "oklch(0.3 0.03 268)" }} />
            <span style={{ position: "absolute", width: 1.8, height: 14, background: "oklch(0.3 0.03 268)" }} />
          </div>
          <div onClick={() => setZoom(zoom - 0.5)} style={{ width: 38, height: 38, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}>
            <span style={{ width: 14, height: 1.8, background: "oklch(0.3 0.03 268)" }} />
          </div>
          <div
            onClick={() => {
              setZoomState(1);
              setTx(0);
              setTy(0);
            }}
            style={{ width: 38, height: 30, background: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: 10.5, fontWeight: 700, color: "oklch(0.45 0.02 268)", fontVariantNumeric: "tabular-nums" }}
          >
            {zoomLabel}
          </div>
        </div>

        {/* minimap */}
        <div style={{ position: "absolute", right: 18, bottom: 18, width: 120, borderRadius: 9, overflow: "hidden", background: "#fff", padding: 4, boxShadow: "0 8px 24px -10px oklch(0.2 0.05 268 / 40%)" }}>
          <div style={{ position: "relative" }}>
            <img src="/design-preview/slide-tissue.jpg" alt="" style={{ display: "block", width: "100%", height: "auto", borderRadius: 5 }} />
            <div
              style={{
                position: "absolute", left: `${(left * 100).toFixed(1)}%`, top: `${(top * 100).toFixed(1)}%`,
                width: `${(fw * 100).toFixed(1)}%`, height: `${(fh * 100).toFixed(1)}%`,
                border: "1.5px solid rgb(150,90,240)", background: "rgba(150,90,240,0.14)", borderRadius: 3, boxSizing: "border-box",
              }}
            />
          </div>
        </div>

        {/* legend */}
        <div style={{ position: "absolute", left: 18, top: 18, background: "oklch(1 0 0 / 92%)", backdropFilter: "blur(8px)", borderRadius: 12, padding: "13px 15px", boxShadow: "0 8px 28px -12px oklch(0.2 0.05 268 / 40%)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", color: "oklch(0.5 0.02 268)", marginBottom: 9 }}>Классы тканей</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {CLASSES.map((c) => (
              <div key={c.name} style={{ display: "flex", alignItems: "center", gap: 9, whiteSpace: "nowrap" }}>
                <span style={{ width: 11, height: 11, borderRadius: 3, flex: "none", background: c.color }} />
                <span style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.25 0.03 268)" }}>{c.name}</span>
                <span style={{ fontSize: 12, color: "oklch(0.5 0.02 268)", marginLeft: "auto", paddingLeft: 14, fontVariantNumeric: "tabular-nums" }}>{pct(c.share)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* scale bar */}
        <div style={{ position: "absolute", left: 18, bottom: 18, display: "flex", flexDirection: "column", gap: 4, background: "oklch(1 0 0 / 90%)", padding: "7px 10px", borderRadius: 8 }}>
          <div style={{ width: 88, height: 3, background: "oklch(0.3 0.03 268)" }} />
          <div style={{ fontSize: 10, color: "oklch(0.4 0.02 268)", fontVariantNumeric: "tabular-nums" }}>250 мкм</div>
        </div>
      </div>

      {/* summary bar */}
      <div style={{ flex: "none", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 28, padding: "14px 24px 10px", background: "oklch(0.2 0.042 268)", borderTop: "1px solid oklch(1 0 0 / 8%)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 26, minWidth: 0 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, flex: "none" }}>
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", color: "oklch(0.6 0.02 268)" }}>Вердикт</span>
            <span style={{ fontSize: 17, fontWeight: 700, fontFamily: "Sora, sans-serif", color: "oklch(0.68 0.19 27)", whiteSpace: "nowrap" }}>
              Злокачественная <span style={{ fontSize: 13, fontWeight: 600, color: "oklch(0.7 0.02 268)", fontVariantNumeric: "tabular-nums" }}>· 94.2%</span>
            </span>
          </div>

          <div style={{ width: 1, height: 38, background: "oklch(1 0 0 / 10%)", flex: "none" }} />

          <div style={{ display: "flex", flexDirection: "column", gap: 7, minWidth: 0, flex: 1 }}>
            <div style={{ display: "flex", height: 8, borderRadius: 5, overflow: "hidden", minWidth: 280 }}>
              {CLASSES.map((c) => (
                <div key={c.name} style={{ width: pct(c.share), background: c.color }} />
              ))}
            </div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              {CLASSES.map((c) => (
                <span key={c.name} style={{ fontSize: 11.5, color: "oklch(0.72 0.02 268)", whiteSpace: "nowrap", fontVariantNumeric: "tabular-nums" }}>
                  {c.name} <b style={{ color: "#fff", fontWeight: 700 }}>{pct(c.share)}</b>
                </span>
              ))}
            </div>
          </div>
        </div>

        <div style={{ flex: "none", padding: "12px 20px", borderRadius: 11, background: "oklch(1 0 0 / 8%)", border: "1px solid oklch(1 0 0 / 14%)", color: "#fff", fontSize: 13.5, fontWeight: 700, fontFamily: "Sora, sans-serif", whiteSpace: "nowrap" }}>
          Сформировать отчёт
        </div>
      </div>

      <div style={{ flex: "none", padding: "0 24px 11px", background: "oklch(0.2 0.042 268)", fontSize: 10.5, color: "oklch(0.58 0.015 268)" }}>
        Система поддержки принятия решений. Окончательное заключение формулирует врач. · Модель v2.3.1 · демо-прототип, не подключён к анализу
      </div>
    </div>
  );
}

const rangeStyle: CSSProperties = { width: 96 };

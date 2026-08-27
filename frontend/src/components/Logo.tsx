import { useId } from "react";

/**
 * Brand mark, rendered as inline SVG + live text instead of the old
 * /public/logo.png raster file — that file was corrupted (truncated
 * mid-save, present since the very first commit) and showed as visibly
 * cropped at any display size larger than a small nav icon. An SVG mark
 * has no such failure mode and stays crisp at every size this app uses it.
 */
interface LogoProps {
  height?: number;
  variant?: "brand" | "white";
  showWordmark?: boolean;
}

export function Logo({ height = 26, variant = "brand", showWordmark = true }: LogoProps) {
  const gradientId = `hv-logo-gradient-${useId()}`;
  const color = variant === "white" ? "#ffffff" : `url(#${gradientId})`;
  const textColor = variant === "white" ? "#ffffff" : "oklch(0.4 0.2 296)";
  const opacity = variant === "white" ? 0.94 : 1;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: height * 0.32, opacity }}>
      <svg width={height} height={height} viewBox="0 0 40 40" style={{ flex: "none" }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#8B5CF6" />
            <stop offset="100%" stopColor="#5B21B6" />
          </linearGradient>
        </defs>
        <g fill={color}>
          {[0, 60, 120, 180, 240, 300].map((angle) => (
            <ellipse key={angle} cx="20" cy="11" rx="4.5" ry="10" transform={`rotate(${angle} 20 20)`} />
          ))}
        </g>
        <circle cx="20" cy="20" r="3.5" fill={variant === "white" ? "#ffffff" : "#F5F3FF"} />
      </svg>
      {showWordmark && (
        <span
          style={{
            fontFamily: "var(--hv-font-display)",
            fontWeight: 800,
            fontSize: height * 0.62,
            letterSpacing: "0.02em",
            color: textColor,
            whiteSpace: "nowrap",
          }}
        >
          HISTO VISION
        </span>
      )}
    </div>
  );
}

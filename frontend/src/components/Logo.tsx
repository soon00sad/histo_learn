/**
 * Brand mark — /public/logo.png, the real HistoVision logo. The original
 * file (in this exact path, since the repo's first commit) was truncated
 * mid-save and rendered visibly cropped at any size above a small nav
 * icon; the flattened export the corrupted file came from also had no
 * alpha channel. Both are fixed here: re-extracted from the source
 * artwork (Downloads/Histo Vision logo_page-0001.png) with a proper
 * alpha channel recovered from its white background, so `variant="white"`
 * (used on LoginPage's dark background) can still just invert it.
 */
interface LogoProps {
  height?: number;
  variant?: "brand" | "white";
}

export function Logo({ height = 26, variant = "brand" }: LogoProps) {
  return (
    <img
      src="/logo.png"
      alt="HistoVision"
      style={{
        height,
        width: "auto",
        display: "block",
        filter: variant === "white" ? "brightness(0) invert(1)" : undefined,
        opacity: variant === "white" ? 0.94 : 1,
      }}
    />
  );
}

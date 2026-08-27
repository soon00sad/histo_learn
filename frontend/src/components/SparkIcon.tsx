/** Small four-point sparkle glyph, echoing the little accent icon next to
 * links/buttons in the thinkbio.ai reference — drawn as our own SVG, not
 * copied from theirs. Used next to primary-action button labels. */
export function SparkIcon({ size = 10, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 0C12.6 6.4 13.4 8.4 24 12C13.4 15.6 12.6 17.6 12 24C11.4 17.6 10.6 15.6 0 12C10.6 8.4 11.4 6.4 12 0Z"
        fill={color}
      />
    </svg>
  );
}

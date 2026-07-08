
interface TokenExplainedProps {
  token: string;
  // Drives the bar color/intensity (scaled for contrast).
  score: number;
  // Share of total attribution, shown as the number in the bar (signed proportion).
  percentage: number;
}

/**
 * Background color for a score bar: green for positive, red for negative,
 * with opacity scaled by magnitude. Mirrors the text-generation view so both
 * tasks read the same way.
 */
function getScoreColor(score: number) {
  if (Math.abs(score) < 0.001) return "transparent";

  const intensity = Math.min(Math.abs(score) * 2.5, 1);

  if (score >= 0) {
    return `rgba(34, 197, 94, ${intensity})`; // Green-500
  } else {
    return `rgba(239, 68, 68, ${intensity})`; // Red-500
  }
}

function formatPercent(val: number) {
  if (val === undefined || val === null) return "-%";
  return (val * 100).toFixed(0) + "%";
}

export default function TokenExplained({ token, score, percentage }: TokenExplainedProps) {

  const parsed_token = token.replace('Ġ', ' ').replace('##', '');

  return (
    <div
      className="flex max-w-full flex-col items-center justify-between min-w-12.5 min-h-12.5 overflow-hidden border border-border bg-sunken rounded"
      title={`Importance: ${formatPercent(percentage)} of total`}
    >
      <span className="max-w-72 break-all px-2 py-1 text-center text-sm font-mono font-bold leading-snug text-fg">
        {parsed_token}
      </span>

      <div
        className="w-full text-[10px] font-bold text-fg text-center py-0.5 border-t border-border transition-colors duration-300"
        style={{ backgroundColor: getScoreColor(score) }}
      >
        {formatPercent(percentage)}
      </div>
    </div>
  );
}

import { useState, useRef, useEffect } from "react";
import type { TutorialFocusTarget } from "../../lib/tutorialGuide";

// --- HELPERS ---

function getScoreColor(score: number) {
  if (score < 0.01) return "transparent";
  const intensity = Math.min(score * 2.5, 1);
  return `rgba(34, 197, 94, ${intensity})`;
}

function formatPercent(val: number) {
  if (val === undefined || val === null) return "-";
  return (val * 100).toFixed(0) + "%";
}

// Jet-like colormap: t in [0,1] -> [r, g, b] (0-255). Matches the look of the
// backend's matplotlib 'jet' overlays so single-token maps render consistently.
function jet(t: number): [number, number, number] {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const r = clamp(1.5 - Math.abs(4 * t - 3));
  const g = clamp(1.5 - Math.abs(4 * t - 2));
  const b = clamp(1.5 - Math.abs(4 * t - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// --- INTERFACES ---

interface HeatmapData {
  image_base64: string;
  raw_matrix: number[][];
}

export type ImageGenerationTutorialSelection = {
  selectedTokenIndices?: number[];
  hoveredCell?: { x: number; y: number } | null;
};

export type ImageGenerationSelection = {
  selectedTokenIndices: number[];
  hoveredCell: { x: number; y: number } | null;
};

interface ImageGenViewProps {
  baseImage: string;
  tokens: string[];
  heatmaps: HeatmapData[];
  selection?: ImageGenerationSelection | ImageGenerationTutorialSelection | null;
  onSelectionChange?: (selection: ImageGenerationSelection) => void;
  tutorialFocusTarget?: TutorialFocusTarget;
}

const GRID = 64;
const OVERLAY_ALPHA = 0.6;

export default function ImageGenView({ baseImage, tokens, heatmaps, selection, onSelectionChange, tutorialFocusTarget }: ImageGenViewProps) {
  // Token selection (multiple tokens => aggregated heatmap)
  const [internalSelection, setInternalSelection] = useState<ImageGenerationSelection>({
    selectedTokenIndices: [],
    hoveredCell: null,
  });

  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const isControlled = selection !== undefined;
  const activeSelection = isControlled
    ? {
      selectedTokenIndices: selection?.selectedTokenIndices ?? [],
      hoveredCell: selection?.hoveredCell ?? null,
    }
    : internalSelection;
  const selectedTokensForRender = activeSelection.selectedTokenIndices;
  const hoveredCellForRender = activeSelection.hoveredCell;
  const hasSelection = selectedTokensForRender.length > 0;

  const getTutorialFocusClass = (target: TutorialFocusTarget) => (
    tutorialFocusTarget === target ? " tutorial-inner-highlight" : ""
  );

  // Render the (aggregated) heatmap onto the overlay canvas. Recomputes only when
  // the selection or the heatmaps change -- not on hover.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const valid = selectedTokensForRender.filter((i) => heatmaps[i]?.raw_matrix);
    if (valid.length === 0) {
      ctx.clearRect(0, 0, GRID, GRID);
      return;
    }

    // Mean of the selected tokens' raw matrices.
    const agg = new Array(GRID * GRID).fill(0);
    for (const i of valid) {
      const m = heatmaps[i].raw_matrix;
      for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) agg[y * GRID + x] += m[y][x];
      }
    }
    for (let k = 0; k < agg.length; k++) agg[k] /= valid.length;

    // Robust 1st-99th percentile normalization (matches the backend overlays).
    const sorted = [...agg].sort((a, b) => a - b);
    const pct = (q: number) =>
      sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
    const vmin = pct(0.01);
    const vmax = pct(0.99);
    const denom = vmax - vmin || 1e-8;

    const img = ctx.createImageData(GRID, GRID);
    for (let k = 0; k < agg.length; k++) {
      const t = Math.max(0, Math.min(1, (agg[k] - vmin) / denom));
      const [r, g, b] = jet(t);
      img.data[k * 4] = r;
      img.data[k * 4 + 1] = g;
      img.data[k * 4 + 2] = b;
      img.data[k * 4 + 3] = Math.round(OVERLAY_ALPHA * 255);
    }
    ctx.putImageData(img, 0, 0);
  }, [selectedTokensForRender, heatmaps]);

  // --- SELECTION ---
  const updateSelection = (nextSelection: ImageGenerationSelection) => {
    if (isControlled) {
      onSelectionChange?.(nextSelection);
      return;
    }
    setInternalSelection(nextSelection);
  };

  const handleTokenClick = (e: React.MouseEvent, idx: number) => {
    // Ctrl/Cmd (and Shift, as a fallback since some Linux WMs grab the Super key)
    // add the token to the aggregate instead of replacing the selection.
    const additive = e.metaKey || e.ctrlKey || e.shiftKey;
    if (isControlled && !onSelectionChange) return;

    const nextSelectedTokens = (() => {
      const prev = selectedTokensForRender;
      if (additive) {
        // Ctrl/Cmd-click toggles this token in/out of the aggregate.
        return prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx];
      }
      // Plain click selects only this token (or clears if it was the only one).
      if (prev.length === 1 && prev[0] === idx) return [];
      return [idx];
    })();

    updateSelection({
      selectedTokenIndices: nextSelectedTokens,
      hoveredCell: hoveredCellForRender,
    });
  };

  // --- MOUSE (HOVER) ---
  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current) return;

    const rect = imgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, x / rect.width));
    const normY = Math.max(0, Math.min(1, y / rect.height));

    const gridX = Math.floor(normX * GRID);
    const gridY = Math.floor(normY * GRID);

    if (isControlled && !onSelectionChange) return;

    updateSelection({
      selectedTokenIndices: selectedTokensForRender,
      hoveredCell: { x: Math.min(gridX, GRID - 1), y: Math.min(gridY, GRID - 1) },
    });
  };

  const handleMouseLeaveImage = () => {
    if (isControlled && !onSelectionChange) return;

    updateSelection({
      selectedTokenIndices: selectedTokensForRender,
      hoveredCell: null,
    });
  };

  const selectedLabel = selectedTokensForRender.map((i) => tokens[i]).filter(Boolean).join(" + ");

  return (
    <div className="flex min-w-0 flex-col gap-8 w-full select-none">

      {/* INTERACTIVE IMAGE */}
      <div className={`flex flex-col items-center bg-sunken p-4 rounded-lg border border-border${getTutorialFocusClass("output-image")}`}>
        <div className="flex w-full max-w-lg flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-fg-subtle text-sm uppercase font-bold">
            Generated Image
          </h3>
          <span className="text-xs text-fg-faint italic sm:text-right">
            {hasSelection
              ? `Heatmap: ${selectedLabel}${selectedTokensForRender.length > 1 ? " (aggregated)" : ""}`
              : hoveredCellForRender
                ? "Inspecting Pixel Attention"
                : "Hover image to inspect pixels"}
          </span>
        </div>

        <div className="relative inline-block overflow-hidden rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            <img
            ref={imgRef}
            src={`data:image/png;base64,${baseImage}`}
            alt="Generated"
            className="max-w-lg w-full h-auto cursor-crosshair"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeaveImage}
            />
            {/* Aggregated heatmap overlay (drawn from the selected tokens' raw matrices) */}
            <canvas
              ref={canvasRef}
              width={GRID}
              height={GRID}
              className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-200"
              style={{ opacity: hasSelection ? 1 : 0 }}
            />
            {hoveredCellForRender && !hasSelection && (
                <div
                    className="absolute border border-white/50 bg-white/20 pointer-events-none"
                    style={{
                        left: `${(hoveredCellForRender.x / GRID) * 100}%`,
                        top: `${(hoveredCellForRender.y / GRID) * 100}%`,
                        width: `${100 / GRID}%`,
                        height: `${100 / GRID}%`
                    }}
                />
            )}
        </div>
      </div>

      {/* TOKEN INPUT */}
      <div className={`bg-sunken p-6 rounded-lg border border-border${getTutorialFocusClass("output-image-prompt")}`}>
        <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
            <h3 className="text-fg-subtle text-sm uppercase font-bold">
            Input Prompt
            </h3>
            <span className="text-xs text-fg-faint italic sm:text-right">
            Click a token &middot; Shift / Ctrl / Cmd-click to aggregate multiple
            </span>
        </div>

        <div className="flex min-w-0 flex-wrap gap-2">
          {tokens.map((token, idx) => {
            let score = 0;
            let color = "transparent";

            if (hoveredCellForRender && heatmaps[idx]?.raw_matrix) {
              score = heatmaps[idx].raw_matrix[hoveredCellForRender.y][hoveredCellForRender.x];
              color = getScoreColor(score);
            }

            const isSelected = selectedTokensForRender.includes(idx);

            return (
              <div
                key={idx}
                onClick={(e) => handleTokenClick(e, idx)}
                className={`
                  flex max-w-full flex-col items-center justify-between overflow-hidden
                  min-w-15 min-h-12.5
                  border rounded transition-all duration-150 cursor-pointer
                  ${isSelected
                    ? "border-accent shadow-[0_0_15px_rgba(168,85,247,0.4)] bg-fill-strong"
                    : "border-border bg-sunken hover:border-border-strong"}
                `}
              >
                <span className="max-w-72 break-all px-2 py-2 text-center text-sm font-mono font-bold leading-snug text-fg">
                  {token}
                </span>
                <div
                  className="w-full text-[10px] text-center font-bold text-fg py-0.5 border-t border-border transition-colors duration-75"
                  style={{ backgroundColor: color }}
                >
                  {hoveredCellForRender ? formatPercent(score) : "-"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}

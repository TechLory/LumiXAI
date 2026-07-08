import { useEffect, useRef, useState } from "react";

// Jet-like colormap: t in [0,1] -> [r, g, b] (0-255). Matches the backend's matplotlib
// 'jet' overlays and the one used by ImageGenView, so heatmaps look consistent across views.
function jet(t: number): [number, number, number] {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const r = clamp(1.5 - Math.abs(4 * t - 3));
  const g = clamp(1.5 - Math.abs(4 * t - 2));
  const b = clamp(1.5 - Math.abs(4 * t - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

import type { TutorialFocusTarget } from "../../lib/tutorialGuide";

interface HeatmapData {
  image_base64: string;
  raw_matrix: number[][];
}

interface ImageClassificationViewProps {
  baseImage: string;
  heatmap: HeatmapData;
  predictedLabel: string;
  tutorialFocusTarget?: TutorialFocusTarget;
}

const GRID = 64;

export default function ImageClassificationView({ baseImage, heatmap, predictedLabel, tutorialFocusTarget }: ImageClassificationViewProps) {
  const getTutorialFocusClass = (target: TutorialFocusTarget) => (
    tutorialFocusTarget === target ? " tutorial-inner-highlight" : ""
  );
  const [showOverlay, setShowOverlay] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !heatmap?.raw_matrix) return;

    const matrix = heatmap.raw_matrix;
    const flat = matrix.flat();

    // Robust 1st-99th percentile normalization (matches the backend overlays).
    const sorted = [...flat].sort((a, b) => a - b);
    const pct = (q: number) =>
      sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
    const vmin = pct(0.01);
    const vmax = pct(0.99);
    const denom = vmax - vmin || 1e-8;

    const img = ctx.createImageData(GRID, GRID);
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        const k = y * GRID + x;
        const t = Math.max(0, Math.min(1, (matrix[y][x] - vmin) / denom));
        const [r, g, b] = jet(t);
        img.data[k * 4] = r;
        img.data[k * 4 + 1] = g;
        img.data[k * 4 + 2] = b;
        img.data[k * 4 + 3] = Math.round(0.6 * 255);
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [heatmap]);

  return (
    <div className="flex min-w-0 flex-col gap-6 w-full select-none">
      <div className={`flex flex-col items-center bg-sunken p-4 rounded-lg border border-border${getTutorialFocusClass("output-image")}`}>
        <div className="flex w-full max-w-lg flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-fg-subtle text-sm uppercase font-bold">Input Image</h3>
          <button
            type="button"
            onClick={() => setShowOverlay((prev) => !prev)}
            className="text-xs text-fg-faint italic underline underline-offset-4 hover:text-fg transition-colors cursor-pointer"
          >
            {showOverlay ? "Hide" : "Show"} attribution overlay
          </button>
        </div>

        <div className="relative inline-block overflow-hidden rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.5)]">
          <img
            src={`data:image/png;base64,${baseImage}`}
            alt="Input"
            className="max-w-lg w-full h-auto"
          />
          <canvas
            ref={canvasRef}
            width={GRID}
            height={GRID}
            className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-200"
            style={{ opacity: showOverlay ? 1 : 0 }}
          />
        </div>
      </div>

      {/* Label Box */}
      <div className={`font-mono text-lg flex flex-col gap-3 items-center justify-center p-4 bg-sunken rounded-lg border border-border text-center sm:flex-row${getTutorialFocusClass("output-classification-label")}`}>
        <div className="uppercase text-fg-faint text-sm tracking-wider">Predicted Class: </div>
        <div className="font-bold text-info bg-info-soft px-3 py-1 rounded">
          {predictedLabel.toUpperCase()}
        </div>
      </div>
    </div>
  );
}

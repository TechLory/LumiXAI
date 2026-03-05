import { useState, useRef } from "react";

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

// --- INTERFACES ---

interface HeatmapData {
  image_base64: string;
  raw_matrix: number[][];
}

interface ImageGenViewProps {
  baseImage: string;
  tokens: string[];
  heatmaps: HeatmapData[];
}

export default function ImageGenView({ baseImage, tokens, heatmaps }: ImageGenViewProps) {
  // Token selection
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null); 
  // Pixel hover
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);
  
  const imgRef = useRef<HTMLImageElement>(null);

  // --- MOUSE (HOVER) ---
  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current) return;
    
    const rect = imgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const normX = Math.max(0, Math.min(1, x / rect.width));
    const normY = Math.max(0, Math.min(1, y / rect.height));

    const gridX = Math.floor(normX * 64);
    const gridY = Math.floor(normY * 64);

    setHoveredCell({ x: Math.min(gridX, 63), y: Math.min(gridY, 63) });
  };

  const handleMouseLeaveImage = () => {
    setHoveredCell(null);
  };

  // --- CHOOSE DISPLAYED IMAGE ---
  const displayImage = 
    selectedTokenIndex !== null && heatmaps[selectedTokenIndex]
      ? heatmaps[selectedTokenIndex].image_base64
      : baseImage;

  return (
    <div className="flex flex-col gap-8 w-full select-none">
      
      {/* INTERACTIVE IMAGE */}
      <div className="flex flex-col items-center bg-neutral-900 p-6 rounded-lg border border-neutral-700">
        <div className="flex justify-between w-full max-w-lg mb-4">
          <h3 className="text-gray-400 text-sm uppercase font-bold">
            Generated Image
          </h3>
          <span className="text-xs text-gray-500 italic">
            {hoveredCell ? "Inspecting Pixel Attention" : "Hover image to inspect pixels"}
          </span>
        </div>
        
        <div className="relative inline-block overflow-hidden rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            <img
            ref={imgRef}
            src={`data:image/png;base64,${displayImage}`}
            alt="Generated"
            className="max-w-lg w-full h-auto cursor-crosshair transition-opacity duration-200"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeaveImage}
            />
            {hoveredCell && selectedTokenIndex === null && (
                <div 
                    className="absolute border border-white/50 bg-white/20 pointer-events-none"
                    style={{
                        left: `${(hoveredCell.x / 64) * 100}%`,
                        top: `${(hoveredCell.y / 64) * 100}%`,
                        width: `${100 / 64}%`,
                        height: `${100 / 64}%`
                    }}
                />
            )}
        </div>
      </div>

      {/* TOKEN INPUT */}
      <div className="bg-neutral-900 p-6 rounded-lg border border-neutral-700">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-gray-400 text-sm uppercase font-bold">
            Input Prompt
            </h3>
            <span className="text-xs text-gray-500 italic">
            Click token to lock spatial heatmap
            </span>
        </div>
        
        <div className="flex flex-wrap gap-2">
          {tokens.map((token, idx) => {
            let score = 0;
            let color = "transparent";

            if (hoveredCell && heatmaps[idx]?.raw_matrix) {
              score = heatmaps[idx].raw_matrix[hoveredCell.y][hoveredCell.x];
              color = getScoreColor(score);
            }

            const isSelected = selectedTokenIndex === idx;

            return (
              <div
                key={idx}
                // Toggle selection on click
                onClick={() => setSelectedTokenIndex(isSelected ? null : idx)}
                className={`
                  flex flex-col items-center justify-between
                  min-w-15 min-h-12.5
                  border rounded transition-all duration-150 cursor-pointer
                  ${isSelected 
                    ? "border-purple-500 shadow-[0_0_15px_rgba(168,85,247,0.4)] bg-neutral-800" 
                    : "border-neutral-700 bg-neutral-900 hover:border-neutral-500"}
                `}
              >
                <span className="text-sm font-mono font-bold text-white px-2 py-2">
                  {token}
                </span>
                <div
                  className="w-full text-[10px] text-center font-bold text-white py-0.5 border-t border-neutral-700 transition-colors duration-75"
                  style={{ backgroundColor: color }}
                >
                  {hoveredCell ? formatPercent(score) : "-"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
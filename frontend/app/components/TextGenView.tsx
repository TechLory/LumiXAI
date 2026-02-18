import { useState } from "react";

// --- HELPERS ---

/**
 * Calculate the background color for a score, using green for positive and red for negative, with intensity based on the score magnitude. 
 * Uses opacity to reflect the strength of the score.
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

// --- INTERFACES ---

interface StepData {
  generated_token: string;
  probability: number; // Model confidence (0-1)
  context_tokens: string[]; // Prompt + output up to this step
  attribution_scores: number[]; // Importance of each context_token
}

interface TextGenViewProps {
  trace: StepData[];
}

export default function TextGenView({ trace }: TextGenViewProps) {
  // --- DATA PARSING ---
  const inputTokens = trace[0].context_tokens;
  const outputTokens = trace.map(t => t.generated_token);

  // --- STATE ---
  const [selectedType, setSelectedType] = useState<"input" | "output" | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // --- LOGIC: GET SCORES FOR DISPLAY ---

  /**
   * Function to get the score and color for an 
   * output box at index `outIdx`, based on current selection
   */
  const getOutputBoxData = (outIdx: number) => {
    const step = trace[outIdx];
    const isSelfSelected = selectedType === "output" && selectedIndex === outIdx;

    // Case A: This output is selected -> Show model confidence
    if (isSelfSelected) {
      return { score: step.probability, color: "rgba(255, 255, 255, 0.2)", label: "CONF" };
    }

    // Case B: Other output is selected -> Show attribution if this output is in the context of the selected output
    if (selectedType === "output" && selectedIndex !== null) {
      if (outIdx < selectedIndex) {
        // This output is in the past of the selected output -> Show how much it influenced the selected output
        const targetStep = trace[selectedIndex];
        const ctxIdx = inputTokens.length + outIdx;
        const attrScore = targetStep.attribution_scores[ctxIdx] || 0;
        return { score: attrScore, color: getScoreColor(attrScore), label: "ATTR" };
      }
      // This output is in the future of the selected output (ignore)
      return { score: 0, color: "transparent", label: "-" };
    }

    // Case C: Input is selected -> Show how much it influenced this output
    if (selectedType === "input" && selectedIndex !== null) {
      const attrScore = step.attribution_scores[selectedIndex] || 0;
      return { score: attrScore, color: getScoreColor(attrScore), label: "INFL" };
    }

    // Default: No selection -> Show model confidence with low opacity
    return { score: step.probability, color: "rgba(255, 255, 255, 0.1)", label: "PROB" };
  };

  /**
   * Function to get the score and color for an 
   * input box at index `inIdx`, based on current selection
   */
  const getInputBoxData = (inIdx: number) => {
    const isSelfSelected = selectedType === "input" && selectedIndex === inIdx;
    if (isSelfSelected) {
      return { score: 1.0, color: "rgba(59, 130, 246, 0.5)", label: "SEL" }; // Blue highlight
    }
    if (selectedType === "output" && selectedIndex !== null) {
      const targetStep = trace[selectedIndex];
      const attrScore = targetStep.attribution_scores[inIdx] || 0;
      return { score: attrScore, color: getScoreColor(attrScore), label: "ATTR" };
    }
    return { score: 0, color: "transparent", label: "-" };
  };

  // --- 4. HANDLERS ---
  const handleInputClick = (idx: number) => {
    if (selectedType === "input" && selectedIndex === idx) {
      setSelectedType(null); setSelectedIndex(null);
    } else {
      setSelectedType("input"); setSelectedIndex(idx);
    }
  };
  const handleOutputClick = (idx: number) => {
    if (selectedType === "output" && selectedIndex === idx) {
      setSelectedType(null); setSelectedIndex(null);
    } else {
      setSelectedType("output"); setSelectedIndex(idx);
    }
  };

  // --- RENDER COMPONENT (Reusable Box) ---
  const TokenBox = ({
    word,
    data,
    onClick,
    isSelected
  }: {
    word: string,
    data: { score: number, color: string, label: string },
    onClick: () => void,
    isSelected: boolean
  }) => (
    <button
      onClick={onClick}
      className={`
        flex flex-col items-center justify-between
        min-w-12.5 min-h-12.5 m-1
        border rounded transition-all duration-200
        ${isSelected
          ? "border-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)] bg-neutral-800"
          : "border-neutral-700 bg-neutral-900 hover:border-neutral-500"}
      `}
    >
      <span className="text-sm font-mono font-bold text-white px-2 py-1">
        {word.replace("Ġ", "")}
      </span>

      <div
        className="w-full text-[10px] font-bold text-white py-0.5 border-t border-neutral-700 transition-colors duration-300"
        style={{ backgroundColor: data.color }}
      >
        {data.label === "-" ? "-" : formatPercent(data.score)}
      </div>
    </button>
  );

  return (
    <div className="flex flex-col w-full gap-4 select-none">

      {/* SECTION: INPUTS */}
      <div>
        <h4 className="text-neutral-400 text-xs font-bold uppercase mb-2">Input</h4>
        <div className="flex flex-wrap">
          {inputTokens.map((token, idx) => (
            <TokenBox
              key={`in-${idx}`}
              word={token}
              data={getInputBoxData(idx)}
              onClick={() => handleInputClick(idx)}
              isSelected={selectedType === "input" && selectedIndex === idx}
            />
          ))}
        </div>
      </div>

      {/* SEPARATOR */}
      <div className="border-t border-neutral-700 w-full my-2"></div>

      {/* SECTION: OUTPUTS */}
      <div>
        <h4 className="text-neutral-400 text-xs font-bold uppercase mb-2">Output</h4>
        <div className="flex flex-wrap">
          {outputTokens.map((token, idx) => (
            <TokenBox
              key={`out-${idx}`}
              word={token}
              data={getOutputBoxData(idx)}
              onClick={() => handleOutputClick(idx)}
              isSelected={selectedType === "output" && selectedIndex === idx}
            />
          ))}
        </div>
      </div>

    </div>
  );
}
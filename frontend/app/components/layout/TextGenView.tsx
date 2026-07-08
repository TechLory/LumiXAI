import { useState } from "react";
import type { TutorialFocusTarget } from "../../lib/tutorialGuide";

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

// In "relative" mode, rescales `score` so the strongest value in `referenceScores`
// reaches full color intensity -- different attributors normalize to the same unit
// L2 norm but spread that norm very differently, so raw peaks aren't comparable
// across methods. "absolute" mode passes the score through untouched.
function scaleScore(score: number, referenceScores: number[], mode: "relative" | "absolute") {
  if (mode === "absolute") return score;
  const maxAbs = referenceScores.reduce((max, s) => Math.max(max, Math.abs(s)), 0);
  return maxAbs > 0 ? score / maxAbs : score;
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

export type TextGenerationTutorialSelection = {
  selectedType: "input" | "output";
  selectedIndex: number;
};

interface TextGenViewProps {
  trace: StepData[];
  // Optional tokenizer metadata (attribution values untouched). When `hideSpecialTokens`
  // is on, special tokens are simply not rendered; their original indices are preserved so
  // the bidirectional attribution mapping keeps working.
  inputSpecialMask?: boolean[];
  outputSpecialMask?: boolean[];
  inputTemplateMask?: boolean[];
  hideSpecialTokens?: boolean;
  hideTemplateTokens?: boolean;
  colorScaleMode?: "relative" | "absolute";
  tutorialSelection?: TextGenerationTutorialSelection;
  tutorialFocusTarget?: TutorialFocusTarget;
}

export default function TextGenView({
  trace,
  inputSpecialMask,
  outputSpecialMask,
  inputTemplateMask,
  hideSpecialTokens = false,
  hideTemplateTokens = false,
  colorScaleMode = "relative",
  tutorialSelection,
  tutorialFocusTarget,
}: TextGenViewProps) {
  // Special and template categories overlap: a chat template's control tokens (e.g.
  // <|im_start|>) are flagged by BOTH masks. To give each toggle an independent, visible
  // effect, treat the categories as disjoint by priority — special tokens are governed
  // solely by the special toggle, and the template toggle governs the remaining
  // scaffolding (role markers, newlines). Otherwise hiding template would keep every
  // special token hidden regardless of the special toggle, making it look like a no-op.
  const isInputHidden = (idx: number) => {
    if (inputSpecialMask?.[idx]) return hideSpecialTokens;
    if (inputTemplateMask?.[idx]) return hideTemplateTokens;
    return false;
  };
  // --- DATA PARSING ---
  const inputTokens = trace[0].context_tokens;
  const outputTokens = trace.map(t => t.generated_token);

  // --- STATE ---
  const [selectedType, setSelectedType] = useState<"input" | "output" | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const activeSelectedType = tutorialSelection?.selectedType ?? selectedType;
  const activeSelectedIndex = tutorialSelection?.selectedIndex ?? selectedIndex;

  const getTutorialFocusClass = (target: TutorialFocusTarget) => (
    tutorialFocusTarget === target ? " tutorial-inner-highlight" : ""
  );

  // --- LOGIC: GET SCORES FOR DISPLAY ---

  /**
   * Function to get the score and color for an 
   * output box at index `outIdx`, based on current selection
   */
  const getOutputBoxData = (outIdx: number) => {
    const step = trace[outIdx];
    const isSelfSelected = activeSelectedType === "output" && activeSelectedIndex === outIdx;

    // Case A: This output is selected -> Show model confidence
    if (isSelfSelected) {
      return { score: step.probability, color: "rgba(128, 128, 128, 0.28)", label: "CONF" };
    }

    // Case B: Other output is selected -> Show attribution if this output is in the context of the selected output
    if (activeSelectedType === "output" && activeSelectedIndex !== null) {
      if (outIdx < activeSelectedIndex) {
        // This output is in the past of the selected output -> Show how much it influenced the selected output
        const targetStep = trace[activeSelectedIndex];
        const ctxIdx = inputTokens.length + outIdx;
        const attrScore = targetStep.attribution_scores[ctxIdx] || 0;
        const scaled = scaleScore(attrScore, targetStep.attribution_scores, colorScaleMode);
        return { score: attrScore, color: getScoreColor(scaled), label: "ATTR" };
      }
      // This output is in the future of the selected output (ignore)
      return { score: 0, color: "transparent", label: "-" };
    }

    // Case C: Input is selected -> Show how much it influenced this output
    if (activeSelectedType === "input" && activeSelectedIndex !== null) {
      const attrScore = step.attribution_scores[activeSelectedIndex] || 0;
      const columnScores = trace.map((s) => s.attribution_scores[activeSelectedIndex] || 0);
      const scaled = scaleScore(attrScore, columnScores, colorScaleMode);
      return { score: attrScore, color: getScoreColor(scaled), label: "INFL" };
    }

    // Default: No selection -> Show model confidence with low opacity
    return { score: step.probability, color: "rgba(128, 128, 128, 0.15)", label: "PROB" };
  };

  /**
   * Function to get the score and color for an 
   * input box at index `inIdx`, based on current selection
   */
  const getInputBoxData = (inIdx: number) => {
    const isSelfSelected = activeSelectedType === "input" && activeSelectedIndex === inIdx;
    if (isSelfSelected) {
      return { score: 1.0, color: "rgba(59, 130, 246, 0.5)", label: "SEL" }; // Blue highlight
    }
    if (activeSelectedType === "output" && activeSelectedIndex !== null) {
      const targetStep = trace[activeSelectedIndex];
      const attrScore = targetStep.attribution_scores[inIdx] || 0;
      const scaled = scaleScore(attrScore, targetStep.attribution_scores, colorScaleMode);
      return { score: attrScore, color: getScoreColor(scaled), label: "ATTR" };
    }
    return { score: 0, color: "transparent", label: "-" };
  };

  // --- 4. HANDLERS ---
  const handleInputClick = (idx: number) => {
    if (tutorialSelection) return;

    if (selectedType === "input" && selectedIndex === idx) {
      setSelectedType(null); setSelectedIndex(null);
    } else {
      setSelectedType("input"); setSelectedIndex(idx);
    }
  };
  const handleOutputClick = (idx: number) => {
    if (tutorialSelection) return;

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
        flex max-w-full flex-col items-center justify-between overflow-hidden
        min-w-12.5 min-h-12.5 m-1
        border rounded transition-all duration-200
        ${isSelected
          ? "border-info shadow-[0_0_10px_rgba(59,130,246,0.5)] bg-fill-strong"
          : "border-border bg-sunken hover:border-border-strong"}
      `}
    >
      <span className="max-w-72 break-all px-2 py-1 text-center text-sm font-mono font-bold leading-snug text-fg">
        {word.replace("Ġ", "")}
      </span>

      <div
        className="w-full text-[10px] font-bold text-fg py-0.5 border-t border-border transition-colors duration-300"
        style={{ backgroundColor: data.color }}
      >
        {data.label === "-" ? "-" : formatPercent(data.score)}
      </div>
    </button>
  );

  return (
    <div className="flex min-w-0 flex-col w-full gap-4 select-none">

      {/* SECTION: INPUTS */}
      <div className={`min-w-0${getTutorialFocusClass("output-generation-input")}`}>
        <h4 className="text-fg-subtle text-xs font-bold uppercase mb-2">Input</h4>
        <div className="flex min-w-0 max-w-full flex-wrap">
          {inputTokens.map((token, idx) => {
            if (isInputHidden(idx)) return null;
            return (
              <TokenBox
                key={`in-${idx}`}
                word={token}
                data={getInputBoxData(idx)}
                onClick={() => handleInputClick(idx)}
                isSelected={activeSelectedType === "input" && activeSelectedIndex === idx}
              />
            );
          })}
        </div>
      </div>

      {/* SEPARATOR */}
      <div className="border-t border-border w-full my-2"></div>

      {/* SECTION: OUTPUTS */}
      <div className={`min-w-0${getTutorialFocusClass("output-generation-output")}`}>
        <h4 className="text-fg-subtle text-xs font-bold uppercase mb-2">Output</h4>
        <div className="flex min-w-0 max-w-full flex-wrap">
          {outputTokens.map((token, idx) => {
            if (hideSpecialTokens && outputSpecialMask?.[idx]) return null;
            return (
              <TokenBox
                key={`out-${idx}`}
                word={token}
                data={getOutputBoxData(idx)}
                onClick={() => handleOutputClick(idx)}
                isSelected={activeSelectedType === "output" && activeSelectedIndex === idx}
              />
            );
          })}
        </div>
      </div>

    </div>
  );
}

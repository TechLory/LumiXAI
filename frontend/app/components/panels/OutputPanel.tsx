import { useState } from "react";
import TokenExplained from "../layout/TokenExplained";
import TextGenView from "../layout/TextGenView";
import ImageGenView from "../layout/ImageGenView";

export interface OutputResult {
  target_id?: string | number;
  predicted_token?: string;
  tokens?: string[];
  scores?: any; // array / matrix
  generated_image?: string;
  // Tokenizer metadata (attribution values untouched) used to optionally hide
  // structural tokens from the visualization so heatmaps don't sink into them.
  special_tokens_mask?: boolean[];      // classification: aligned with `tokens`
  input_special_mask?: boolean[];       // generation: aligned with input tokens
  output_special_mask?: boolean[];      // generation: aligned with generated tokens
  input_template_mask?: boolean[];      // generation: chat-template scaffolding on the input side
}

interface OutputPanelProps {
  outputResult: OutputResult | null;
}

export default function OutputPanel({ outputResult }: OutputPanelProps) {
  // Pure display preferences: hiding tokens does NOT re-run the job, it only filters
  // (and rescales) the already-computed attribution for the visualization.
  const [hideSpecialTokens, setHideSpecialTokens] = useState(true);
  const [hideTemplateTokens, setHideTemplateTokens] = useState(true);

  if (!outputResult) return null; // todo: placeholder when no output is available

  // Dynamic Title
  let taskTitle = "Text Classification";
  if (outputResult.generated_image) taskTitle = "Image Generation";
  if (outputResult.target_id === "text_generation") taskTitle = "Text Generation";

  const isImage = !!outputResult.generated_image;
  const isTextGeneration = !isImage && outputResult.target_id === "text_generation";
  const isClassification = !isImage && !isTextGeneration;

  // Special-token filtering is only meaningful for the text views. DAAM (image) already
  // filters special tokens during generation, so the toggle is hidden there.
  const showSpecialTokenToggle = isClassification || isTextGeneration;
  // Chat-template scaffolding only exists for generation with a chat template; hide the
  // toggle entirely when there are no template tokens (e.g. GPT-2, classification).
  const hasTemplateTokens = isTextGeneration && !!outputResult.input_template_mask?.some(Boolean);

  // --- CLASSIFICATION: filter + rescale the visible tokens ---
  const rawTokens = outputResult.tokens || [];
  const rawScores: number[] = Array.isArray(outputResult.scores) ? outputResult.scores : [];
  const classMask = outputResult.special_tokens_mask;

  const visibleClassTokens = rawTokens
    .map((token, index) => ({
      token,
      score: typeof rawScores[index] === "number" ? rawScores[index] : 0,
      isSpecial: !!classMask?.[index],
    }))
    .filter((entry) => !(hideSpecialTokens && entry.isSpecial));

  // Rescale so the strongest visible token reaches full color intensity (prevents the
  // washed-out "everything sinks to [CLS]/[SEP]" look). Only applied while hiding.
  const classMaxAbs = visibleClassTokens.reduce((max, entry) => Math.max(max, Math.abs(entry.score)), 0);
  const classScale = hideSpecialTokens && classMaxAbs > 0 ? 1 / classMaxAbs : 1;

  return (
    <div className="mt-5 mb-10">
      <div className="flex flex-col gap-0">

        <div className="uppercase p-2 text-sm bg-neutral-400/10 text-neutral-300 font-mono mb-2 flex justify-between items-center gap-4">
          <div><span className="font-bold mr-5">// TASK:</span>{taskTitle}</div>

          {/* DISPLAY TOGGLES (instant, no re-run) */}
          <div className="flex items-center gap-4">
            {showSpecialTokenToggle && (
              <button
                type="button"
                onClick={() => setHideSpecialTokens((prev) => !prev)}
                title="Special tokens (e.g. [CLS], [SEP], BOS/EOS) are shown or hidden in the visualization only. This does not re-run the job."
                className="normal-case text-xs text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer flex items-center gap-2"
              >
                <i className={`bx ${hideSpecialTokens ? "bx-hide" : "bx-show"} text-base`}></i>
                {hideSpecialTokens ? "Special tokens hidden" : "Special tokens shown"}
              </button>
            )}
            {hasTemplateTokens && (
              <button
                type="button"
                onClick={() => setHideTemplateTokens((prev) => !prev)}
                title="Chat-template scaffolding (role markers, control tokens, formatting) is shown or hidden in the visualization only. This does not re-run the job."
                className="normal-case text-xs text-neutral-400 hover:text-neutral-200 transition-colors cursor-pointer flex items-center gap-2"
              >
                <i className={`bx ${hideTemplateTokens ? "bx-hide" : "bx-show"} text-base`}></i>
                {hideTemplateTokens ? "Template tokens hidden" : "Template tokens shown"}
              </button>
            )}
          </div>
        </div>

        <div className="bg-neutral-400/10 p-6">

          {/* --- CASE 1: IMAGE GENERATION --- */}
          {isImage && (
            <ImageGenView
              baseImage={outputResult.generated_image!}
              tokens={outputResult.tokens || []}
              heatmaps={Array.isArray(outputResult.scores) ? outputResult.scores : [outputResult.scores]}
            />
          )}

          {/* --- CASE 2: TEXT GENERATION --- */}
          {isTextGeneration && (
            <TextGenView
              trace={outputResult.scores}
              inputSpecialMask={outputResult.input_special_mask}
              outputSpecialMask={outputResult.output_special_mask}
              inputTemplateMask={outputResult.input_template_mask}
              hideSpecialTokens={hideSpecialTokens}
              hideTemplateTokens={hideTemplateTokens}
            />
          )}

          {/* --- CASE 3: TEXT CLASSIFICATION --- */}
          {isClassification && (
            <div className="flex flex-col gap-6">

              {/* Token Box */}
              <div className="flex gap-2 flex-wrap p-5 bg-neutral-900/50 rounded-lg border border-neutral-700/50">
                {visibleClassTokens.map((entry, index) => (
                  <TokenExplained
                    key={index}
                    token={entry.token}
                    score={entry.score * classScale}
                  />
                ))}
              </div>

              {/* Label Box */}
              <div className="font-mono text-lg flex gap-3 items-center justify-center p-4 bg-neutral-900/50 rounded-lg border border-neutral-700/50">
                <div className="uppercase text-neutral-500 text-sm tracking-wider">Predicted Class: </div>

                {outputResult.target_id === 0 ? (
                  <div className="font-bold text-red-400 bg-red-400/10 px-3 py-1 rounded">NEGATIVE</div>
                ) : outputResult.target_id === 1 ? (
                  <div className="font-bold text-emerald-400 bg-emerald-400/10 px-3 py-1 rounded">POSITIVE</div>
                ) : (
                  <div className="font-bold text-blue-400 bg-blue-400/10 px-3 py-1 rounded">
                    {outputResult.predicted_token || "UNKNOWN"}
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { useState } from "react";
import TokenExplained from "../layout/TokenExplained";
import TextGenView from "../layout/TextGenView";
import ImageGenView from "../layout/ImageGenView";
import ImageClassificationView from "../layout/ImageClassificationView";
import type { TutorialOutputInteraction } from "../../types";

export interface OutputResult {
  target_id?: string | number;
  predicted_token?: string;
  tokens?: string[];
  scores?: any; // array / matrix
  generated_image?: string;
  // Original uploaded image (base64), set only by image classification attributors;
  // distinguishes this modality from text classification (which shares the same
  // `target_id`/`predicted_token`/`scores` shape).
  input_image?: string;
  // Tokenizer metadata (attribution values untouched) used to optionally hide
  // structural tokens from the visualization so heatmaps don't sink into them.
  special_tokens_mask?: boolean[];      // classification: aligned with `tokens`
  input_special_mask?: boolean[];       // generation: aligned with input tokens
  output_special_mask?: boolean[];      // generation: aligned with generated tokens
  input_template_mask?: boolean[];      // generation: chat-template scaffolding on the input side
}

interface OutputPanelProps {
  outputResult: OutputResult | null;
  tutorialInteraction?: TutorialOutputInteraction;
}

export default function OutputPanel({ outputResult, tutorialInteraction }: OutputPanelProps) {
  // Pure display preferences: hiding tokens does NOT re-run the job, it only filters
  // the already-computed attribution for the visualization.
  const [hideSpecialTokens, setHideSpecialTokens] = useState(true);
  const [hideTemplateTokens, setHideTemplateTokens] = useState(true);
  // Different attributors normalize to the same unit L2 norm but spread that "energy"
  // very differently (a peaked LIME vector vs. a diffuse DeepLift one), so their peak
  // values aren't comparable. "relative" rescales so the strongest visible token always
  // reaches full color intensity (comparable within a run, not across runs); "absolute"
  // renders the normalized score as-is (comparable across runs of the same attributor,
  // but can look washed out for methods that spread attribution across many tokens).
  const [colorScaleMode, setColorScaleMode] = useState<"relative" | "absolute">("relative");

  if (!outputResult) return null; // todo: placeholder when no output is available

  // Dynamic Title
  let taskTitle = "Text Classification";
  if (outputResult.generated_image) taskTitle = "Image Generation";
  if (outputResult.target_id === "text_generation") taskTitle = "Text Generation";
  if (outputResult.input_image) taskTitle = "Image Classification";

  const isImage = !!outputResult.generated_image;
  const isTextGeneration = !isImage && outputResult.target_id === "text_generation";
  const isImageClassification = !isImage && !isTextGeneration && !!outputResult.input_image;
  const isClassification = !isImage && !isTextGeneration && !isImageClassification;

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
      rawIndex: index,
    }))
    .filter((entry) => !(hideSpecialTokens && entry.isSpecial));

  // Rescale so the strongest visible token reaches full color intensity (prevents the
  // washed-out "everything sinks to [CLS]/[SEP]" look) when in relative mode.
  const classMaxAbs = visibleClassTokens.reduce((max, entry) => Math.max(max, Math.abs(entry.score)), 0);
  const classScale = colorScaleMode === "relative" && classMaxAbs > 0 ? 1 / classMaxAbs : 1;
  const rawPredictedClassLabel =
    typeof outputResult.predicted_token === "string" ? outputResult.predicted_token.trim() : "";
  const fallbackClassLabel =
    outputResult.target_id === 0 ? "NEGATIVE" :
      outputResult.target_id === 1 ? "POSITIVE" :
        "UNKNOWN";
  const displayClassLabel =
    rawPredictedClassLabel && !rawPredictedClassLabel.startsWith("[")
      ? rawPredictedClassLabel
      : fallbackClassLabel;
  const normalizedClassLabel = displayClassLabel.toLowerCase();
  const predictedClassBadgeClass = normalizedClassLabel.includes("negative")
    ? "font-bold text-danger bg-danger-soft px-3 py-1 rounded"
    : normalizedClassLabel.includes("positive")
      ? "font-bold text-ok bg-ok-soft px-3 py-1 rounded"
      : "font-bold text-info bg-info-soft px-3 py-1 rounded";

  return (
    <div className="mt-5 mb-10">
      <div className="flex flex-col gap-0">

        <div className="uppercase p-2 text-sm bg-fill text-fg-muted font-mono mb-2 flex justify-between items-center gap-4">
          <div><span className="font-bold mr-5">// TASK:</span>{taskTitle}</div>

          {/* DISPLAY TOGGLES (instant, no re-run) */}
          <div className="flex items-center gap-4">
            {showSpecialTokenToggle && (
              <button
                type="button"
                onClick={() => setHideSpecialTokens((prev) => !prev)}
                title="Special tokens (e.g. [CLS], [SEP], BOS/EOS) are shown or hidden in the visualization only. This does not re-run the job."
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2"
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
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2"
              >
                <i className={`bx ${hideTemplateTokens ? "bx-hide" : "bx-show"} text-base`}></i>
                {hideTemplateTokens ? "Template tokens hidden" : "Template tokens shown"}
              </button>
            )}
            {(isClassification || isTextGeneration) && (
              <button
                type="button"
                onClick={() => setColorScaleMode((prev) => (prev === "relative" ? "absolute" : "relative"))}
                title="Relative: strongest token always shows full color intensity (comparable within this run). Absolute: raw normalized score (comparable across attributors/runs, may look faint)."
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2"
              >
                <i className={`bx ${colorScaleMode === "relative" ? "bx-trending-up" : "bx-line-chart"} text-base`}></i>
                {colorScaleMode === "relative" ? "Color scale: relative" : "Color scale: absolute"}
              </button>
            )}
          </div>
        </div>

        <div className="bg-fill p-3">

          {/* --- CASE 1: IMAGE GENERATION --- */}
          {isImage && (
            <ImageGenView
              baseImage={outputResult.generated_image!}
              tokens={outputResult.tokens || []}
              heatmaps={Array.isArray(outputResult.scores) ? outputResult.scores : [outputResult.scores]}
              tutorialSelection={tutorialInteraction?.imageSelection}
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
              colorScaleMode={colorScaleMode}
              tutorialSelection={tutorialInteraction?.textGenerationSelection}
            />
          )}

          {/* --- CASE 3: IMAGE CLASSIFICATION --- */}
          {isImageClassification && Array.isArray(outputResult.scores) && outputResult.scores[0] && (
            <ImageClassificationView
              baseImage={outputResult.input_image!}
              heatmap={outputResult.scores[0]}
              predictedLabel={displayClassLabel}
            />
          )}

          {/* --- CASE 4: TEXT CLASSIFICATION --- */}
          {isClassification && (
            <div className="flex flex-col gap-6">

              {/* Token Box */}
              <div className="flex gap-2 flex-wrap p-5 bg-sunken rounded-lg border border-border">
                {visibleClassTokens.map((entry, index) => {
                  const isTutorialToken = tutorialInteraction?.classificationTokenIndex === entry.rawIndex;

                  return (
                    <div
                      key={index}
                      className={isTutorialToken ? "rounded ring-2 ring-info ring-offset-2 ring-offset-sunken" : ""}
                    >
                      <TokenExplained
                        token={entry.token}
                        score={entry.score * classScale}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Label Box */}
              <div className="font-mono text-lg flex gap-3 items-center justify-center p-4 bg-sunken rounded-lg border border-border">
                <div className="uppercase text-fg-faint text-sm tracking-wider">Predicted Class: </div>

                <div className={predictedClassBadgeClass}>
                  {displayClassLabel.toUpperCase()}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

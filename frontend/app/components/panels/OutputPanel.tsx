import { useEffect, useState } from "react";
import TokenExplained from "../layout/TokenExplained";
import TextGenView, { type TextGenerationSelection } from "../layout/TextGenView";
import ImageGenView, { type ImageGenerationSelection } from "../layout/ImageGenView";
import ImageClassificationView from "../layout/ImageClassificationView";
import type { TutorialOutputInteraction } from "../../types";
import type { TutorialFocusTarget } from "../../lib/tutorialGuide";

export interface OutputResult {
  target_id?: string | number;
  predicted_token?: string;
  tokens?: string[];
  scores?: unknown; // array / matrix
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
  tutorialFocusTarget?: TutorialFocusTarget;
}

type ColorScaleMode = "relative" | "absolute";

interface TextGenerationStep {
  generated_token: string;
  probability: number;
  context_tokens: string[];
  attribution_scores: number[];
}

interface HeatmapData {
  image_base64: string;
  raw_matrix: number[][];
}

const defaultImageSelection = (): ImageGenerationSelection => ({
  selectedTokenIndices: [],
  hoveredCell: null,
});

function getTaskTitle(outputResult: OutputResult) {
  if (outputResult.generated_image) return "Image Generation";
  if (outputResult.target_id === "text_generation") return "Text Generation";
  if (outputResult.input_image) return "Image Classification";
  return "Text Classification";
}

function getResultKind(outputResult: OutputResult) {
  const isImage = !!outputResult.generated_image;
  const isTextGeneration = !isImage && outputResult.target_id === "text_generation";
  const isImageClassification = !isImage && !isTextGeneration && !!outputResult.input_image;
  const isClassification = !isImage && !isTextGeneration && !isImageClassification;

  return { isImage, isTextGeneration, isImageClassification, isClassification };
}

function isTextGenerationStep(value: unknown): value is TextGenerationStep {
  if (!value || typeof value !== "object") return false;

  const step = value as Partial<TextGenerationStep>;
  return (
    typeof step.generated_token === "string" &&
    typeof step.probability === "number" &&
    Array.isArray(step.context_tokens) &&
    Array.isArray(step.attribution_scores)
  );
}

function getTextGenerationSteps(scores: unknown): TextGenerationStep[] {
  return Array.isArray(scores) ? scores.filter(isTextGenerationStep) : [];
}

function isHeatmapData(value: unknown): value is HeatmapData {
  if (!value || typeof value !== "object") return false;

  const heatmap = value as Partial<HeatmapData>;
  return Array.isArray(heatmap.raw_matrix);
}

function getHeatmaps(scores: unknown): HeatmapData[] {
  if (Array.isArray(scores)) return scores.filter(isHeatmapData);
  return isHeatmapData(scores) ? [scores] : [];
}

interface ResultVisualizationProps {
  outputResult: OutputResult;
  hideSpecialTokens: boolean;
  hideTemplateTokens: boolean;
  colorScaleMode: ColorScaleMode;
  textGenerationSelection: TextGenerationSelection;
  imageGenerationSelection: ImageGenerationSelection;
  imageClassificationOverlayVisible: boolean;
  onTextGenerationSelectionChange?: (selection: TextGenerationSelection) => void;
  onImageGenerationSelectionChange?: (selection: ImageGenerationSelection) => void;
  onImageClassificationOverlayChange?: (showOverlay: boolean) => void;
  tutorialInteraction?: TutorialOutputInteraction;
  tutorialFocusTarget?: TutorialFocusTarget;
}

function ResultVisualization({
  outputResult,
  hideSpecialTokens,
  hideTemplateTokens,
  colorScaleMode,
  textGenerationSelection,
  imageGenerationSelection,
  imageClassificationOverlayVisible,
  onTextGenerationSelectionChange,
  onImageGenerationSelectionChange,
  onImageClassificationOverlayChange,
  tutorialInteraction,
  tutorialFocusTarget,
}: ResultVisualizationProps) {
  const getTutorialFocusClass = (target: TutorialFocusTarget) => (
    tutorialFocusTarget === target ? " tutorial-inner-highlight" : ""
  );

  const { isImage, isTextGeneration, isImageClassification, isClassification } = getResultKind(outputResult);
  const heatmaps = getHeatmaps(outputResult.scores);
  const textGenerationSteps = getTextGenerationSteps(outputResult.scores);

  // --- CLASSIFICATION: filter + rescale the visible tokens ---
  const rawTokens = outputResult.tokens || [];
  const rawScores: unknown[] = Array.isArray(outputResult.scores) ? outputResult.scores : [];
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
  // Displayed percentage is each token's share of the total attribution (|score| summed
  // over visible tokens), so the numbers are a real proportion that sums to 100% and is
  // independent of the color-scale toggle.
  const classSumAbs = visibleClassTokens.reduce((sum, entry) => sum + Math.abs(entry.score), 0);
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

  const effectiveTextSelection = tutorialInteraction?.textGenerationSelection ?? textGenerationSelection;
  const effectiveImageSelection = tutorialInteraction?.imageSelection
    ? {
      selectedTokenIndices: tutorialInteraction.imageSelection.selectedTokenIndices ?? [],
      hoveredCell: tutorialInteraction.imageSelection.hoveredCell ?? null,
    }
    : imageGenerationSelection;

  return (
    <>
      {/* --- CASE 1: IMAGE GENERATION --- */}
      {isImage && (
        <ImageGenView
          baseImage={outputResult.generated_image!}
          tokens={outputResult.tokens || []}
          heatmaps={heatmaps}
          selection={effectiveImageSelection}
          onSelectionChange={tutorialInteraction?.imageSelection ? undefined : onImageGenerationSelectionChange}
          tutorialFocusTarget={tutorialFocusTarget}
        />
      )}

      {/* --- CASE 2: TEXT GENERATION --- */}
      {isTextGeneration && (
        <TextGenView
          trace={textGenerationSteps}
          inputSpecialMask={outputResult.input_special_mask}
          outputSpecialMask={outputResult.output_special_mask}
          inputTemplateMask={outputResult.input_template_mask}
          hideSpecialTokens={hideSpecialTokens}
          hideTemplateTokens={hideTemplateTokens}
          colorScaleMode={colorScaleMode}
          selection={effectiveTextSelection}
          onSelectionChange={tutorialInteraction?.textGenerationSelection ? undefined : onTextGenerationSelectionChange}
          tutorialFocusTarget={tutorialFocusTarget}
        />
      )}

      {/* --- CASE 3: IMAGE CLASSIFICATION --- */}
      {isImageClassification && heatmaps[0] && (
        <ImageClassificationView
          baseImage={outputResult.input_image!}
          heatmap={heatmaps[0]}
          predictedLabel={displayClassLabel}
          showOverlay={imageClassificationOverlayVisible}
          onShowOverlayChange={onImageClassificationOverlayChange}
          tutorialFocusTarget={tutorialFocusTarget}
        />
      )}

      {/* --- CASE 4: TEXT CLASSIFICATION --- */}
      {isClassification && (
        <div className="flex flex-col gap-6">

          {/* Token Box */}
          <div className={`flex gap-2 flex-wrap p-5 bg-sunken rounded-lg border border-border${getTutorialFocusClass("output-classification-tokens")}`}>
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
                    percentage={classSumAbs > 0 ? entry.score / classSumAbs : 0}
                  />
                </div>
              );
            })}
          </div>

          {/* Label Box */}
          <div className={`font-mono text-lg flex flex-col gap-3 items-center justify-center p-4 bg-sunken rounded-lg border border-border text-center sm:flex-row${getTutorialFocusClass("output-classification-label")}`}>
            <div className="uppercase text-fg-faint text-sm tracking-wider">Predicted Class: </div>

            <div className={predictedClassBadgeClass}>
              {displayClassLabel.toUpperCase()}
            </div>
          </div>

        </div>
      )}
    </>
  );
}

export default function OutputPanel({ outputResult, tutorialInteraction, tutorialFocusTarget }: OutputPanelProps) {
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
  const [colorScaleMode, setColorScaleMode] = useState<ColorScaleMode>("relative");
  const [textGenerationSelection, setTextGenerationSelection] = useState<TextGenerationSelection>(null);
  const [imageGenerationSelection, setImageGenerationSelection] = useState<ImageGenerationSelection>(defaultImageSelection);
  const [imageClassificationOverlayVisible, setImageClassificationOverlayVisible] = useState(true);

  useEffect(() => {
    setTextGenerationSelection(null);
    setImageGenerationSelection(defaultImageSelection());
    setImageClassificationOverlayVisible(true);
  }, [outputResult]);

  if (!outputResult) return null; // todo: placeholder when no output is available

  const taskTitle = getTaskTitle(outputResult);
  const { isTextGeneration, isClassification } = getResultKind(outputResult);

  // Special-token filtering is only meaningful for the text views. DAAM (image) already
  // filters special tokens during generation, so the toggle is hidden there.
  const showSpecialTokenToggle = isClassification || isTextGeneration;
  // Chat-template scaffolding only exists for generation with a chat template; hide the
  // toggle entirely when there are no template tokens (e.g. GPT-2, classification).
  const hasTemplateTokens = isTextGeneration && !!outputResult.input_template_mask?.some(Boolean);

  return (
    <div className="mt-5 mb-10 min-w-0">
      <div className="flex min-w-0 flex-col gap-0">

        <div className="uppercase p-2 text-sm bg-fill text-fg-muted font-mono mb-2 flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0 break-words"><span className="font-bold mr-5">{"// TASK:"}</span>{taskTitle}</div>

          {/* DISPLAY TOGGLES (instant, no re-run) */}
          <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2">
            {showSpecialTokenToggle && (
              <button
                type="button"
                onClick={() => setHideSpecialTokens((prev) => !prev)}
                title="Special tokens (e.g. [CLS], [SEP], BOS/EOS) are shown or hidden in the visualization only. This does not re-run the job."
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2 text-left"
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
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2 text-left"
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
                className="normal-case text-xs text-fg-subtle hover:text-fg transition-colors cursor-pointer flex items-center gap-2 text-left"
              >
                <i className={`bx ${colorScaleMode === "relative" ? "bx-trending-up" : "bx-line-chart"} text-base`}></i>
                {colorScaleMode === "relative" ? "Color scale: relative" : "Color scale: absolute"}
              </button>
            )}
          </div>
        </div>

        <div className={`bg-fill p-3 min-w-0 overflow-x-auto${tutorialFocusTarget === "output-result" ? " tutorial-inner-highlight" : ""}`}>
          <ResultVisualization
            outputResult={outputResult}
            hideSpecialTokens={hideSpecialTokens}
            hideTemplateTokens={hideTemplateTokens}
            colorScaleMode={colorScaleMode}
            textGenerationSelection={textGenerationSelection}
            imageGenerationSelection={imageGenerationSelection}
            imageClassificationOverlayVisible={imageClassificationOverlayVisible}
            onTextGenerationSelectionChange={setTextGenerationSelection}
            onImageGenerationSelectionChange={setImageGenerationSelection}
            onImageClassificationOverlayChange={setImageClassificationOverlayVisible}
            tutorialInteraction={tutorialInteraction}
            tutorialFocusTarget={tutorialFocusTarget}
          />
        </div>

      </div>

    </div>
  );
}

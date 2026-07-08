import { useEffect, useRef, useState } from "react";
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

export interface ResultExportContext {
  inputText: string;
  inputImageBase64?: string | null;
  inputImageFileName?: string | null;
  sourceName?: string | null;
  sourceId?: string | null;
  modelName?: string | null;
  attributorName?: string | null;
  detectedTask?: string | null;
  wrapperName?: string | null;
  seed?: string | null;
  maxNewTokens?: string | null;
}

interface OutputPanelProps {
  outputResult: OutputResult | null;
  exportContext?: ResultExportContext;
  tutorialInteraction?: TutorialOutputInteraction;
  tutorialFocusTarget?: TutorialFocusTarget;
}

type ExportFormat = "pdf" | "svg" | "jpg" | "png";
type ExportLayout = "side-by-side" | "stacked";
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

function withImagePrefix(base64: string) {
  return base64.startsWith("data:") ? base64 : `data:image/png;base64,${base64}`;
}

function sanitizeFilePart(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "result";
}

function dataUrlToBlob(dataUrl: string) {
  return fetch(dataUrl).then((res) => res.blob());
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function waitForNextFrame() {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

function waitForImages(node: HTMLElement) {
  const images = Array.from(node.querySelectorAll("img"));
  return Promise.all(images.map((image) => {
    if (image.complete) return Promise.resolve();
    return new Promise<void>((resolve) => {
      image.onload = () => resolve();
      image.onerror = () => resolve();
    });
  })).then(() => undefined);
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load export image."));
    image.src = src;
  });
}

function dataUrlToBytes(dataUrl: string) {
  const base64 = dataUrl.split(",")[1] ?? "";
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function appendPdfString(chunks: Array<string | Uint8Array>, value: string) {
  chunks.push(value);
  return value.length;
}

function appendPdfBytes(chunks: Array<string | Uint8Array>, value: Uint8Array) {
  chunks.push(value);
  return value.byteLength;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function buildImagePdfBlob(pages: Array<{
  bytes: Uint8Array;
  pixelWidth: number;
  pixelHeight: number;
  displayWidth: number;
  displayHeight: number;
  x: number;
  y: number;
}>, pageWidth: number, pageHeight: number) {
  const chunks: Array<string | Uint8Array> = [];
  const offsets: number[] = [0];
  let byteOffset = 0;

  const appendString = (value: string) => {
    byteOffset += appendPdfString(chunks, value);
  };
  const appendBytes = (value: Uint8Array) => {
    byteOffset += appendPdfBytes(chunks, value);
  };
  const addObject = (objectNumber: number, writeBody: () => void) => {
    offsets[objectNumber] = byteOffset;
    appendString(`${objectNumber} 0 obj\n`);
    writeBody();
    appendString("\nendobj\n");
  };

  appendString("%PDF-1.4\n");

  addObject(1, () => {
    appendString("<< /Type /Catalog /Pages 2 0 R >>");
  });

  addObject(2, () => {
    const kids = pages.map((_, index) => `${3 + index * 3} 0 R`).join(" ");
    appendString(`<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  });

  pages.forEach((page, index) => {
    const pageObject = 3 + index * 3;
    const contentObject = pageObject + 1;
    const imageObject = pageObject + 2;
    const imageName = `Im${index + 1}`;
    const contentStream = [
      "q",
      `${page.displayWidth.toFixed(4)} 0 0 ${page.displayHeight.toFixed(4)} ${page.x.toFixed(4)} ${page.y.toFixed(4)} cm`,
      `/${imageName} Do`,
      "Q",
    ].join("\n");

    addObject(pageObject, () => {
      appendString(
        `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] ` +
        `/Resources << /XObject << /${imageName} ${imageObject} 0 R >> >> ` +
        `/Contents ${contentObject} 0 R >>`
      );
    });

    addObject(contentObject, () => {
      appendString(`<< /Length ${contentStream.length} >>\nstream\n${contentStream}\nendstream`);
    });

    addObject(imageObject, () => {
      appendString(
        `<< /Type /XObject /Subtype /Image /Width ${page.pixelWidth} /Height ${page.pixelHeight} ` +
        `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${page.bytes.byteLength} >>\nstream\n`
      );
      appendBytes(page.bytes);
      appendString("\nendstream");
    });
  });

  const xrefOffset = byteOffset;
  appendString(`xref\n0 ${offsets.length}\n`);
  appendString("0000000000 65535 f \n");

  for (let objectNumber = 1; objectNumber < offsets.length; objectNumber += 1) {
    appendString(`${String(offsets[objectNumber]).padStart(10, "0")} 00000 n \n`);
  }

  appendString(
    `trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`
  );

  const blobParts = chunks.map((chunk) => (
    typeof chunk === "string" ? chunk : toArrayBuffer(chunk)
  ));

  return new Blob(blobParts, { type: "application/pdf" });
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Failed to export this result.";
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

function getTextGenerationSelectionLabel(outputResult: OutputResult, selection: TextGenerationSelection) {
  const steps = getTextGenerationSteps(outputResult.scores);
  if (!selection || steps.length === 0) return "";

  const firstStep = steps[0];
  const inputTokens: string[] = Array.isArray(firstStep?.context_tokens) ? firstStep.context_tokens : [];
  const outputTokens: string[] = steps.map((step) => step.generated_token).filter(Boolean);
  const token = selection.selectedType === "input"
    ? inputTokens[selection.selectedIndex]
    : outputTokens[selection.selectedIndex];

  return token ? `${selection.selectedType} token: ${token}` : "";
}

function getImageSelectionLabel(outputResult: OutputResult, selection: ImageGenerationSelection) {
  const tokens = outputResult.tokens ?? [];
  const selectedTokens = selection.selectedTokenIndices
    .map((index) => tokens[index])
    .filter(Boolean);

  return selectedTokens.length > 0 ? `prompt token${selectedTokens.length > 1 ? "s" : ""}: ${selectedTokens.join(" + ")}` : "";
}

function buildConfigRows(
  outputResult: OutputResult,
  exportContext: ResultExportContext | undefined,
  taskTitle: string,
  hideSpecialTokens: boolean,
  hideTemplateTokens: boolean,
  colorScaleMode: ColorScaleMode,
  textGenerationSelection: TextGenerationSelection,
  imageGenerationSelection: ImageGenerationSelection,
  imageClassificationOverlayVisible: boolean
) {
  const { isImage, isTextGeneration, isImageClassification, isClassification } = getResultKind(outputResult);
  const showSpecialTokenToggle = isClassification || isTextGeneration;
  const hasTemplateTokens = isTextGeneration && !!outputResult.input_template_mask?.some(Boolean);
  const selectedTokenLabel = isTextGeneration
    ? getTextGenerationSelectionLabel(outputResult, textGenerationSelection)
    : isImage
      ? getImageSelectionLabel(outputResult, imageGenerationSelection)
      : "";

  const rows: Array<[string, string | null | undefined]> = [
    ["Task", taskTitle],
    ["Model", exportContext?.modelName],
    ["Attribution type", exportContext?.attributorName],
    ["Source", exportContext?.sourceName || exportContext?.sourceId],
    ["Detected task", exportContext?.detectedTask],
    ["Wrapper", exportContext?.wrapperName],
    ["Input file", exportContext?.inputImageFileName],
    ["Color scale", (isClassification || isTextGeneration) ? colorScaleMode : ""],
    ["Special tokens", showSpecialTokenToggle ? (hideSpecialTokens ? "hidden" : "shown") : ""],
    ["Template tokens", hasTemplateTokens ? (hideTemplateTokens ? "hidden" : "shown") : ""],
    ["Image overlay", isImageClassification ? (imageClassificationOverlayVisible ? "shown" : "hidden") : ""],
    ["Selected token", selectedTokenLabel],
    ["Seed", isImage ? (exportContext?.seed?.trim() || "random") : ""],
    ["Max new tokens", isTextGeneration ? (exportContext?.maxNewTokens?.trim() || "default") : ""],
  ];

  return rows.filter(([, value]) => typeof value === "string" && value.trim().length > 0) as Array<[string, string]>;
}

async function savePdfFromPng(dataUrl: string, filename: string, layout: ExportLayout) {
  const image = await loadImage(dataUrl);
  const pageWidth = layout === "side-by-side" ? 841.89 : 595.28;
  const pageHeight = layout === "side-by-side" ? 595.28 : 841.89;
  const margin = 32;
  const contentWidth = pageWidth - margin * 2;
  const contentHeight = pageHeight - margin * 2;
  const sliceHeightPx = Math.max(1, Math.floor((contentHeight / contentWidth) * image.width));
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create a PDF canvas.");

  canvas.width = image.width;
  const pages: Array<{
    bytes: Uint8Array;
    pixelWidth: number;
    pixelHeight: number;
    displayWidth: number;
    displayHeight: number;
    x: number;
    y: number;
  }> = [];

  for (let offsetY = 0; offsetY < image.height; offsetY += sliceHeightPx) {
    const sourceHeight = Math.min(sliceHeightPx, image.height - offsetY);
    canvas.height = sourceHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, offsetY, image.width, sourceHeight, 0, 0, image.width, sourceHeight);

    const sliceDataUrl = canvas.toDataURL("image/jpeg", 0.95);
    const displayHeight = (sourceHeight / image.width) * contentWidth;
    pages.push({
      bytes: dataUrlToBytes(sliceDataUrl),
      pixelWidth: canvas.width,
      pixelHeight: canvas.height,
      displayWidth: contentWidth,
      displayHeight,
      x: margin,
      y: pageHeight - margin - displayHeight,
    });
  }

  downloadBlob(buildImagePdfBlob(pages, pageWidth, pageHeight), filename);
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

interface ResultExportSheetProps {
  outputResult: OutputResult;
  exportContext?: ResultExportContext;
  taskTitle: string;
  layout: ExportLayout;
  includeConfig: boolean;
  hideSpecialTokens: boolean;
  hideTemplateTokens: boolean;
  colorScaleMode: ColorScaleMode;
  textGenerationSelection: TextGenerationSelection;
  imageGenerationSelection: ImageGenerationSelection;
  imageClassificationOverlayVisible: boolean;
  tutorialInteraction?: TutorialOutputInteraction;
}

function ResultExportSheet({
  outputResult,
  exportContext,
  taskTitle,
  layout,
  includeConfig,
  hideSpecialTokens,
  hideTemplateTokens,
  colorScaleMode,
  textGenerationSelection,
  imageGenerationSelection,
  imageClassificationOverlayVisible,
  tutorialInteraction,
}: ResultExportSheetProps) {
  const inputImage = exportContext?.inputImageBase64 ?? outputResult.input_image ?? null;
  const configRows = buildConfigRows(
    outputResult,
    exportContext,
    taskTitle,
    hideSpecialTokens,
    hideTemplateTokens,
    colorScaleMode,
    textGenerationSelection,
    imageGenerationSelection,
    imageClassificationOverlayVisible
  );

  return (
    <div className="bg-surface text-fg p-8 font-sans">
      <div className="mb-6 flex items-start justify-between gap-6 border-b border-border pb-4 font-mono">
        <div>
          <div className="text-xs uppercase text-fg-faint">{"// LumiXAI Export"}</div>
          <div className="mt-1 text-xl font-bold text-fg">Input and Output Result</div>
        </div>
        <div className="text-right text-xs uppercase text-fg-faint">{new Date().toLocaleString()}</div>
      </div>

      {includeConfig && configRows.length > 0 && (
        <div className="mb-6 grid gap-2 border border-border bg-fill p-4 font-mono text-xs sm:grid-cols-2">
          {configRows.map(([label, value]) => (
            <div key={label} className="min-w-0">
              <span className="font-bold uppercase text-fg-subtle">{"// "}{label}: </span>
              <span className="break-words text-fg">{value}</span>
            </div>
          ))}
        </div>
      )}

      <div className={layout === "side-by-side" ? "grid grid-cols-2 gap-6" : "flex flex-col gap-6"}>
        <section className="min-w-0 border border-border bg-fill p-4">
          <h2 className="mb-3 font-mono text-sm font-bold uppercase text-fg-subtle">Input</h2>
          {inputImage ? (
            <div className="flex flex-col items-center gap-3 bg-sunken p-4">
              <img
                src={withImagePrefix(inputImage)}
                alt="Exported input"
                className="max-h-[520px] max-w-full object-contain"
              />
              {exportContext?.inputImageFileName && (
                <div className="break-all font-mono text-xs text-fg-subtle">{exportContext.inputImageFileName}</div>
              )}
            </div>
          ) : (
            <div className="min-h-48 whitespace-pre-wrap break-words bg-sunken p-4 font-mono text-sm leading-relaxed text-fg">
              {exportContext?.inputText?.trim() || "(empty input)"}
            </div>
          )}
        </section>

        <section className="min-w-0 border border-border bg-fill p-4">
          <h2 className="mb-3 font-mono text-sm font-bold uppercase text-fg-subtle">Output</h2>
          <div className="mb-3 bg-sunken px-3 py-2 font-mono text-xs uppercase text-fg-subtle">
            <span className="font-bold">{"// Task: "}</span>{taskTitle}
          </div>
          <div className="min-w-0 overflow-hidden bg-fill p-3">
            <ResultVisualization
              outputResult={outputResult}
              hideSpecialTokens={hideSpecialTokens}
              hideTemplateTokens={hideTemplateTokens}
              colorScaleMode={colorScaleMode}
              textGenerationSelection={textGenerationSelection}
              imageGenerationSelection={imageGenerationSelection}
              imageClassificationOverlayVisible={imageClassificationOverlayVisible}
              tutorialInteraction={tutorialInteraction}
            />
          </div>
        </section>
      </div>
    </div>
  );
}

export default function OutputPanel({ outputResult, exportContext, tutorialInteraction, tutorialFocusTarget }: OutputPanelProps) {
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
  const [exportFormat, setExportFormat] = useState<ExportFormat>("png");
  const [exportLayout, setExportLayout] = useState<ExportLayout>("side-by-side");
  const [includeExportConfig, setIncludeExportConfig] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTextGenerationSelection(null);
    setImageGenerationSelection(defaultImageSelection());
    setImageClassificationOverlayVisible(true);
    setExportError(null);
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

  const handleExport = async () => {
    const node = exportRef.current;
    if (!node || isExporting) return;

    setIsExporting(true);
    setExportError(null);

    try {
      await document.fonts?.ready;
      await waitForImages(node);
      await waitForNextFrame();

      const { toJpeg, toPng, toSvg } = await import("html-to-image");
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      const stem = `lumixai-${sanitizeFilePart(taskTitle)}-${timestamp}`;
      const backgroundColor = getComputedStyle(node).backgroundColor || "#ffffff";
      const commonOptions = {
        cacheBust: true,
        backgroundColor,
        pixelRatio: 2,
      };

      if (exportFormat === "svg") {
        const dataUrl = await toSvg(node, commonOptions);
        downloadBlob(await dataUrlToBlob(dataUrl), `${stem}.svg`);
        return;
      }

      if (exportFormat === "jpg") {
        const dataUrl = await toJpeg(node, { ...commonOptions, quality: 0.95 });
        downloadBlob(await dataUrlToBlob(dataUrl), `${stem}.jpg`);
        return;
      }

      const dataUrl = await toPng(node, commonOptions);

      if (exportFormat === "pdf") {
        await savePdfFromPng(dataUrl, `${stem}.pdf`, exportLayout);
        return;
      }

      downloadBlob(await dataUrlToBlob(dataUrl), `${stem}.png`);
    } catch (error: unknown) {
      setExportError(getErrorMessage(error));
    } finally {
      setIsExporting(false);
    }
  };

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
            <div className="flex min-w-0 flex-wrap items-center gap-2 normal-case text-xs text-fg-subtle">
              <select
                value={exportFormat}
                onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                className="max-w-24 border border-border bg-sunken px-2 py-1 text-fg outline-none focus:border-border-strong"
                aria-label="Export format"
              >
                <option value="png">PNG</option>
                <option value="jpg">JPG</option>
                <option value="svg">SVG</option>
                <option value="pdf">PDF</option>
              </select>
              <select
                value={exportLayout}
                onChange={(event) => setExportLayout(event.target.value as ExportLayout)}
                className="max-w-36 border border-border bg-sunken px-2 py-1 text-fg outline-none focus:border-border-strong"
                aria-label="Export layout"
              >
                <option value="side-by-side">Side by side</option>
                <option value="stacked">Stacked</option>
              </select>
              <label className="flex cursor-pointer items-center gap-1 text-fg-subtle hover:text-fg">
                <input
                  type="checkbox"
                  checked={includeExportConfig}
                  onChange={(event) => setIncludeExportConfig(event.target.checked)}
                  className="accent-info"
                />
                Config
              </label>
              <button
                type="button"
                onClick={handleExport}
                disabled={isExporting}
                className="flex items-center gap-1 border border-info-line bg-info-soft px-2 py-1 text-info transition-colors hover:bg-info-hover disabled:cursor-not-allowed disabled:opacity-50"
              >
                <i className={`bx ${isExporting ? "bx-loader animate-spin" : "bx-download"} text-base`}></i>
                {isExporting ? "Saving" : "Save"}
              </button>
            </div>
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

        {exportError && (
          <div className="mt-2 bg-danger-soft px-3 py-2 font-mono text-xs text-danger">
            Export failed: {exportError}
          </div>
        )}
      </div>

      <div className="pointer-events-none fixed top-0 -left-[20000px]" aria-hidden="true">
        <div
          ref={exportRef}
          className="bg-surface text-fg"
          style={{ width: exportLayout === "side-by-side" ? 1440 : 920 }}
        >
          <ResultExportSheet
            outputResult={outputResult}
            exportContext={exportContext}
            taskTitle={taskTitle}
            layout={exportLayout}
            includeConfig={includeExportConfig}
            hideSpecialTokens={hideSpecialTokens}
            hideTemplateTokens={hideTemplateTokens}
            colorScaleMode={colorScaleMode}
            textGenerationSelection={tutorialInteraction?.textGenerationSelection ?? textGenerationSelection}
            imageGenerationSelection={tutorialInteraction?.imageSelection ? {
              selectedTokenIndices: tutorialInteraction.imageSelection.selectedTokenIndices ?? [],
              hoveredCell: tutorialInteraction.imageSelection.hoveredCell ?? null,
            } : imageGenerationSelection}
            imageClassificationOverlayVisible={imageClassificationOverlayVisible}
            tutorialInteraction={tutorialInteraction}
          />
        </div>
      </div>
    </div>
  );
}

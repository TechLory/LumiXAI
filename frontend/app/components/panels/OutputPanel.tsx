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

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load export image."));
    image.src = src;
  });
}

async function exportSvgToDataUrl(
  svgText: string,
  width: number,
  height: number,
  type: "image/png" | "image/jpeg",
  backgroundColor: string,
  pixelRatio: number = 2,
  quality?: number
) {
  const svgBlob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  try {
    const image = await loadImage(svgUrl);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create an export canvas.");

    canvas.width = Math.ceil(width * pixelRatio);
    canvas.height = Math.ceil(height * pixelRatio);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = backgroundColor || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL(type, quality);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
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

interface ExportColors {
  surface: string;
  fill: string;
  sunken: string;
  border: string;
  borderStrong: string;
  fg: string;
  fgMuted: string;
  fgSubtle: string;
  fgFaint: string;
  info: string;
  infoSoft: string;
  ok: string;
  okSoft: string;
  danger: string;
  dangerSoft: string;
  accent: string;
}

interface LoadedExportImage {
  src: string;
  width: number;
  height: number;
}

interface ExportResources {
  inputImage: LoadedExportImage | null;
  generatedImage: LoadedExportImage | null;
  classificationImage: LoadedExportImage | null;
}

interface SvgLayout {
  svgText: string;
  width: number;
  height: number;
}

type SvgParts = string[];

function escapeXml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function getExportColors(): ExportColors {
  return {
    surface: "#ffffff",
    fill: "#ffffff",
    sunken: "#f8fafc",
    border: "#cbd5e1",
    borderStrong: "#64748b",
    fg: "#111827",
    fgMuted: "#374151",
    fgSubtle: "#4b5563",
    fgFaint: "#6b7280",
    info: "#1d4ed8",
    infoSoft: "#dbeafe",
    ok: "#047857",
    okSoft: "#d1fae5",
    danger: "#b91c1c",
    dangerSoft: "#fee2e2",
    accent: "#7e22ce",
  };
}

function svgRect(parts: SvgParts, x: number, y: number, width: number, height: number, fill: string, stroke?: string, strokeWidth: number = 1, radius: number = 0) {
  parts.push(
    `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" ry="${radius}" fill="${escapeXml(fill)}"${stroke ? ` stroke="${escapeXml(stroke)}" stroke-width="${strokeWidth}"` : ""}/>`
  );
}

function svgLine(parts: SvgParts, x1: number, y1: number, x2: number, y2: number, stroke: string) {
  parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${escapeXml(stroke)}" stroke-width="1"/>`);
}

function svgImage(parts: SvgParts, href: string, x: number, y: number, width: number, height: number, opacity?: number) {
  parts.push(
    `<image href="${escapeXml(href)}" x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="none"${opacity !== undefined ? ` opacity="${opacity}"` : ""}/>`
  );
}

function svgText(
  parts: SvgParts,
  text: string,
  x: number,
  y: number,
  color: string,
  size: number = 14,
  weight: number | string = 400,
  family: "mono" | "sans" = "mono",
  anchor: "start" | "middle" | "end" = "start"
) {
  const fontFamily = family === "mono" ? "Courier New, monospace" : "Arial, Helvetica, sans-serif";
  parts.push(
    `<text x="${x}" y="${y + size}" fill="${escapeXml(color)}" font-family="${fontFamily}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${escapeXml(text)}</text>`
  );
}

function getMeasureContext() {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create an export measurement canvas.");
  return context;
}

function setFont(ctx: CanvasRenderingContext2D, size: number, weight: number | string = 400, family: "mono" | "sans" = "mono") {
  const fontFamily = family === "mono" ? "Courier New, monospace" : "Arial, Helvetica, sans-serif";
  ctx.font = `${weight} ${size}px ${fontFamily}`;
}

function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, size: number = 14, weight: number | string = 400, family: "mono" | "sans" = "mono") {
  setFont(ctx, size, weight, family);
  const paragraphs = text.split(/\n/);
  const lines: string[] = [];

  paragraphs.forEach((paragraph) => {
    const words = paragraph.trim() ? paragraph.split(/\s+/) : [""];
    let line = "";

    words.forEach((word) => {
      const candidate = line ? `${line} ${word}` : word;
      if (ctx.measureText(candidate).width <= maxWidth) {
        line = candidate;
        return;
      }

      if (line) lines.push(line);
      line = word;

      while (ctx.measureText(line).width > maxWidth && line.length > 1) {
        let cut = line.length - 1;
        while (cut > 1 && ctx.measureText(line.slice(0, cut)).width > maxWidth) cut -= 1;
        lines.push(line.slice(0, cut));
        line = line.slice(cut);
      }
    });

    lines.push(line);
  });

  return lines;
}

function drawWrappedText(parts: SvgParts, ctx: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, color: string, size: number = 14, weight: number | string = 400, family: "mono" | "sans" = "mono") {
  const lines = wrapText(ctx, text, maxWidth, size, weight, family);
  const lineHeight = Math.ceil(size * 1.45);

  lines.forEach((line, index) => {
    svgText(parts, line, x, y + index * lineHeight, color, size, weight, family);
  });

  return lines.length * lineHeight;
}

function containSize(sourceWidth: number, sourceHeight: number, maxWidth: number, maxHeight: number) {
  if (sourceWidth <= 0 || sourceHeight <= 0) return { width: maxWidth, height: maxHeight };
  const scale = Math.min(maxWidth / sourceWidth, maxHeight / sourceHeight, 1);
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

function getScoreColor(score: number) {
  if (Math.abs(score) < 0.001) return "transparent";
  const intensity = Math.min(Math.abs(score) * 2.5, 1);
  return score >= 0
    ? `rgba(34, 197, 94, ${intensity})`
    : `rgba(239, 68, 68, ${intensity})`;
}

function getImageScoreColor(score: number) {
  if (score < 0.01) return "transparent";
  const intensity = Math.min(score * 2.5, 1);
  return `rgba(34, 197, 94, ${intensity})`;
}

function formatPercent(val: number) {
  if (val === undefined || val === null) return "-";
  return `${(val * 100).toFixed(0)}%`;
}

function scaleScore(score: number, referenceScores: number[], mode: ColorScaleMode) {
  if (mode === "absolute") return score;
  const maxAbs = referenceScores.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
  return maxAbs > 0 ? score / maxAbs : score;
}

function jet(t: number): [number, number, number] {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const r = clamp(1.5 - Math.abs(4 * t - 3));
  const g = clamp(1.5 - Math.abs(4 * t - 2));
  const b = clamp(1.5 - Math.abs(4 * t - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function createHeatmapDataUrl(matrix: number[][], alpha: number = 0.6) {
  const height = matrix.length;
  const width = matrix[0]?.length ?? 0;
  if (!width || !height) return "";

  const flat = matrix.flat();
  const sorted = [...flat].sort((a, b) => a - b);
  const pct = (q: number) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
  const vmin = pct(0.01);
  const vmax = pct(0.99);
  const denom = vmax - vmin || 1e-8;

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) return "";

  canvas.width = width;
  canvas.height = height;
  const image = context.createImageData(width, height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const value = Math.max(0, Math.min(1, (matrix[y][x] - vmin) / denom));
      const [r, g, b] = jet(value);
      image.data[index * 4] = r;
      image.data[index * 4 + 1] = g;
      image.data[index * 4 + 2] = b;
      image.data[index * 4 + 3] = Math.round(alpha * 255);
    }
  }

  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function aggregateHeatmaps(heatmaps: HeatmapData[], selectedTokenIndices: number[]) {
  const valid = selectedTokenIndices
    .map((index) => heatmaps[index]?.raw_matrix)
    .filter((matrix): matrix is number[][] => Array.isArray(matrix));
  if (valid.length === 0) return null;

  const height = valid[0].length;
  const width = valid[0][0]?.length ?? 0;
  const aggregate = Array.from({ length: height }, () => new Array(width).fill(0));

  valid.forEach((matrix) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        aggregate[y][x] += matrix[y]?.[x] ?? 0;
      }
    }
  });

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      aggregate[y][x] /= valid.length;
    }
  }

  return aggregate;
}

function tokenDisplayText(token: string) {
  return token.replace("Ġ", " ").replace("##", "");
}

interface TokenBoxSpec {
  text: string;
  footer: string;
  footerColor: string;
  selected?: boolean;
  selectedColor?: string;
}

function layoutTokenBoxes(ctx: CanvasRenderingContext2D, specs: TokenBoxSpec[], width: number) {
  setFont(ctx, 14, 700, "mono");
  const gap = 8;
  const boxHeight = 52;
  const boxes: Array<TokenBoxSpec & { x: number; y: number; width: number; height: number }> = [];
  let x = 0;
  let y = 0;
  let rowHeight = boxHeight;

  specs.forEach((spec) => {
    const measuredWidth = Math.ceil(ctx.measureText(spec.text).width) + 20;
    const boxWidth = Math.min(288, Math.max(52, measuredWidth));

    if (x > 0 && x + boxWidth > width) {
      x = 0;
      y += rowHeight + gap;
      rowHeight = boxHeight;
    }

    boxes.push({ ...spec, x, y, width: boxWidth, height: boxHeight });
    x += boxWidth + gap;
  });

  return {
    boxes,
    height: boxes.length === 0 ? boxHeight : y + rowHeight,
  };
}

function drawTokenBoxes(parts: SvgParts, boxes: Array<TokenBoxSpec & { x: number; y: number; width: number; height: number }>, baseX: number, baseY: number, colors: ExportColors) {
  boxes.forEach((box) => {
    const border = box.selected ? (box.selectedColor ?? colors.info) : colors.border;
    svgRect(parts, baseX + box.x, baseY + box.y, box.width, box.height, colors.sunken, border, box.selected ? 2 : 1, 4);
    svgText(parts, box.text, baseX + box.x + box.width / 2, baseY + box.y + 8, colors.fg, 13, 700, "mono", "middle");
    svgRect(parts, baseX + box.x, baseY + box.y + box.height - 16, box.width, 16, box.footerColor, colors.border, 1, 0);
    svgText(parts, box.footer, baseX + box.x + box.width / 2, baseY + box.y + box.height - 15, colors.fg, 10, 700, "mono", "middle");
  });
}

async function loadExportImage(base64: string | null | undefined) {
  if (!base64) return null;
  const src = withImagePrefix(base64);
  const image = await loadImage(src);
  return {
    src,
    width: image.naturalWidth || image.width,
    height: image.naturalHeight || image.height,
  };
}

async function loadExportResources(outputResult: OutputResult, exportContext?: ResultExportContext): Promise<ExportResources> {
  const inputImageBase64 = exportContext?.inputImageBase64 ?? outputResult.input_image ?? null;

  const [inputImage, generatedImage, classificationImage] = await Promise.all([
    loadExportImage(inputImageBase64).catch(() => null),
    loadExportImage(outputResult.generated_image).catch(() => null),
    loadExportImage(outputResult.input_image).catch(() => null),
  ]);

  return { inputImage, generatedImage, classificationImage };
}

function buildFigureCaption(rows: Array<[string, string]>) {
  return rows.map(([label, value]) => `${label}: ${value}`).join("; ");
}

function drawFigureCaption(parts: SvgParts, ctx: CanvasRenderingContext2D, caption: string, x: number, y: number, width: number, colors: ExportColors) {
  if (!caption) return 0;
  svgLine(parts, x, y, x + width, y, colors.border);
  return drawWrappedText(parts, ctx, caption, x, y + 12, width, colors.fgMuted, 12, 400, "sans") + 16;
}

function measureInputSection(ctx: CanvasRenderingContext2D, width: number, resources: ExportResources, outputResult: OutputResult, exportContext?: ResultExportContext) {
  const inputImage = resources.inputImage;
  if (inputImage) {
    const size = containSize(inputImage.width, inputImage.height, width - 64, 520);
    return 74 + size.height + (exportContext?.inputImageFileName ? 24 : 0);
  }

  const text = exportContext?.inputText?.trim() || (outputResult.input_image ? "(image input)" : "(empty input)");
  const lines = wrapText(ctx, text, width - 64, 14, 400, "mono");
  return 76 + Math.max(192, lines.length * 21 + 32);
}

function drawInputSection(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, resources: ExportResources, outputResult: OutputResult, exportContext: ResultExportContext | undefined, colors: ExportColors) {
  svgRect(parts, x, y, width, height, colors.fill, colors.border);
  svgText(parts, "(a) Input", x + 16, y + 16, colors.fg, 16, 700, "sans");

  const contentX = x + 16;
  const contentY = y + 52;
  const contentWidth = width - 32;
  const inputImage = resources.inputImage;

  if (inputImage) {
    const size = containSize(inputImage.width, inputImage.height, contentWidth - 32, 520);
    const boxHeight = size.height + 32 + (exportContext?.inputImageFileName ? 24 : 0);
    svgRect(parts, contentX, contentY, contentWidth, boxHeight, colors.sunken);
    svgImage(parts, inputImage.src, contentX + (contentWidth - size.width) / 2, contentY + 16, size.width, size.height);
    if (exportContext?.inputImageFileName) {
      svgText(parts, exportContext.inputImageFileName, contentX + 16, contentY + size.height + 24, colors.fgSubtle, 12, 400, "mono");
    }
    return;
  }

  const text = exportContext?.inputText?.trim() || (outputResult.input_image ? "(image input)" : "(empty input)");
  svgRect(parts, contentX, contentY, contentWidth, height - 68, colors.sunken);
  drawWrappedText(parts, ctx, text, contentX + 16, contentY + 16, contentWidth - 32, colors.fg, 14, 400, "mono");
}

function measureTextClassification(ctx: CanvasRenderingContext2D, width: number, outputResult: OutputResult, hideSpecialTokens: boolean) {
  const rawTokens = outputResult.tokens || [];
  const rawScores: unknown[] = Array.isArray(outputResult.scores) ? outputResult.scores : [];
  const visible = rawTokens
    .map((token, index) => ({ token, score: typeof rawScores[index] === "number" ? rawScores[index] : 0, isSpecial: !!outputResult.special_tokens_mask?.[index] }))
    .filter((entry) => !(hideSpecialTokens && entry.isSpecial));
  const specs = visible.map((entry) => ({ text: tokenDisplayText(entry.token), footer: formatPercent(entry.score), footerColor: "transparent" }));
  const layout = layoutTokenBoxes(ctx, specs, width - 40);
  return layout.height + 130;
}

function drawTextClassification(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, outputResult: OutputResult, hideSpecialTokens: boolean, colorScaleMode: ColorScaleMode, colors: ExportColors, tutorialInteraction?: TutorialOutputInteraction) {
  const rawTokens = outputResult.tokens || [];
  const rawScores: unknown[] = Array.isArray(outputResult.scores) ? outputResult.scores : [];
  const visible = rawTokens
    .map((token, index) => ({ token, score: typeof rawScores[index] === "number" ? rawScores[index] : 0, isSpecial: !!outputResult.special_tokens_mask?.[index], rawIndex: index }))
    .filter((entry) => !(hideSpecialTokens && entry.isSpecial));
  const maxAbs = visible.reduce((max, entry) => Math.max(max, Math.abs(entry.score)), 0);
  const scale = colorScaleMode === "relative" && maxAbs > 0 ? 1 / maxAbs : 1;
  const sumAbs = visible.reduce((sum, entry) => sum + Math.abs(entry.score), 0);
  const specs = visible.map((entry) => ({
    text: tokenDisplayText(entry.token),
    footer: formatPercent(sumAbs > 0 ? entry.score / sumAbs : 0),
    footerColor: getScoreColor(entry.score * scale),
    selected: tutorialInteraction?.classificationTokenIndex === entry.rawIndex,
    selectedColor: colors.info,
  }));
  const layout = layoutTokenBoxes(ctx, specs, width - 40);

  svgRect(parts, x, y, width, layout.height + 40, colors.sunken, colors.border, 1, 6);
  drawTokenBoxes(parts, layout.boxes, x + 20, y + 20, colors);

  const rawLabel = typeof outputResult.predicted_token === "string" ? outputResult.predicted_token.trim() : "";
  const fallbackLabel = outputResult.target_id === 0 ? "NEGATIVE" : outputResult.target_id === 1 ? "POSITIVE" : "UNKNOWN";
  const label = rawLabel && !rawLabel.startsWith("[") ? rawLabel : fallbackLabel;
  const labelY = y + layout.height + 56;
  svgRect(parts, x, labelY, width, 58, colors.sunken, colors.border, 1, 6);
  svgText(parts, "Predicted Class:", x + width / 2 - 78, labelY + 18, colors.fgFaint, 14, 700, "mono", "end");
  svgRect(parts, x + width / 2 - 62, labelY + 17, Math.min(220, Math.max(90, label.length * 10 + 28)), 26, colors.infoSoft, undefined, 1, 4);
  svgText(parts, label.toUpperCase(), x + width / 2 - 48, labelY + 19, colors.info, 14, 700, "mono");
}

function measureTextGeneration(ctx: CanvasRenderingContext2D, width: number, outputResult: OutputResult, hideSpecialTokens: boolean, hideTemplateTokens: boolean) {
  const steps = getTextGenerationSteps(outputResult.scores);
  const inputTokens = steps[0]?.context_tokens ?? [];
  const inputSpecs = inputTokens
    .map((token, index) => ({ token, index }))
    .filter(({ index }) => {
      if (outputResult.input_special_mask?.[index]) return !hideSpecialTokens;
      if (outputResult.input_template_mask?.[index]) return !hideTemplateTokens;
      return true;
    })
    .map(({ token }) => ({ text: tokenDisplayText(token), footer: "-", footerColor: "transparent" }));
  const outputSpecs = steps
    .map((step, index) => ({ step, index }))
    .filter(({ index }) => !(hideSpecialTokens && outputResult.output_special_mask?.[index]))
    .map(({ step }) => ({ text: tokenDisplayText(step.generated_token), footer: formatPercent(step.probability), footerColor: "rgba(128, 128, 128, 0.15)" }));
  const inputLayout = layoutTokenBoxes(ctx, inputSpecs, width);
  const outputLayout = layoutTokenBoxes(ctx, outputSpecs, width);
  return inputLayout.height + outputLayout.height + 92;
}

function drawTextGeneration(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, outputResult: OutputResult, hideSpecialTokens: boolean, hideTemplateTokens: boolean, colorScaleMode: ColorScaleMode, selection: TextGenerationSelection, colors: ExportColors) {
  const steps = getTextGenerationSteps(outputResult.scores);
  const inputTokens = steps[0]?.context_tokens ?? [];
  const outputTokens = steps.map((step) => step.generated_token);

  const isInputHidden = (index: number) => {
    if (outputResult.input_special_mask?.[index]) return hideSpecialTokens;
    if (outputResult.input_template_mask?.[index]) return hideTemplateTokens;
    return false;
  };
  const getOutputData = (outIdx: number) => {
    const step = steps[outIdx];
    if (selection?.selectedType === "output" && selection.selectedIndex === outIdx) {
      return { value: step.probability, color: "rgba(128, 128, 128, 0.28)", label: "CONF" };
    }
    if (selection?.selectedType === "output") {
      if (outIdx < selection.selectedIndex) {
        const targetStep = steps[selection.selectedIndex];
        const score = targetStep.attribution_scores[inputTokens.length + outIdx] || 0;
        return { value: score, color: getScoreColor(scaleScore(score, targetStep.attribution_scores, colorScaleMode)), label: "ATTR" };
      }
      return { value: 0, color: "transparent", label: "-" };
    }
    if (selection?.selectedType === "input") {
      const score = step.attribution_scores[selection.selectedIndex] || 0;
      const columnScores = steps.map((traceStep) => traceStep.attribution_scores[selection.selectedIndex] || 0);
      return { value: score, color: getScoreColor(scaleScore(score, columnScores, colorScaleMode)), label: "INFL" };
    }
    return { value: step.probability, color: "rgba(128, 128, 128, 0.15)", label: "PROB" };
  };
  const getInputData = (inIdx: number) => {
    if (selection?.selectedType === "input" && selection.selectedIndex === inIdx) {
      return { value: 1, color: "rgba(59, 130, 246, 0.5)", label: "SEL" };
    }
    if (selection?.selectedType === "output") {
      const targetStep = steps[selection.selectedIndex];
      const score = targetStep.attribution_scores[inIdx] || 0;
      return { value: score, color: getScoreColor(scaleScore(score, targetStep.attribution_scores, colorScaleMode)), label: "ATTR" };
    }
    return { value: 0, color: "transparent", label: "-" };
  };

  svgText(parts, "Input", x, y, colors.fgSubtle, 12, 700, "mono");
  const inputSpecs = inputTokens
    .map((token, index) => ({ token, index }))
    .filter(({ index }) => !isInputHidden(index))
    .map(({ token, index }) => {
      const data = getInputData(index);
      return {
        text: tokenDisplayText(token),
        footer: data.label === "-" ? "-" : formatPercent(data.value),
        footerColor: data.color,
        selected: selection?.selectedType === "input" && selection.selectedIndex === index,
        selectedColor: colors.info,
      };
    });
  const inputLayout = layoutTokenBoxes(ctx, inputSpecs, width);
  drawTokenBoxes(parts, inputLayout.boxes, x, y + 28, colors);

  const dividerY = y + inputLayout.height + 48;
  svgLine(parts, x, dividerY, x + width, dividerY, colors.border);
  svgText(parts, "Output", x, dividerY + 18, colors.fgSubtle, 12, 700, "mono");

  const outputSpecs = outputTokens
    .map((token, index) => ({ token, index }))
    .filter(({ index }) => !(hideSpecialTokens && outputResult.output_special_mask?.[index]))
    .map(({ token, index }) => {
      const data = getOutputData(index);
      return {
        text: tokenDisplayText(token),
        footer: data.label === "-" ? "-" : formatPercent(data.value),
        footerColor: data.color,
        selected: selection?.selectedType === "output" && selection.selectedIndex === index,
        selectedColor: colors.info,
      };
    });
  const outputLayout = layoutTokenBoxes(ctx, outputSpecs, width);
  drawTokenBoxes(parts, outputLayout.boxes, x, dividerY + 46, colors);
}

function measureImageGeneration(ctx: CanvasRenderingContext2D, width: number, outputResult: OutputResult, resources: ExportResources) {
  const image = resources.generatedImage;
  const imageSize = image ? containSize(image.width, image.height, Math.min(width - 32, 512), 512) : { width: Math.min(width - 32, 512), height: 320 };
  const tokenSpecs = (outputResult.tokens || []).map((token) => ({ text: tokenDisplayText(token), footer: "-", footerColor: "transparent" }));
  const tokenLayout = layoutTokenBoxes(ctx, tokenSpecs, width - 40);
  return imageSize.height + tokenLayout.height + 154;
}

function drawImageGeneration(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, outputResult: OutputResult, resources: ExportResources, selection: ImageGenerationSelection, colors: ExportColors) {
  const image = resources.generatedImage;
  const heatmaps = getHeatmaps(outputResult.scores);
  const imageSize = image ? containSize(image.width, image.height, Math.min(width - 32, 512), 512) : { width: Math.min(width - 32, 512), height: 320 };
  const panelHeight = imageSize.height + 66;

  svgRect(parts, x, y, width, panelHeight, colors.sunken, colors.border, 1, 6);
  svgText(parts, "Generated Image", x + 16, y + 16, colors.fgSubtle, 13, 700, "mono");
  const imageX = x + (width - imageSize.width) / 2;
  const imageY = y + 50;
  if (image) svgImage(parts, image.src, imageX, imageY, imageSize.width, imageSize.height);
  const aggregate = aggregateHeatmaps(heatmaps, selection.selectedTokenIndices);
  const heatmapUrl = aggregate ? createHeatmapDataUrl(aggregate) : "";
  if (heatmapUrl) svgImage(parts, heatmapUrl, imageX, imageY, imageSize.width, imageSize.height);
  if (!heatmapUrl && selection.hoveredCell) {
    const grid = 64;
    svgRect(
      parts,
      imageX + (selection.hoveredCell.x / grid) * imageSize.width,
      imageY + (selection.hoveredCell.y / grid) * imageSize.height,
      imageSize.width / grid,
      imageSize.height / grid,
      "rgba(255, 255, 255, 0.2)",
      "rgba(255, 255, 255, 0.6)"
    );
  }

  const promptY = y + panelHeight + 24;
  const tokenSpecs = (outputResult.tokens || []).map((token, index) => {
    const score = selection.hoveredCell && heatmaps[index]?.raw_matrix
      ? heatmaps[index].raw_matrix[selection.hoveredCell.y]?.[selection.hoveredCell.x] ?? 0
      : 0;
    return {
      text: tokenDisplayText(token),
      footer: selection.hoveredCell ? formatPercent(score) : "-",
      footerColor: selection.hoveredCell ? getImageScoreColor(score) : "transparent",
      selected: selection.selectedTokenIndices.includes(index),
      selectedColor: colors.accent,
    };
  });
  const tokenLayout = layoutTokenBoxes(ctx, tokenSpecs, width - 40);
  svgRect(parts, x, promptY, width, tokenLayout.height + 40, colors.sunken, colors.border, 1, 6);
  svgText(parts, "Input Prompt", x + 16, promptY + 14, colors.fgSubtle, 13, 700, "mono");
  drawTokenBoxes(parts, tokenLayout.boxes, x + 20, promptY + 42, colors);
}

function measureImageClassification(width: number, resources: ExportResources) {
  const image = resources.classificationImage;
  const imageSize = image ? containSize(image.width, image.height, Math.min(width - 32, 512), 512) : { width: Math.min(width - 32, 512), height: 320 };
  return imageSize.height + 148;
}

function drawImageClassification(parts: SvgParts, x: number, y: number, width: number, outputResult: OutputResult, resources: ExportResources, showOverlay: boolean, colors: ExportColors) {
  const image = resources.classificationImage;
  const heatmaps = getHeatmaps(outputResult.scores);
  const imageSize = image ? containSize(image.width, image.height, Math.min(width - 32, 512), 512) : { width: Math.min(width - 32, 512), height: 320 };

  svgRect(parts, x, y, width, imageSize.height + 66, colors.sunken, colors.border, 1, 6);
  svgText(parts, "Input Image", x + 16, y + 16, colors.fgSubtle, 13, 700, "mono");
  const imageX = x + (width - imageSize.width) / 2;
  const imageY = y + 50;
  if (image) svgImage(parts, image.src, imageX, imageY, imageSize.width, imageSize.height);
  if (showOverlay && heatmaps[0]?.raw_matrix) {
    const heatmapUrl = createHeatmapDataUrl(heatmaps[0].raw_matrix);
    if (heatmapUrl) svgImage(parts, heatmapUrl, imageX, imageY, imageSize.width, imageSize.height);
  }

  const rawLabel = typeof outputResult.predicted_token === "string" ? outputResult.predicted_token.trim() : "";
  const fallbackLabel = outputResult.target_id === 0 ? "NEGATIVE" : outputResult.target_id === 1 ? "POSITIVE" : "UNKNOWN";
  const label = rawLabel && !rawLabel.startsWith("[") ? rawLabel : fallbackLabel;
  const labelY = y + imageSize.height + 90;
  svgRect(parts, x, labelY, width, 58, colors.sunken, colors.border, 1, 6);
  svgText(parts, "Predicted Class:", x + width / 2 - 78, labelY + 18, colors.fgFaint, 14, 700, "mono", "end");
  svgRect(parts, x + width / 2 - 62, labelY + 17, Math.min(220, Math.max(90, label.length * 10 + 28)), 26, colors.infoSoft, undefined, 1, 4);
  svgText(parts, label.toUpperCase(), x + width / 2 - 48, labelY + 19, colors.info, 14, 700, "mono");
}

function measureOutputVisualization(ctx: CanvasRenderingContext2D, width: number, outputResult: OutputResult, resources: ExportResources, hideSpecialTokens: boolean, hideTemplateTokens: boolean) {
  const { isImage, isTextGeneration, isImageClassification } = getResultKind(outputResult);
  if (isImage) return measureImageGeneration(ctx, width, outputResult, resources);
  if (isTextGeneration) return measureTextGeneration(ctx, width, outputResult, hideSpecialTokens, hideTemplateTokens);
  if (isImageClassification) return measureImageClassification(width, resources);
  return measureTextClassification(ctx, width, outputResult, hideSpecialTokens);
}

function drawOutputVisualization(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, outputResult: OutputResult, resources: ExportResources, hideSpecialTokens: boolean, hideTemplateTokens: boolean, colorScaleMode: ColorScaleMode, textSelection: TextGenerationSelection, imageSelection: ImageGenerationSelection, imageClassificationOverlayVisible: boolean, colors: ExportColors, tutorialInteraction?: TutorialOutputInteraction) {
  const { isImage, isTextGeneration, isImageClassification } = getResultKind(outputResult);
  if (isImage) {
    drawImageGeneration(parts, ctx, x, y, width, outputResult, resources, imageSelection, colors);
    return;
  }
  if (isTextGeneration) {
    drawTextGeneration(parts, ctx, x, y, width, outputResult, hideSpecialTokens, hideTemplateTokens, colorScaleMode, textSelection, colors);
    return;
  }
  if (isImageClassification) {
    drawImageClassification(parts, x, y, width, outputResult, resources, imageClassificationOverlayVisible, colors);
    return;
  }
  drawTextClassification(parts, ctx, x, y, width, outputResult, hideSpecialTokens, colorScaleMode, colors, tutorialInteraction);
}

function measureOutputSection(ctx: CanvasRenderingContext2D, width: number, outputResult: OutputResult, resources: ExportResources, hideSpecialTokens: boolean, hideTemplateTokens: boolean) {
  return 122 + measureOutputVisualization(ctx, width - 32, outputResult, resources, hideSpecialTokens, hideTemplateTokens);
}

function drawOutputSection(parts: SvgParts, ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, outputResult: OutputResult, resources: ExportResources, taskTitle: string, hideSpecialTokens: boolean, hideTemplateTokens: boolean, colorScaleMode: ColorScaleMode, textSelection: TextGenerationSelection, imageSelection: ImageGenerationSelection, imageClassificationOverlayVisible: boolean, colors: ExportColors, tutorialInteraction?: TutorialOutputInteraction) {
  svgRect(parts, x, y, width, height, colors.fill, colors.border);
  svgText(parts, "(b) Attribution output", x + 16, y + 16, colors.fg, 16, 700, "sans");
  svgLine(parts, x + 16, y + 56, x + width - 16, y + 56, colors.border);
  svgText(parts, `Task: ${taskTitle}`, x + 16, y + 66, colors.fgSubtle, 12, 400, "sans");
  drawOutputVisualization(
    parts,
    ctx,
    x + 16,
    y + 106,
    width - 32,
    outputResult,
    resources,
    hideSpecialTokens,
    hideTemplateTokens,
    colorScaleMode,
    textSelection,
    imageSelection,
    imageClassificationOverlayVisible,
    colors,
    tutorialInteraction
  );
}

async function buildNativeExportSvg({
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
}: {
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
}): Promise<SvgLayout> {
  const ctx = getMeasureContext();
  const colors = getExportColors();
  const resources = await loadExportResources(outputResult, exportContext);
  const width = layout === "side-by-side" ? 1200 : 860;
  const margin = 44;
  const gap = 32;
  const contentWidth = width - margin * 2;
  const parts: SvgParts = [];
  let y = margin;

  const effectiveTextSelection = tutorialInteraction?.textGenerationSelection ?? textGenerationSelection;
  const effectiveImageSelection = tutorialInteraction?.imageSelection
    ? {
      selectedTokenIndices: tutorialInteraction.imageSelection.selectedTokenIndices ?? [],
      hoveredCell: tutorialInteraction.imageSelection.hoveredCell ?? null,
    }
    : imageGenerationSelection;

  const configRows = includeConfig
    ? buildConfigRows(
      outputResult,
      exportContext,
      taskTitle,
      hideSpecialTokens,
      hideTemplateTokens,
      colorScaleMode,
      effectiveTextSelection,
      effectiveImageSelection,
      imageClassificationOverlayVisible
    )
    : [];
  const caption = buildFigureCaption(configRows);

  svgRect(parts, 0, 0, width, 1, colors.surface);
  svgText(parts, `${taskTitle} attribution`, width / 2, y, colors.fg, 22, 700, "sans", "middle");
  y += 48;
  svgLine(parts, margin, y - 10, width - margin, y - 10, colors.border);

  if (layout === "side-by-side") {
    const columnWidth = (contentWidth - gap) / 2;
    const inputHeight = measureInputSection(ctx, columnWidth, resources, outputResult, exportContext);
    const outputHeight = measureOutputSection(ctx, columnWidth, outputResult, resources, hideSpecialTokens, hideTemplateTokens);
    const sectionHeight = Math.max(inputHeight, outputHeight);
    drawInputSection(parts, ctx, margin, y, columnWidth, sectionHeight, resources, outputResult, exportContext, colors);
    drawOutputSection(
      parts,
      ctx,
      margin + columnWidth + gap,
      y,
      columnWidth,
      sectionHeight,
      outputResult,
      resources,
      taskTitle,
      hideSpecialTokens,
      hideTemplateTokens,
      colorScaleMode,
      effectiveTextSelection,
      effectiveImageSelection,
      imageClassificationOverlayVisible,
      colors,
      tutorialInteraction
    );
    y += sectionHeight + 28;
  } else {
    const inputHeight = measureInputSection(ctx, contentWidth, resources, outputResult, exportContext);
    drawInputSection(parts, ctx, margin, y, contentWidth, inputHeight, resources, outputResult, exportContext, colors);
    y += inputHeight + gap;
    const outputHeight = measureOutputSection(ctx, contentWidth, outputResult, resources, hideSpecialTokens, hideTemplateTokens);
    drawOutputSection(
      parts,
      ctx,
      margin,
      y,
      contentWidth,
      outputHeight,
      outputResult,
      resources,
      taskTitle,
      hideSpecialTokens,
      hideTemplateTokens,
      colorScaleMode,
      effectiveTextSelection,
      effectiveImageSelection,
      imageClassificationOverlayVisible,
      colors,
      tutorialInteraction
    );
    y += outputHeight + 28;
  }

  if (caption) {
    y += drawFigureCaption(parts, ctx, caption, margin, y, contentWidth, colors);
  }

  y += margin;

  const height = Math.ceil(y);
  const svgTextValue = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect x="0" y="0" width="${width}" height="${height}" fill="${escapeXml(colors.surface)}"/>`,
    ...parts,
    "</svg>",
  ].join("");

  return { svgText: svgTextValue, width, height };
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
    if (isExporting) return;

    setIsExporting(true);
    setExportError(null);

    try {
      await document.fonts?.ready;
      await waitForNextFrame();

      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      const stem = `lumixai-${sanitizeFilePart(taskTitle)}-${timestamp}`;
      const { svgText, width, height } = await buildNativeExportSvg({
        outputResult,
        exportContext,
        taskTitle,
        layout: exportLayout,
        includeConfig: includeExportConfig,
        hideSpecialTokens,
        hideTemplateTokens,
        colorScaleMode,
        textGenerationSelection,
        imageGenerationSelection,
        imageClassificationOverlayVisible,
        tutorialInteraction,
      });
      const backgroundColor = getExportColors().surface;

      if (exportFormat === "svg") {
        downloadBlob(new Blob([svgText], { type: "image/svg+xml;charset=utf-8" }), `${stem}.svg`);
        return;
      }

      if (exportFormat === "jpg") {
        const dataUrl = await exportSvgToDataUrl(svgText, width, height, "image/jpeg", backgroundColor, 2, 0.95);
        downloadBlob(await dataUrlToBlob(dataUrl), `${stem}.jpg`);
        return;
      }

      const dataUrl = await exportSvgToDataUrl(svgText, width, height, "image/png", backgroundColor, 2);

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

    </div>
  );
}

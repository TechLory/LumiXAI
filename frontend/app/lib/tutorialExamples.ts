import type { OutputResult } from "../components/panels/OutputPanel";
import type { JobHistoryItem, TutorialKind } from "../types";

export type TutorialExampleConfiguration = {
  sourceId: string;
  sourceName: string;
  modelName: string;
  attributorId: string;
  attributorName: string;
  detectedTask: string;
};

export interface TutorialExampleJob extends JobHistoryItem {
  status: "completed";
  is_builtin_example: true;
  tutorial_kind: TutorialKind;
  config: TutorialExampleConfiguration;
  payload: OutputResult;
}

type TextGenerationStep = {
  generated_token: string;
  probability: number;
  context_tokens: string[];
  attribution_scores: number[];
};

type ImageHeatmap = {
  image_base64: string;
  raw_matrix: number[][];
};

const GRID_SIZE = 64;
const EXAMPLE_SOURCE_ID = "huggingface";
const EXAMPLE_SOURCE_NAME = "Hugging Face Hub";
const CAPTUM_ATTRIBUTOR_ID = "captum_ig";
const CAPTUM_ATTRIBUTOR = "Integrated Gradients (Captum)";
const DAAM_ATTRIBUTOR_ID = "daam";
const DAAM_ATTRIBUTOR = "DAAM (Diffusion Attentive Attribution Maps)";

const EXAMPLE_IMAGE_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAB7UlEQVR42u2WOU4DMRiF30EoqSAECISwBSJFoqChoKGg4iDsS1jDzhk4BhJL2NeBKDQUCBaIo00Qsnvicn4xVga6WvG0vP7P0u2Bi3Tp04D5wVaZ86cBonZM6dBYu7cadA2d+40SM5fOA2SCyW7fJ3kdVDF0b5YsoumgCqOjqVLu2gKqOLuC3QuX9lFU0AVR2rl2i6aAqo4UoVru+gKKOLoKtzYRVNAFUf36o1dNAVUcaTXbu2iKaCKI71+ZxddAUUcPRt3dtEUUMWR2bi3i6aAKo7M5j2Dj8/3Mt/euI//yShC79YDA1GAUYS+4iMDUYBRxBWogCLQv/3EQBRgFGFg54mBl89UwyjCwO4zA1mAUITBvRcGogCjCNm9FwaiAKMI2X2PgSxAKMLQgcdAfIUYRRg+fGUgCjCKkDt6YyAKMIrcFxgpHjMQBRhFLAHxFYoFjAq0T074uCcQjP4bNwTE0UWNPwmEn4UZgZqjV0yg8wrpH0ckgT+NHhAiUMdudQrUN3qIQJQNVSYwPjpVoNoExkdvjEAAjI/uI75CLAHGvo0TaB6bDWia2g+IuC4KGNw/AIxNy5+iAOOwYoGq9dFsLhzDAsZPpaaA4TsQC9gWcP4OxAKW1+Hi0NxLHAv8j3+hxt2BWCB+haKt/wA0/iqaDX+ggQAAAABJRU5ErkJggg==";

const roundScore = (value: number) => Number(value.toFixed(4));

const gaussian = (x: number, y: number, centerX: number, centerY: number, radius: number) => {
  const dx = (x - centerX) / radius;
  const dy = (y - centerY) / radius;
  return Math.exp(-(dx * dx + dy * dy));
};

const buildImageHeatmaps = (tokens: readonly string[]): ImageHeatmap[] => {
  const centers = [
    [48, 13, 9],
    [31, 35, 13],
    [32, 34, 9],
    [33, 43, 11],
    [22, 44, 16],
    [48, 50, 17],
    [42, 14, 10],
    [15, 18, 16],
  ];

  return tokens.map((_, tokenIndex) => {
    const [centerX, centerY, radius] = centers[tokenIndex] ?? [32, 32, 18];

    return {
      image_base64: "",
      raw_matrix: Array.from({ length: GRID_SIZE }, (_, y) =>
        Array.from({ length: GRID_SIZE }, (_, x) => {
          const primary = gaussian(x, y, centerX, centerY, radius);
          const secondary = gaussian(x, y, 64 - centerX, Math.max(8, centerY - 6), radius * 1.35);
          const ripple = Math.max(0, Math.sin((x + tokenIndex * 7) / 6) * Math.cos((y + tokenIndex * 3) / 8)) * 0.06;
          return roundScore(Math.min(1, primary + secondary * 0.28 + ripple));
        })
      ),
    };
  });
};

const buildTextGenerationTrace = (): TextGenerationStep[] => {
  const inputTokens = ["Explain", "Ġwhy", "Ġattribution", "Ġheatmaps", "Ġhelp", "Ġresearchers", ":"];
  const outputTokens = ["Ġthey", "Ġreveal", "Ġwhich", "Ġwords", "Ġguide", "Ġeach", "Ġprediction", "."];
  const probabilities = [0.74, 0.68, 0.63, 0.71, 0.59, 0.61, 0.66, 0.82];
  const inputInfluence = [0.08, 0.11, 0.31, 0.37, 0.18, 0.27, 0.12];

  return outputTokens.map((generatedToken, outputIndex) => {
    const contextLength = inputTokens.length + outputIndex;
    const attribution_scores = Array.from({ length: contextLength }, (_, contextIndex) => {
      if (contextIndex < inputTokens.length) {
        const pulse = Math.cos((outputIndex + 1) * (contextIndex + 2)) * 0.025;
        const decay = 1 - outputIndex * 0.018;
        return roundScore(inputInfluence[contextIndex] * decay + pulse);
      }

      const previousOutputIndex = contextIndex - inputTokens.length;
      const distance = outputIndex - previousOutputIndex;
      return roundScore(Math.max(0.04, 0.2 - distance * 0.028));
    });

    return {
      generated_token: generatedToken,
      probability: probabilities[outputIndex],
      context_tokens: inputTokens,
      attribution_scores,
    };
  });
};

const imageTokens = ["pixel", "art", "lighthouse", "sunset", "glowing", "ocean", "warm", "sky"];

export const tutorialExampleJobs: readonly TutorialExampleJob[] = [
  {
    id: "example-text-classification",
    status: "completed",
    prompt: "The interface is surprisingly clear and the heatmaps make every decision easier to inspect.",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "distilbert-base-uncased-finetuned-sst-2-english",
    attributor_name: CAPTUM_ATTRIBUTOR,
    created_at: "2026-01-10T09:00:00.000Z",
    execution_time_sec: 1.42,
    is_builtin_example: true,
    tutorial_kind: "text-classification",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "distilbert-base-uncased-finetuned-sst-2-english",
      attributorId: CAPTUM_ATTRIBUTOR_ID,
      attributorName: CAPTUM_ATTRIBUTOR,
      detectedTask: "text-classification",
    },
    payload: {
      target_id: 1,
      predicted_token: "POSITIVE",
      tokens: ["[CLS]", "the", "interface", "is", "surprisingly", "clear", "and", "the", "heat", "##maps", "make", "every", "decision", "easier", "to", "inspect", ".", "[SEP]"],
      scores: [0.03, 0.08, 0.36, 0.09, 0.44, 0.72, 0.12, 0.05, 0.48, 0.51, 0.42, 0.22, 0.39, 0.57, 0.15, 0.33, 0.04, 0.02],
      special_tokens_mask: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true],
    },
  },
  {
    id: "example-text-generation",
    status: "completed",
    prompt: "Explain why attribution heatmaps help researchers:",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "gpt2-large",
    attributor_name: CAPTUM_ATTRIBUTOR,
    created_at: "2026-01-10T09:05:00.000Z",
    execution_time_sec: 4.86,
    is_builtin_example: true,
    tutorial_kind: "text-generation",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "gpt2-large",
      attributorId: CAPTUM_ATTRIBUTOR_ID,
      attributorName: CAPTUM_ATTRIBUTOR,
      detectedTask: "text-generation",
    },
    payload: {
      target_id: "text_generation",
      predicted_token: undefined,
      tokens: [],
      scores: buildTextGenerationTrace(),
      input_special_mask: [false, false, false, false, false, false, false],
      output_special_mask: [false, false, false, false, false, false, false, false],
      input_template_mask: [false, false, false, false, false, false, false],
    },
  },
  {
    id: "example-txt2img-generation",
    status: "completed",
    prompt: "pixel art lighthouse at sunset with glowing ocean and warm sky",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "runwayml/stable-diffusion-v1-5",
    attributor_name: DAAM_ATTRIBUTOR,
    created_at: "2026-01-10T09:10:00.000Z",
    execution_time_sec: 18.3,
    is_builtin_example: true,
    tutorial_kind: "txt2img-generation",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "runwayml/stable-diffusion-v1-5",
      attributorId: DAAM_ATTRIBUTOR_ID,
      attributorName: DAAM_ATTRIBUTOR,
      detectedTask: "text-to-image",
    },
    payload: {
      target_id: "image_generation",
      predicted_token: undefined,
      tokens: imageTokens,
      scores: buildImageHeatmaps(imageTokens),
      generated_image: EXAMPLE_IMAGE_BASE64,
    },
  },
];

export const tutorialExampleSummaries: JobHistoryItem[] = tutorialExampleJobs.map((job) => ({
  id: job.id,
  status: job.status,
  prompt: job.prompt,
  source_name: job.source_name,
  model_name: job.model_name,
  attributor_name: job.attributor_name,
  created_at: job.created_at,
  execution_time_sec: job.execution_time_sec,
  is_builtin_example: job.is_builtin_example,
  tutorial_kind: job.tutorial_kind,
}));

export const getTutorialExampleJob = (jobId: string) =>
  tutorialExampleJobs.find((job) => job.id === jobId) ?? null;

export const getTutorialExampleForKind = (tutorialKind: TutorialKind) =>
  tutorialExampleJobs.find((job) => job.tutorial_kind === tutorialKind) ?? null;

export const getTutorialExampleConfiguration = (tutorialKind: TutorialKind) =>
  getTutorialExampleForKind(tutorialKind)?.config ?? null;

export const isTutorialExampleJobId = (jobId: string) =>
  tutorialExampleJobs.some((job) => job.id === jobId);

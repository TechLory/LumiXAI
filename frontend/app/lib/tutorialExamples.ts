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

export interface TutorialExampleMeta extends JobHistoryItem {
  status: "completed";
  is_builtin_example: true;
  pinned: true;
  tutorial_kind: TutorialKind;
  config: TutorialExampleConfiguration;
  payloadUrl: string;
}

const EXAMPLE_SOURCE_ID = "huggingface";
const EXAMPLE_SOURCE_NAME = "Hugging Face Hub";
const CAPTUM_ATTRIBUTOR_ID = "captum_ig";
const CAPTUM_ATTRIBUTOR = "Integrated Gradients (Captum)";
const DEEPLIFT_ATTRIBUTOR_ID = "captum_deeplift";
const DEEPLIFT_ATTRIBUTOR = "DeepLift (Captum)";
const DAAM_ATTRIBUTOR_ID = "daam";
const DAAM_ATTRIBUTOR = "DAAM (Diffusion Attentive Attribution Maps)";
const GRADCAM_ATTRIBUTOR_ID = "captum_gradcam";
const GRADCAM_ATTRIBUTOR = "Grad-CAM (Captum)";

// Each example below is a real, previously-run job (prompt, model, and output all captured
// from an actual backend inference) rather than a synthetic fixture. The heavy payload is
// shipped alongside the app as a static JSON file and fetched lazily so the JS bundle stays
// small; only the lightweight metadata below is loaded eagerly.
export const tutorialExampleMetas: readonly TutorialExampleMeta[] = [
  {
    id: "example-text-classification",
    status: "completed",
    prompt: "I like this movie a lot!",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest",
    attributor_name: CAPTUM_ATTRIBUTOR,
    created_at: "2026-07-06T18:10:55.264Z",
    execution_time_sec: 0.25,
    is_builtin_example: true,
    pinned: true,
    tutorial_kind: "text-classification",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "cardiffnlp/twitter-roberta-base-sentiment-latest",
      attributorId: CAPTUM_ATTRIBUTOR_ID,
      attributorName: CAPTUM_ATTRIBUTOR,
      detectedTask: "text-classification",
    },
    payloadUrl: "/tutorial-examples/text-classification.json",
  },
  {
    id: "example-text-generation",
    status: "completed",
    prompt: "Which is the capital of Italy?",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "Qwen/Qwen3-0.6B",
    attributor_name: DEEPLIFT_ATTRIBUTOR,
    created_at: "2026-07-07T05:54:49.615Z",
    execution_time_sec: 1.84,
    is_builtin_example: true,
    pinned: true,
    tutorial_kind: "text-generation",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "Qwen/Qwen3-0.6B",
      attributorId: DEEPLIFT_ATTRIBUTOR_ID,
      attributorName: DEEPLIFT_ATTRIBUTOR,
      detectedTask: "text-generation",
    },
    payloadUrl: "/tutorial-examples/text-generation.json",
  },
  {
    id: "example-txt2img-generation",
    status: "completed",
    prompt: "a painting of an elephant in the Baroque style",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "stabilityai/stable-diffusion-xl-base-1.0",
    attributor_name: DAAM_ATTRIBUTOR,
    created_at: "2026-07-06T18:13:00.713Z",
    execution_time_sec: 4.16,
    is_builtin_example: true,
    pinned: true,
    tutorial_kind: "txt2img-generation",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "stabilityai/stable-diffusion-xl-base-1.0",
      attributorId: DAAM_ATTRIBUTOR_ID,
      attributorName: DAAM_ATTRIBUTOR,
      detectedTask: "text-to-image",
    },
    payloadUrl: "/tutorial-examples/txt2img-generation.json",
  },
  {
    id: "example-image-classification",
    status: "completed",
    prompt: "mnist_digit_3.png",
    source_name: EXAMPLE_SOURCE_NAME,
    model_name: "farleyknight-org-username/vit-base-mnist",
    attributor_name: GRADCAM_ATTRIBUTOR,
    created_at: "2026-07-07T08:37:55.734Z",
    execution_time_sec: 0.13,
    is_builtin_example: true,
    pinned: true,
    tutorial_kind: "image-classification",
    config: {
      sourceId: EXAMPLE_SOURCE_ID,
      sourceName: EXAMPLE_SOURCE_NAME,
      modelName: "farleyknight-org-username/vit-base-mnist",
      attributorId: GRADCAM_ATTRIBUTOR_ID,
      attributorName: GRADCAM_ATTRIBUTOR,
      detectedTask: "image-classification",
    },
    payloadUrl: "/tutorial-examples/image-classification.json",
  },
];

const payloadCache = new Map<string, Promise<OutputResult>>();

export const loadTutorialExamplePayload = (meta: TutorialExampleMeta): Promise<OutputResult> => {
  const cached = payloadCache.get(meta.id);
  if (cached) return cached;

  const request = fetch(meta.payloadUrl).then((res) => {
    if (!res.ok) throw new Error(`Failed to load tutorial example payload: ${meta.id}`);
    return res.json() as Promise<OutputResult>;
  });

  payloadCache.set(meta.id, request);
  return request;
};

export const tutorialExampleSummaries: JobHistoryItem[] = tutorialExampleMetas.map(
  ({ config, payloadUrl, ...summary }) => summary
);

export const getTutorialExampleMeta = (jobId: string) =>
  tutorialExampleMetas.find((meta) => meta.id === jobId) ?? null;

export const getTutorialExampleForKind = (tutorialKind: TutorialKind) =>
  tutorialExampleMetas.find((meta) => meta.tutorial_kind === tutorialKind) ?? null;

export const getTutorialExampleConfiguration = (tutorialKind: TutorialKind) =>
  getTutorialExampleForKind(tutorialKind)?.config ?? null;

export const isTutorialExampleJobId = (jobId: string) =>
  tutorialExampleMetas.some((meta) => meta.id === jobId);

import type { TutorialKind, TutorialOutputInteraction } from "../types";

export type TutorialTarget = "system" | "configuration" | "input" | "output" | "history";

export type TutorialFocusTarget =
  | "configuration-source"
  | "configuration-model"
  | "configuration-attributor"
  | "configuration-action"
  | "input-editor"
  | "input-action"
  | "output-result"
  | "output-classification-tokens"
  | "output-classification-label"
  | "output-generation-input"
  | "output-generation-output"
  | "output-image"
  | "output-image-prompt"
  | "history-item";

export type TutorialHydrationPhase =
  | "source"
  | "model"
  | "attributor"
  | "configuration"
  | "input"
  | "result";

export type TutorialStep = {
  title: string;
  body: string;
  target: TutorialTarget;
  focusTarget?: TutorialFocusTarget;
  phase: TutorialHydrationPhase;
  outputInteraction?: TutorialOutputInteraction;
};

const outputDescriptions: Record<TutorialKind, string> = {
  "text-classification": `The CardiffNLP Twitter RoBERTa model predicts POSITIVE for "I like this movie a lot!". Token color intensity shows which input pieces pushed most strongly toward that sentiment label.`,
  "text-generation": `Each generated token is shown with its model probability. Qwen3-0.6B answers "The capital of Italy is **Rome**." from the prompt above, with "Rome" split into the token fragments "R" and "ome".`,
  "txt2img-generation": `The SDXL output pairs the generated Baroque elephant image with one spatial attribution matrix per prompt token, showing where each word influenced the pixels.`,
};

const outputOverviewFocusTargets: Record<TutorialKind, TutorialFocusTarget> = {
  "text-classification": "output-result",
  "text-generation": "output-generation-output",
  "txt2img-generation": "output-image",
};

const inputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Inspect input attributions",
    body: `For classification, the attribution lives on the input tokens themselves. "like" carries the strongest visible signal toward POSITIVE for this sentence, and clicking any token reveals its exact score.`,
    target: "output",
    focusTarget: "output-classification-tokens",
    phase: "result",
    outputInteraction: { classificationTokenIndex: 2 },
  },
  "text-generation": {
    title: "Input to output heatmap",
    body: `Selecting an input token answers: which generated tokens did it influence most? Select "Italy" and compare how it contributes to the answer tokens that name Rome.`,
    target: "output",
    focusTarget: "output-generation-input",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "input", selectedIndex: 8 } },
  },
  "txt2img-generation": {
    title: "Prompt token to image heatmap",
    body: `Selecting a prompt token overlays its spatial heatmap on the image. "elephant" lights up the generated subject, while the style tokens explain more of the painted look around it.`,
    target: "output",
    focusTarget: "output-image-prompt",
    phase: "result",
    outputInteraction: { imageSelection: { selectedTokenIndices: [4] } },
  },
};

const outputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Output target context",
    body: "The label above the tokens names the predicted class. Everything shown here explains why this run landed on POSITIVE for the sentence, not negative or neutral.",
    target: "output",
    focusTarget: "output-classification-label",
    phase: "result",
  },
  "text-generation": {
    title: "Output to input heatmap",
    body: `Selecting a generated token flips the view: which earlier words contributed to it? Select the "ome" fragment to trace the completion of Rome back through the prompt.`,
    target: "output",
    focusTarget: "output-generation-output",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "output", selectedIndex: 7 } },
  },
  "txt2img-generation": {
    title: "Image region to prompt heatmap",
    body: `Hovering a region of the image asks the reverse question: which prompt tokens were active there? This cell sits over the elephant, so the subject token rises among the token scores.`,
    target: "output",
    focusTarget: "output-image",
    phase: "result",
    outputInteraction: { imageSelection: { hoveredCell: { x: 32, y: 34 } } },
  },
};

export const getTutorialSteps = (tutorialKind: TutorialKind): TutorialStep[] => [
  {
    title: "Choose a source",
    body: "Every run starts here: pick where the model lives. Hugging Face Hub is the only source LumiXAI ships with today.",
    target: "configuration",
    focusTarget: "configuration-source",
    phase: "source",
  },
  {
    title: "Choose a model",
    body: "Paste in any compatible Hugging Face model id. This field now holds the model that actually produced the result you are about to see.",
    target: "configuration",
    focusTarget: "configuration-model",
    phase: "model",
  },
  {
    title: "Choose an attributor",
    body: "Attributors are the algorithms that compute the explanation. Captum methods cover classifiers and text generators; DAAM computes spatial maps for diffusion image models.",
    target: "configuration",
    focusTarget: "configuration-attributor",
    phase: "attributor",
  },
  {
    title: "Load configuration",
    body: "In a live session this button loads the model and attributor onto the backend. Here it just confirms readiness, since the example result already exists.",
    target: "configuration",
    focusTarget: "configuration-action",
    phase: "configuration",
  },
  {
    title: "Add input text",
    body: "This is exactly what you would type or paste for a real run. The panel now holds the prompt behind the result you are about to open.",
    target: "input",
    focusTarget: "input-editor",
    phase: "input",
  },
  {
    title: "Run attribution",
    body: "In a live session this button starts inference on the backend. Here it jumps straight to the already-computed result, so you can see the output without waiting.",
    target: "input",
    focusTarget: "input-action",
    phase: "result",
  },
  {
    title: "Read the output",
    body: outputDescriptions[tutorialKind],
    target: "output",
    focusTarget: outputOverviewFocusTargets[tutorialKind],
    phase: "result",
  },
  inputHeatmapSteps[tutorialKind],
  outputHeatmapSteps[tutorialKind],
  {
    title: "Find it in history",
    body: "This run is pinned at the top of Job History so it is easy to find again. Pin any of your own runs the same way to keep them close by. Every real run you explain lands in this same list.",
    target: "history",
    focusTarget: "history-item",
    phase: "result",
  },
];

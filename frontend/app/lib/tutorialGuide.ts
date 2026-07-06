import type { TutorialKind, TutorialOutputInteraction } from "../types";

export type TutorialTarget = "system" | "configuration" | "input" | "output" | "history";

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
  phase: TutorialHydrationPhase;
  outputInteraction?: TutorialOutputInteraction;
};

const outputDescriptions: Record<TutorialKind, string> = {
  "text-classification": 'The model\'s real prediction for "I like this movie a lot!" is NEGATIVE — a reminder that sentiment models don\'t always match intuition. Token color intensity is the attribution value: stronger tokens pushed harder toward that class.',
  "text-generation": 'Each generated token is shown with its model probability. The full reply — "The capital of Italy is Rome." — was produced by Qwen2.5-3B-Instruct from the single prompt above.',
  "txt2img-generation": "The output pairs a generated image with one spatial attribution matrix per prompt token, showing where in the image each word's influence landed.",
};

const inputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Inspect input attributions",
    body: 'For classification, the attribution lives on the input tokens themselves. "lot" carries one of the strongest signals toward the predicted class — click any token to read its exact score.',
    target: "output",
    phase: "result",
    outputInteraction: { classificationTokenIndex: 6 },
  },
  "text-generation": {
    title: "Input to output heatmap",
    body: 'Selecting an input token answers: which generated tokens did it influence most? Select "Italy" — it drove the model straight to naming Rome.',
    target: "output",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "input", selectedIndex: 29 } },
  },
  "txt2img-generation": {
    title: "Prompt token to image heatmap",
    body: 'Selecting a prompt token overlays its spatial heatmap on the image. "horses" lights up almost exactly where the horses were rendered.',
    target: "output",
    phase: "result",
    outputInteraction: { imageSelection: { selectedTokenIndices: [3] } },
  },
};

const outputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Output target context",
    body: "The label above the tokens names the predicted class. Everything shown here explains why the model landed on that class for this sentence, not some other one.",
    target: "output",
    phase: "result",
    outputInteraction: { classificationTokenIndex: 4 },
  },
  "text-generation": {
    title: "Output to input heatmap",
    body: 'Selecting a generated token flips the view: which earlier words — prompt or already-generated — contributed to it? Select "Rome" to trace it back to "Italy" in the prompt.',
    target: "output",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "output", selectedIndex: 5 } },
  },
  "txt2img-generation": {
    title: "Image region to prompt heatmap",
    body: 'Hovering a region of the image asks the reverse question: which prompt tokens were active there? This cell sits on the horses — hover it and watch "horses" light up among the token scores.',
    target: "output",
    phase: "result",
    outputInteraction: { imageSelection: { hoveredCell: { x: 26, y: 31 } } },
  },
};

export const getTutorialSteps = (tutorialKind: TutorialKind): TutorialStep[] => [
  {
    title: "Choose a source",
    body: "Every run starts here: pick where the model lives. Hugging Face Hub is the only source LumiXAI ships with today.",
    target: "configuration",
    phase: "source",
  },
  {
    title: "Choose a model",
    body: "Paste in any compatible Hugging Face model id. This field now holds the model that actually produced the result you're about to see.",
    target: "configuration",
    phase: "model",
  },
  {
    title: "Choose an attributor",
    body: "Attributors are the algorithms that compute the explanation. Integrated Gradients (Captum) covers classifiers and text generators; DAAM computes spatial maps for diffusion image models.",
    target: "configuration",
    phase: "attributor",
  },
  {
    title: "Load configuration",
    body: "In a live session this button loads the model and attributor onto the backend. Here it just confirms readiness, since this example's result already exists.",
    target: "configuration",
    phase: "configuration",
  },
  {
    title: "Add input text",
    body: "This is exactly what you'd type or paste for a real run. The panel now holds the prompt behind the result you're about to open.",
    target: "input",
    phase: "input",
  },
  {
    title: "Run attribution",
    body: "In a live session this button starts inference on the backend. Here it jumps straight to the already-computed result, so you can see the output without waiting.",
    target: "input",
    phase: "result",
  },
  {
    title: "Read the output",
    body: outputDescriptions[tutorialKind],
    target: "output",
    phase: "result",
  },
  inputHeatmapSteps[tutorialKind],
  outputHeatmapSteps[tutorialKind],
  {
    title: "Find it in history",
    body: "This run is pinned at the top of Job History so it's easy to find again — pin any of your own runs the same way to keep them close by. Every real run you explain lands in this same list.",
    target: "history",
    phase: "result",
  },
];

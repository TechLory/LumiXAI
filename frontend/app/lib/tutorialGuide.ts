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

const tutorialNames: Record<TutorialKind, string> = {
  "text-classification": "text classification",
  "text-generation": "text generation",
  "txt2img-generation": "text-to-image generation",
};

const outputDescriptions: Record<TutorialKind, string> = {
  "text-classification": "The prepared result predicts a positive class. Token color intensity is the attribution value: stronger green tokens support the selected class more.",
  "text-generation": "The prepared result shows generated tokens with probabilities. When nothing is selected, each output token displays its model probability.",
  "txt2img-generation": "The prepared result contains a generated image, prompt tokens, and one spatial attribution matrix per token.",
};

const inputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Inspect input attributions",
    body: "For classification, the input tokens themselves carry the attribution heatmap. Stronger token colors indicate larger contribution toward the predicted class.",
    target: "output",
    phase: "result",
    outputInteraction: { classificationTokenIndex: 5 },
  },
  "text-generation": {
    title: "Input to output heatmap",
    body: "Selecting an input token answers: which generated tokens did this input token influence most? The output boxes switch from probability to influence values.",
    target: "output",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "input", selectedIndex: 3 } },
  },
  "txt2img-generation": {
    title: "Prompt token to image heatmap",
    body: "Selecting a prompt token overlays its prepared spatial heatmap on the image. Here the lighthouse token lights up the generated structure.",
    target: "output",
    phase: "result",
    outputInteraction: { imageSelection: { selectedTokenIndices: [2] } },
  },
};

const outputHeatmapSteps: Record<TutorialKind, TutorialStep> = {
  "text-classification": {
    title: "Output target context",
    body: "The output target is the predicted class. The label tells you which class the displayed token attributions explain.",
    target: "output",
    phase: "result",
    outputInteraction: { classificationTokenIndex: 13 },
  },
  "text-generation": {
    title: "Output to input heatmap",
    body: "Selecting a generated token answers: which input tokens and earlier outputs contributed to this token? The input panel inside the output view becomes the heatmap.",
    target: "output",
    phase: "result",
    outputInteraction: { textGenerationSelection: { selectedType: "output", selectedIndex: 6 } },
  },
  "txt2img-generation": {
    title: "Image region to prompt heatmap",
    body: "Inspecting an image region answers: which prompt tokens were most active for this part of the image? The token scores update from the prepared pixel attribution matrix.",
    target: "output",
    phase: "result",
    outputInteraction: { imageSelection: { hoveredCell: { x: 32, y: 34 } } },
  },
};

export const getTutorialSteps = (tutorialKind: TutorialKind): TutorialStep[] => [
  {
    title: "Choose a source",
    body: `This ${tutorialNames[tutorialKind]} tutorial uses a bundled example. The source is filled first, just like a normal run, but no backend request is made.`,
    target: "configuration",
    phase: "source",
  },
  {
    title: "Choose a model",
    body: "The model field is filled with the model that produced the prepared example. This keeps the walkthrough reproducible after installation.",
    target: "configuration",
    phase: "model",
  },
  {
    title: "Choose an attributor",
    body: "The attributor is selected next. Captum powers the text examples, while DAAM powers the text-to-image example.",
    target: "configuration",
    phase: "attributor",
  },
  {
    title: "Load configuration",
    body: "The tutorial marks this configuration as ready from the bundled fixture. In a real run this button loads the model and attributor on the backend.",
    target: "configuration",
    phase: "configuration",
  },
  {
    title: "Add input text",
    body: "The input panel is filled with the example prompt. You can read it as the exact text that produced the prepared result.",
    target: "input",
    phase: "input",
  },
  {
    title: "Run attribution",
    body: "The run button is highlighted, but the tutorial loads the prepared result instead of starting inference. This makes the walkthrough instant and deterministic.",
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
    body: "This example is always available in Job History. The same history can also show results created by normal UI runs or inserted by scripts through the app data layer.",
    target: "history",
    phase: "result",
  },
];

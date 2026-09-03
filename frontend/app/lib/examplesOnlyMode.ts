const truthyValues = new Set(["1", "true", "yes", "on"]);

export const isExamplesOnlyMode = truthyValues.has(
  (process.env.NEXT_PUBLIC_LUMIXAI_EXAMPLES_ONLY ?? "").trim().toLowerCase()
);


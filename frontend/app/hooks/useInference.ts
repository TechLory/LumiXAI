import { useState, useRef } from "react";
import { apiFetch, buildApiUrl } from "../lib/api";
import { AsyncState, ResultMetadata } from "../types";

export function useInference() {
  const [inputText, setInputText] = useState("Astronauts riding horses on Mars.");
  // Uploaded image, base64-encoded (no data-URI prefix), used instead of `inputText`
  // for image classification models. Cleared whenever a new image is picked or the
  // input is reset.
  const [inputImageBase64, setInputImageBase64] = useState<string | null>(null);
  // Original file name of the uploaded image, sent alongside `inputImageBase64` so the
  // backend can label the job with it instead of a generic placeholder.
  const [inputImageFileName, setInputImageFileName] = useState<string | null>(null);
  // Optional seed for reproducible image generation. Empty string => random
  // (the backend picks fresh noise, preserving the previous default behavior).
  const [seed, setSeed] = useState<string>("");
  const [maxNewTokens, setMaxNewTokens] = useState<string>("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [resultMetadata, setResultMetadata] = useState<ResultMetadata | null>(null);
  const [inferenceState, setInferenceState] = useState<AsyncState>({
    status: 'idle', data: null, error: null
  });

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pendingResultMetadataRef = useRef<ResultMetadata | null>(null);

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const pollJobStatus = async (jobId: string) => {
    try {
      const res = await fetch(buildApiUrl(`/api/jobs/${jobId}`));
      if (res.status === 404) {
        stopPolling();
        setActiveJobId(null);
        setResultMetadata(null);
        pendingResultMetadataRef.current = null;
        setInferenceState({ status: 'idle', data: null, error: null });
        return;
      }
      if (!res.ok) throw new Error("Failed to fetch job status");

      const job = await res.json();

      if (job.status === "completed") {
        stopPolling();
        setActiveJobId(null);
        setResultMetadata(pendingResultMetadataRef.current);
        pendingResultMetadataRef.current = null;
        setInferenceState({ status: 'success', data: job.payload, error: null });
      } else if (job.status === "failed") {
        stopPolling();
        setActiveJobId(null);
        setResultMetadata(null);
        pendingResultMetadataRef.current = null;
        setInferenceState({ status: 'error', data: null, error: job.error_message || "Job failed during execution." });
      }
      // If still running, do nothing and wait for the next poll
    } catch (e: any) {
      stopPolling();
      setActiveJobId(null);
      setResultMetadata(null);
      pendingResultMetadataRef.current = null;
      setInferenceState({ status: 'error', data: null, error: e.message });
    }
  };

  const handleExplain = async (
    ignoreSpecialTokens: boolean = true,
    disableThinking: boolean = false,
    nextResultMetadata: ResultMetadata | null = null,
    // Names the configuration these results must be about. The backend refuses the job if
    // another session has since loaded something else, instead of explaining that instead.
    configId: string | null = null,
    onConfigLost: ((message: string) => void) | null = null
  ) => {
    if (!inputImageBase64 && !inputText.trim()) return;

    // Spinner
    setResultMetadata(null);
    setInferenceState({ status: 'running', data: null, error: null });
    stopPolling();
    setActiveJobId(null);
    pendingResultMetadataRef.current = nextResultMetadata;

    try {
      // Parse the optional seed: empty/invalid => null (random generation).
      const trimmedSeed = seed.trim();
      const parsedSeed = trimmedSeed === "" ? null : Number.parseInt(trimmedSeed, 10);
      const seedValue = parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null;
      const trimmedMaxNewTokens = maxNewTokens.trim();
      const parsedMaxNewTokens = trimmedMaxNewTokens === "" ? null : Number.parseInt(trimmedMaxNewTokens, 10);
      const maxNewTokensValue = parsedMaxNewTokens !== null && Number.isFinite(parsedMaxNewTokens) && parsedMaxNewTokens > 0
        ? parsedMaxNewTokens
        : null;

      // 1. Creiamo il Job
      const res = await apiFetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputImageBase64 ? undefined : inputText,
          image_base64: inputImageBase64 ?? undefined,
          image_filename: inputImageBase64 ? inputImageFileName ?? undefined : undefined,
          ignore_special_tokens: ignoreSpecialTokens,
          seed: seedValue,
          max_new_tokens: maxNewTokensValue,
          disable_thinking: disableThinking,
          config_id: configId ?? undefined
        })
      });

      const data = await res.json();

      // 409: this tab's configuration is gone (another session loaded its own, or the
      // model was reaped). Hand it to the caller so the Configuration panel can recover.
      if (res.status === 409) {
        onConfigLost?.(data.detail || "Your configuration is no longer loaded on the backend.");
        throw new Error(data.detail || "Your configuration is no longer loaded on the backend.");
      }

      if (!res.ok) throw new Error(data.detail || "Failed to start inference job");

      // 2. Got job_id -> Start polling
      setActiveJobId(data.job_id);
      pollingIntervalRef.current = setInterval(() => pollJobStatus(data.job_id), 5000);

    } catch (e: any) {
      setActiveJobId(null);
      setResultMetadata(null);
      pendingResultMetadataRef.current = null;
      setInferenceState({ status: 'error', data: null, error: e.message });
    }
  };

  const resetInferenceState = (nextInputText: string = "") => {
    stopPolling();
    setActiveJobId(null);
    setResultMetadata(null);
    pendingResultMetadataRef.current = null;
    setInputText(nextInputText);
    setInputImageBase64(null);
    setInputImageFileName(null);
    setInferenceState({ status: 'idle', data: null, error: null });
  };

  // Reload past job
  const loadPastJob = (payload: any, prompt: string, nextResultMetadata: ResultMetadata | null = null) => {
    stopPolling();
    setActiveJobId(null);
    setResultMetadata(nextResultMetadata);
    pendingResultMetadataRef.current = null;
    setInputText(prompt);
    setInputImageBase64(null);
    setInputImageFileName(null);
    setInferenceState({ status: 'success', data: payload, error: null });
  };

  const handleDeletedJob = (jobId: string) => {
    if (jobId !== activeJobId) {
      return;
    }

    stopPolling();
    setActiveJobId(null);
    setResultMetadata(null);
    pendingResultMetadataRef.current = null;
    setInferenceState({ status: 'idle', data: null, error: null });
  };

  return { inputText, setInputText, inputImageBase64, setInputImageBase64, inputImageFileName, setInputImageFileName, seed, setSeed, maxNewTokens, setMaxNewTokens, inferenceState, resultMetadata, handleExplain, loadPastJob, resetInferenceState, handleDeletedJob };
}

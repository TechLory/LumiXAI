import { useState, useRef } from "react";
import { AsyncState } from "../types";

export function useInference() {
  const isAppLocal = true;
  const ipAddress = isAppLocal ? "localhost" : "192.168.1.23";

  const [inputText, setInputText] = useState("Astronauts riding horses on Mars.");
  const [inferenceState, setInferenceState] = useState<AsyncState>({
    status: 'idle', data: null, error: null
  });

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const pollJobStatus = async (jobId: string) => {
    try {
      const res = await fetch(`http://${ipAddress}:8000/api/jobs/${jobId}`);
      if (!res.ok) throw new Error("Failed to fetch job status");

      const job = await res.json();

      if (job.status === "completed") {
        stopPolling();
        setInferenceState({ status: 'success', data: job.payload, error: null });
      } else if (job.status === "failed") {
        stopPolling();
        setInferenceState({ status: 'error', data: null, error: job.error_message || "Job failed during execution." });
      }
      // If still running, do nothing and wait for the next poll
    } catch (e: any) {
      stopPolling();
      setInferenceState({ status: 'error', data: null, error: e.message });
    }
  };

  const handleExplain = async (ignoreSpecialTokens: boolean = true) => {
    if (!inputText.trim()) return;

    // Spinner
    setInferenceState({ status: 'running', data: null, error: null });
    stopPolling();

    try {
      // 1. Creiamo il Job
      const res = await fetch(`http://${ipAddress}:8000/api/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText,
          ignore_special_tokens: ignoreSpecialTokens
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to start inference job");

      // 2. Got job_id -> Start polling
      pollingIntervalRef.current = setInterval(() => pollJobStatus(data.job_id), 5000);

    } catch (e: any) {
      setInferenceState({ status: 'error', data: null, error: e.message });
    }
  };

  // Reload past job
  const loadPastJob = (payload: any, prompt: string) => {
    stopPolling();
    setInputText(prompt);
    setInferenceState({ status: 'success', data: payload, error: null });
  };

  return { inputText, setInputText, inferenceState, handleExplain, loadPastJob };
}
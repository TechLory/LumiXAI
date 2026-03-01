import { useState } from "react";
import { AsyncState } from "../types";

export function useInference() {
  const [inputText, setInputText] = useState("Astronauts riding horses on Mars.");
  const [inferenceState, setInferenceState] = useState<AsyncState>({
    status: 'idle', data: null, error: null
  });

  const handleExplain = async () => {
    if (!inputText.trim()) return;
    
    setInferenceState({ status: 'running', data: null, error: null });

    try {
      const res = await fetch("http://localhost:8000/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Inference failed");

      setInferenceState({ status: 'success', data: data, error: null });
    } catch (e: any) {
      setInferenceState({ status: 'error', data: null, error: e.message });
    }
  };

  return { inputText, setInputText, inferenceState, handleExplain };
}
import { useState } from "react";
import { buildApiUrl } from "../lib/api";

export type ConfigStep = 'idle' | 'checking_inputs' | 'loading_model' | 'setting_attributor' | 'ready';

export interface ConfigurationState {
  status: 'idle' | 'running' | 'success' | 'error';
  step: ConfigStep;
  errorField: 'source' | 'model' | 'attributor' | 'general' | null;
  errorMessage: string | null;
  logs: string[];
}

export function useModelManager() {
  const [selectedSource, setSelectedSource] = useState("");
  const [modelName, setModelName] = useState("");
  const [selectedAttributor, setSelectedAttributor] = useState("");

  const [configState, setConfigState] = useState<ConfigurationState>({
    status: 'idle',
    step: 'idle',
    errorField: null,
    errorMessage: null,
    logs: []
  });

  const addLog = (msg: string) => {
    setConfigState(prev => ({ ...prev, logs: [...prev.logs, msg] }));
  };

  const handleLoadConfiguration = async () => {
    // Reset
    setConfigState({
      status: 'running',
      step: 'checking_inputs',
      errorField: null,
      errorMessage: null,
      logs: ["Validating selections..."]
    });

    // 1. Input Validation
    if (!selectedSource) {
      setConfigState(prev => ({ ...prev, status: 'error', errorField: 'source', errorMessage: "Please select a model source." }));
      return;
    }
    if (!modelName) {
      setConfigState(prev => ({ ...prev, status: 'error', errorField: 'model', errorMessage: "Please select or type a model name." }));
      return;
    }
    if (!selectedAttributor) {
      setConfigState(prev => ({ ...prev, status: 'error', errorField: 'attributor', errorMessage: "Please select an attributor." }));
      return;
    }

    // 2. Load Model
    setConfigState(prev => ({ ...prev, step: 'loading_model' }));
    addLog(`Loading model '${modelName}' from '${selectedSource}'...`);
    
    try {
      const modelRes = await fetch(buildApiUrl("/api/load"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: selectedSource, model_name: modelName, device: "auto" })
      });

      if (!modelRes.ok) {
        const errorText = await modelRes.text();
        let errorDetail = errorText;
        try { errorDetail = JSON.parse(errorText).detail; } catch {}
        
        setConfigState(prev => ({ ...prev, status: 'error', errorField: 'model', errorMessage: errorDetail }));
        return; 
      }
      
      const modelData = await modelRes.json();
      addLog(`Model loaded successfully. Task: ${modelData.detected_task}`);

      // 3. Set Attributor
      setConfigState(prev => ({ ...prev, step: 'setting_attributor' }));
      addLog(`Setting attributor to '${selectedAttributor}'...`);

      const attrRes = await fetch(buildApiUrl("/api/set_attributor"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attributor_id: selectedAttributor })
      });

      if (!attrRes.ok) {
        setConfigState(prev => ({ ...prev, status: 'error', errorField: 'attributor', errorMessage: "Failed to set attributor on the backend." }));
        return;
      }

      // 4. Success
      setConfigState(prev => ({ 
        ...prev, 
        status: 'success', 
        step: 'ready',
        logs: [...prev.logs, "Configuration fully loaded and ready."]
      }));

    } catch (e: any) {
       setConfigState(prev => ({ ...prev, status: 'error', errorField: 'general', errorMessage: e.message || "Network error." }));
    }
  };

  return {
    selectedSource, setSelectedSource,
    modelName, setModelName,
    selectedAttributor, setSelectedAttributor,
    configState, handleLoadConfiguration
  };
}

import { useState } from "react";
import { buildApiUrl } from "../lib/api";

export type ConfigStep =
  | 'idle'
  | 'checking_inputs'
  | 'loading_model'
  | 'setting_attributor'
  | 'unloading_model'
  | 'ready'
  | 'unloaded';

export interface ConfigurationState {
  status: 'idle' | 'running' | 'success' | 'error';
  step: ConfigStep;
  errorField: 'source' | 'model' | 'attributor' | 'general' | null;
  errorMessage: string | null;
  logs: string[];
}

export interface LoadedConfiguration {
  source: string;
  modelName: string;
  attributor: string;
}

export function useModelManager() {
  const [selectedSource, setSelectedSource] = useState("");
  const [modelName, setModelName] = useState("");
  const [selectedAttributor, setSelectedAttributor] = useState("");
  const [lastLoadedConfiguration, setLastLoadedConfiguration] = useState<LoadedConfiguration | null>(null);
  const [hasActiveConfiguration, setHasActiveConfiguration] = useState(false);
  const [activeAttributorId, setActiveAttributorId] = useState<string | null>(null);

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

  const clearErrors = () => {
    setConfigState(prev => {
      if (!prev.errorField && !prev.errorMessage) {
        return prev;
      }

      return {
        ...prev,
        status: hasActiveConfiguration ? 'success' : 'idle',
        step: hasActiveConfiguration ? 'ready' : 'idle',
        errorField: null,
        errorMessage: null
      };
    });
  };

  const onSourceChange = (newValue: string) => {
    setSelectedSource(newValue);
    clearErrors();
  };

  const onModelNameChange = (newValue: string) => {
    setModelName(newValue);
    clearErrors();
  };

  const onAttributorChange = (newValue: string) => {
    setSelectedAttributor(newValue);
    clearErrors();
  };

  const invalidateActiveConfiguration = () => {
    setHasActiveConfiguration(false);
    setActiveAttributorId(null);
  };

  const restoreLoadedConfiguration = () => {
    if (!lastLoadedConfiguration) {
      return;
    }

    setSelectedSource(lastLoadedConfiguration.source);
    setModelName(lastLoadedConfiguration.modelName);
    setSelectedAttributor(lastLoadedConfiguration.attributor);
    setConfigState(prev => ({
      ...prev,
      status: hasActiveConfiguration ? 'success' : 'idle',
      step: hasActiveConfiguration ? 'ready' : 'idle',
      errorField: null,
      errorMessage: null
    }));
  };

  const handleLoadConfiguration = async () => {
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

    setConfigState(prev => ({ ...prev, step: 'loading_model' }));
    addLog(`Loading model '${modelName}' from '${selectedSource}'...`);

    let hasStartedBackendLoad = false;

    try {
      hasStartedBackendLoad = true;
      const modelRes = await fetch(buildApiUrl("/api/load"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: selectedSource, model_name: modelName, device: "auto" })
      });

      if (!modelRes.ok) {
        const errorText = await modelRes.text();
        let errorDetail = errorText;
        try { errorDetail = JSON.parse(errorText).detail; } catch {}

        invalidateActiveConfiguration();
        setConfigState(prev => ({
          ...prev,
          status: 'error',
          errorField: 'model',
          errorMessage: errorDetail,
          logs: [...prev.logs, `Load failed: ${errorDetail}`]
        }));
        return;
      }

      const modelData = await modelRes.json();
      addLog(`Model loaded successfully. Task: ${modelData.detected_task}`);

      setConfigState(prev => ({ ...prev, step: 'setting_attributor' }));
      addLog(`Setting attributor to '${selectedAttributor}'...`);

      const attrRes = await fetch(buildApiUrl("/api/set_attributor"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attributor_id: selectedAttributor })
      });

      if (!attrRes.ok) {
        let errorDetail = "Failed to set attributor on the backend.";
        try {
          const errorPayload = await attrRes.json();
          errorDetail = errorPayload.detail || errorDetail;
        } catch {}

        invalidateActiveConfiguration();
        setConfigState(prev => ({
          ...prev,
          status: 'error',
          errorField: 'attributor',
          errorMessage: errorDetail,
          logs: [...prev.logs, `Attributor setup failed: ${errorDetail}`]
        }));
        return;
      }

      const loadedConfiguration = {
        source: selectedSource,
        modelName,
        attributor: selectedAttributor
      };

      setLastLoadedConfiguration(loadedConfiguration);
      setHasActiveConfiguration(true);
      setActiveAttributorId(selectedAttributor);
      setConfigState(prev => ({ 
        ...prev, 
        status: 'success', 
        step: 'ready',
        logs: [...prev.logs, "Configuration fully loaded and ready."]
      }));

    } catch (e: any) {
      if (hasStartedBackendLoad) {
        invalidateActiveConfiguration();
      }

      const errorMessage = e.message || "Network error.";
      setConfigState(prev => ({
        ...prev,
        status: 'error',
        errorField: 'general',
        errorMessage,
        logs: [...prev.logs, `Configuration request failed: ${errorMessage}`]
      }));
    }
  };

  const handleResetConfiguration = () => {
    restoreLoadedConfiguration();
  };

  const handleUnloadConfiguration = async () => {
    setConfigState({
      status: 'running',
      step: 'unloading_model',
      errorField: null,
      errorMessage: null,
      logs: ["Unloading configuration and freeing memory..."]
    });

    try {
      const res = await fetch(buildApiUrl("/api/unload"), {
        method: "POST"
      });

      if (!res.ok) {
        let errorDetail = "Failed to unload the current configuration.";
        try {
          const errorPayload = await res.json();
          errorDetail = errorPayload.detail || errorDetail;
        } catch {}

        setConfigState(prev => ({
          ...prev,
          status: 'error',
          step: 'ready',
          errorField: 'general',
          errorMessage: errorDetail,
          logs: [...prev.logs, `Unload failed: ${errorDetail}`]
        }));
        return;
      }

      const data = await res.json();
      setHasActiveConfiguration(false);
      setActiveAttributorId(null);
      setConfigState({
        status: 'idle',
        step: 'unloaded',
        errorField: null,
        errorMessage: null,
        logs: [data.message || "Configuration unloaded and memory freed."]
      });
    } catch (e: any) {
      const errorMessage = e.message || "Network error.";
      setConfigState(prev => ({
        ...prev,
        status: 'error',
        step: 'ready',
        errorField: 'general',
        errorMessage,
        logs: [...prev.logs, `Unload failed: ${errorMessage}`]
      }));
    }
  };

  const isDirty = !!lastLoadedConfiguration && (
    selectedSource !== lastLoadedConfiguration.source ||
    modelName !== lastLoadedConfiguration.modelName ||
    selectedAttributor !== lastLoadedConfiguration.attributor
  );

  return {
    selectedSource, onSourceChange,
    modelName, onModelNameChange,
    selectedAttributor, onAttributorChange,
    configState,
    lastLoadedConfiguration,
    hasActiveConfiguration,
    activeAttributorId,
    isDirty,
    handleLoadConfiguration,
    handleResetConfiguration,
    handleUnloadConfiguration
  };
}

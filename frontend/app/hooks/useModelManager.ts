import { useCallback, useState } from "react";
import { apiFetch } from "../lib/api";
import { guessWrapperFromTask } from "../lib/taskToWrapper";

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
  // Set when the backend refused the load because another session still holds the model
  // (423). Offers the user the deliberate takeover rather than retrying behind their back.
  canTakeOver?: boolean;
}

export interface LoadedConfiguration {
  source: string;
  modelName: string;
  attributor: string;
  detectedTask?: string;
}

export interface HydratedConfiguration {
  source?: string;
  sourceName?: string;
  modelName?: string;
  attributor?: string;
  attributorName?: string;
  detectedTask?: string;
}

export interface AttributorCompatibility {
  id: string;
  compatible_wrappers: string[];
}

export function useModelManager(availableAttributors: AttributorCompatibility[] = []) {
  const [selectedSource, setSelectedSource] = useState("");
  const [modelName, setModelName] = useState("");
  const [selectedAttributor, setSelectedAttributor] = useState("");
  const [lastLoadedConfiguration, setLastLoadedConfiguration] = useState<LoadedConfiguration | null>(null);
  const [hasActiveConfiguration, setHasActiveConfiguration] = useState(false);
  const [activeAttributorId, setActiveAttributorId] = useState<string | null>(null);
  // The backend's token for the configuration this tab loaded. Sent with every explain so
  // the backend can refuse rather than attribute against a model someone else swapped in.
  const [activeConfigId, setActiveConfigId] = useState<string | null>(null);
  const [detectedWrapperName, setDetectedWrapperName] = useState<string | null>(null);
  const [detectedTask, setDetectedTask] = useState<string | null>(null);

  const isAttributorCompatible = (attributorId: string, wrapperName: string | null): boolean => {
    if (!wrapperName) return true;
    const attributor = availableAttributors.find(a => a.id === attributorId);
    if (!attributor || attributor.compatible_wrappers.length === 0) return true;
    return attributor.compatible_wrappers.includes(wrapperName);
  };

  // Updates the detected model type and, if the currently-selected draft attributor is no
  // longer valid for it, clears that draft selection (never touches an already-active
  // backend configuration).
  const applyDetectedWrapper = (wrapperName: string | null, task: string | null) => {
    setDetectedWrapperName(wrapperName);
    setDetectedTask(task);
    setSelectedAttributor(prevAttributor => {
      if (prevAttributor && !isAttributorCompatible(prevAttributor, wrapperName)) {
        return "";
      }
      return prevAttributor;
    });
  };

  const applyDetectedTask = (task?: string | null) => {
    applyDetectedWrapper(guessWrapperFromTask(task), task ?? null);
  };

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

  const onModelNameChange = (newValue: string, task?: string) => {
    setModelName(newValue);
    applyDetectedTask(task);
    clearErrors();
  };

  const onAttributorChange = (newValue: string) => {
    setSelectedAttributor(newValue);
    clearErrors();
  };

  const invalidateActiveConfiguration = () => {
    setHasActiveConfiguration(false);
    setActiveAttributorId(null);
    setActiveConfigId(null);
  };

  const restoreLoadedConfiguration = () => {
    if (!lastLoadedConfiguration) {
      return;
    }

    setSelectedSource(lastLoadedConfiguration.source);
    setModelName(lastLoadedConfiguration.modelName);
    setSelectedAttributor(lastLoadedConfiguration.attributor);
    applyDetectedTask(lastLoadedConfiguration.detectedTask);
    setConfigState(prev => ({
      ...prev,
      status: hasActiveConfiguration ? 'success' : 'idle',
      step: hasActiveConfiguration ? 'ready' : 'idle',
      errorField: null,
      errorMessage: null
    }));
  };

  const hydrateConfiguration = (configuration: HydratedConfiguration, isLoaded: boolean) => {
    const nextSource = configuration.source ?? "";
    const nextModelName = configuration.modelName ?? "";
    const nextAttributor = configuration.attributor ?? "";

    setSelectedSource(nextSource);
    setModelName(nextModelName);
    setSelectedAttributor(nextAttributor);
    applyDetectedTask(configuration.detectedTask);

    if (isLoaded && nextSource && nextModelName && nextAttributor) {
      const loadedConfiguration = {
        source: nextSource,
        modelName: nextModelName,
        attributor: nextAttributor,
        detectedTask: configuration.detectedTask
      };

      setLastLoadedConfiguration(loadedConfiguration);
      setHasActiveConfiguration(true);
      setActiveAttributorId(nextAttributor);
      // The tutorial replays bundled data and never loads a model, so there is no backend
      // configuration to hold a token for.
      setActiveConfigId(null);
      setConfigState({
        status: 'success',
        step: 'ready',
        errorField: null,
        errorMessage: null,
        logs: [
          "Tutorial configuration loaded from bundled example data.",
          "Model '" + nextModelName + "' selected from '" + (configuration.sourceName ?? nextSource) + "'.",
          "Attributor '" + (configuration.attributorName ?? nextAttributor) + "' is ready.",
          configuration.detectedTask ? "Detected task: " + configuration.detectedTask + "." : "Configuration fully loaded and ready."
        ].filter(Boolean)
      });
      return;
    }

    setLastLoadedConfiguration(null);
    setHasActiveConfiguration(false);
    setActiveAttributorId(null);
    setActiveConfigId(null);
    setConfigState({
      status: 'idle',
      step: 'idle',
      errorField: null,
      errorMessage: null,
      logs: []
    });
  };

  const handleLoadConfiguration = async (takeOver: boolean = false) => {
    setConfigState({
      status: 'running',
      step: 'checking_inputs',
      errorField: null,
      errorMessage: null,
      logs: takeOver ? ["Taking over the loaded configuration..."] : ["Validating selections..."]
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
      const modelRes = await apiFetch("/api/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: selectedSource, model_name: modelName, device: "auto", force: takeOver })
      });

      if (!modelRes.ok) {
        const errorText = await modelRes.text();
        let errorDetail = errorText;
        try { errorDetail = JSON.parse(errorText).detail; } catch {}

        // 423: someone else's model is still live. Nothing was touched on the backend, so
        // this tab keeps whatever it had and the user decides whether to take over.
        const isLeaseConflict = modelRes.status === 423;
        if (!isLeaseConflict) invalidateActiveConfiguration();

        setConfigState(prev => ({
          ...prev,
          status: 'error',
          errorField: 'model',
          errorMessage: errorDetail,
          canTakeOver: isLeaseConflict,
          logs: [...prev.logs, `Load failed: ${errorDetail}`]
        }));
        return;
      }

      const modelData = await modelRes.json();
      addLog(`Model loaded successfully. Task: ${modelData.detected_task}`);

      // Authoritative update from the backend (covers manually-typed model ids that never
      // went through ModelSelector's search results). Strips any " (fallback)" suffix so
      // it matches the plain wrapper-name keys used in compatible_wrappers.
      const rawWrapperName = typeof modelData.wrapper === "string" ? modelData.wrapper.split(" ")[0] : null;
      applyDetectedWrapper(rawWrapperName, modelData.detected_task ?? null);

      setConfigState(prev => ({ ...prev, step: 'setting_attributor' }));
      addLog(`Setting attributor to '${selectedAttributor}'...`);

      const attrRes = await apiFetch("/api/set_attributor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ attributor_id: selectedAttributor, force: takeOver })
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

      const attrData = await attrRes.json();
      const loadedConfiguration = {
        source: selectedSource,
        modelName,
        attributor: selectedAttributor,
        detectedTask: modelData.detected_task ?? undefined
      };

      setLastLoadedConfiguration(loadedConfiguration);
      setHasActiveConfiguration(true);
      setActiveAttributorId(selectedAttributor);
      // Attaching the attributor re-mints the token, so this is the one to keep.
      setActiveConfigId(attrData.config_id ?? modelData.config_id ?? null);
      setConfigState(prev => ({
        ...prev,
        status: 'success',
        step: 'ready',
        canTakeOver: false,
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
      const res = await apiFetch("/api/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false })
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
      invalidateActiveConfiguration();
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

  // The backend can drop the model without us asking (the idle reaper), so the active
  // configuration has to be surrendered on its say-so, not only through the Unload button.
  // Keeps the draft selections intact so re-loading is a single click.
  const notifyBackendUnload = useCallback((message: string) => {
    setHasActiveConfiguration(false);
    setActiveAttributorId(null);
    setActiveConfigId(null);
    setConfigState({
      status: 'idle',
      step: 'unloaded',
      errorField: null,
      errorMessage: null,
      logs: [message]
    });
  }, []);

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
    activeConfigId,
    detectedWrapperName,
    detectedTask,
    isDirty,
    handleLoadConfiguration,
    handleResetConfiguration,
    handleUnloadConfiguration,
    hydrateConfiguration,
    notifyBackendUnload
  };
}

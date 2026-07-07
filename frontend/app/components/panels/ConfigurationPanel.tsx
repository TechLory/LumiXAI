import ModelSelector from "../layout/ModelSelector";
import { ConfigurationState } from "../../hooks/useModelManager";
import type { TutorialFocusTarget } from "../../lib/tutorialGuide";

interface ConfigurationPanelProps {
  manifest: {
    sources: { id: string; name: string; type: string }[];
    attributors: { id: string; name: string; compatible_wrappers: string[] }[]
  } | null;
  selectedSource: string;
  modelName: string;
  selectedAttributor: string;
  detectedTask: string | null;
  detectedWrapperName: string | null;

  configState: ConfigurationState;
  isDirty: boolean;
  hasActiveConfiguration: boolean;
  hasResetTarget: boolean;
  isInferenceRunning: boolean;

  onSourceChange: (newValue: string) => void;
  onModelNameChange: (newValue: string, task?: string) => void;
  onAttributorChange: (newValue: string) => void;
  onLoadConfiguration: () => void;
  onResetConfiguration: () => void;
  onUnloadConfiguration: () => void;
  tutorialFocusTarget?: TutorialFocusTarget;
}

export default function ConfigurationPanel(props: ConfigurationPanelProps) {
  const { status, step, errorField, errorMessage, logs } = props.configState;
  const isRunning = status === 'running';
  const isInteractionDisabled = isRunning || props.isInferenceRunning;
  const isUnloadRunning = isRunning && step === 'unloading_model';
  const isLoadRunning = isRunning && step !== 'unloading_model';

  const getTutorialFocusClass = (target: TutorialFocusTarget) => (
    props.tutorialFocusTarget === target ? " tutorial-inner-highlight" : ""
  );

  const getRowClasses = (fieldId: 'source' | 'model' | 'attributor') => {
    const focusTargetByField: Record<'source' | 'model' | 'attributor', TutorialFocusTarget> = {
      source: "configuration-source",
      model: "configuration-model",
      attributor: "configuration-attributor",
    };
    const focusClass = getTutorialFocusClass(focusTargetByField[fieldId]);

    if (errorField === fieldId) {
      return `bg-danger-soft flex items-center${focusClass}`;
    }
    return `bg-fill flex items-center${focusClass}`;
  };

  const getLabelText = (fieldId: 'source' | 'model' | 'attributor', defaultText: string) => {
    if (errorField === fieldId && errorMessage) {
      return `// ERROR: ${errorMessage}`;
    }
    return `// ${defaultText}`;
  };

  return (
    <div className="mt-5">
      <div className="flex flex-col gap-1.5">

        {/* 1: MODEL SOURCE */}
        <div className={getRowClasses('source')}>
          {/* SX Label */}
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'source' ? 'text-danger' : 'text-fg-subtle uppercase'}`}>
            {getLabelText('source', 'select a source')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <select
              className="w-full text-sm font-mono font-medium bg-transparent text-fg outline-none p-2 disabled:opacity-50"
              value={props.selectedSource}
              onChange={e => props.onSourceChange(e.target.value)}
              disabled={isInteractionDisabled}
            >
              <option value="" disabled className="bg-surface text-fg">Select a source...</option>
              {props.manifest?.sources.map(w => (
                <option key={w.id} value={w.id} className="bg-surface text-fg">{w.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 2: MODEL NAME */}
        <div className={getRowClasses('model')}>
          {/* SX Label */}
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'model' ? 'text-danger' : 'text-fg-subtle uppercase'}`}>
            {getLabelText('model', 'select a model')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <ModelSelector
              currentSource={props.selectedSource}
              currentModel={props.modelName}
              onModelSelect={props.onModelNameChange}
              disabled={isInteractionDisabled}
            />
          </div>
        </div>

        {/* DETECTED MODEL TYPE */}
        <div className="px-2 -mt-1 mb-0.5 text-xs font-mono text-fg-faint">
          {props.detectedTask
            ? <>Detected type: <span className="text-fg-subtle">{props.detectedTask}</span></>
            : "Detected type: unknown until a model is selected"}
        </div>

        {/* 3: ATTRIBUTOR */}
        <div className={getRowClasses('attributor')}>
          {/* SX Label */}
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'attributor' ? 'text-danger' : 'text-fg-subtle uppercase'}`}>
            {getLabelText('attributor', 'select an attributor')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <select
              className="w-full text-sm font-mono font-medium bg-transparent text-fg outline-none p-2 disabled:opacity-50"
              value={props.selectedAttributor}
              onChange={e => props.onAttributorChange(e.target.value)}
              disabled={isInteractionDisabled}
            >
              <option value="" disabled className="bg-surface text-fg">Select an attributor...</option>
              {props.manifest?.attributors.map(w => {
                const isCompatible = !props.detectedWrapperName
                  || w.compatible_wrappers.length === 0
                  || w.compatible_wrappers.includes(props.detectedWrapperName);
                return (
                  <option
                    key={w.id}
                    value={w.id}
                    disabled={!isCompatible}
                    title={!isCompatible ? `Requires: ${w.compatible_wrappers.join(", ")}` : undefined}
                    className="bg-surface text-fg"
                  >
                    {w.name}{!isCompatible ? " (incompatible)" : ""}
                  </option>
                );
              })}
            </select>
          </div>
        </div>

        {/* LOAD CONFIGURATION BUTTON */}
        <div className={`mt-5${getTutorialFocusClass("configuration-action")}`}>
          {isLoadRunning ? (
            <button
              className="bg-ok-soft border border-ok-line text-ok w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              disabled
            >
              <><i className='bx bx-loader animate-spin text-lg'></i> Loading Configuration...</>
            </button>
          ) : isUnloadRunning ? (
            <button
              className="bg-danger-soft border border-danger-line text-danger w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              disabled
            >
              <><i className='bx bx-loader animate-spin text-lg'></i> Unloading Configuration...</>
            </button>
          ) : props.hasActiveConfiguration && !props.isDirty ? (
            <button
              className="bg-danger-soft hover:bg-danger-hover border border-danger-line text-danger w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              onClick={props.onUnloadConfiguration}
              disabled={isInteractionDisabled}
            >
              <><i className='bx bx-power-off text-lg'></i> Unload Configuration</>
            </button>
          ) : props.isDirty && props.hasResetTarget ? (
            <div className="flex gap-2">
              <button
                className="bg-ok-soft hover:bg-ok-hover border border-ok-line text-ok flex-1 p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                onClick={props.onLoadConfiguration}
                disabled={isInteractionDisabled}
              >
                Load Configuration
              </button>
              <button
                className="bg-fg hover:opacity-80 border border-fg text-surface flex-1 p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                onClick={props.onResetConfiguration}
                disabled={isInteractionDisabled}
              >
                Reset Configuration
              </button>
            </div>
          ) : (
            <button
              className="bg-ok-soft hover:bg-ok-hover border border-ok-line text-ok w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              onClick={props.onLoadConfiguration}
              disabled={isInteractionDisabled}
            >
              Load Configuration
            </button>
          )}

          {/* LOGS */}
          <div className="mt-4 font-mono text-fg-faint text-xs min-h-15 flex flex-col gap-1">
            {props.hasActiveConfiguration && props.isDirty && (
              <div className="text-warn">
                Draft changes are not active yet. The backend is still using the previously loaded configuration.
              </div>
            )}
            {logs.length === 0 && status !== 'error' && !props.hasResetTarget && (
              <div>No configuration loaded. Please select a model source, a model name, and an attributor, then click "Load Configuration".</div>
            )}
            {logs.map((log, idx) => (
              <div key={idx} className={`${idx === logs.length - 1 && status === 'success' ? 'text-ok' : ''}`}>
                {log}
              </div>
            ))}
            {errorField === 'general' && (
              <div className="text-danger mt-2">Critical Error: {errorMessage}</div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

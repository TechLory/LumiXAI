import { useState } from "react";
import ModelSelector from "../layout/ModelSelector";
import { ConfigurationState } from "../../hooks/useModelManager";

interface ConfigurationPanelProps {
  manifest: {
    sources: { id: string; name: string; type: string }[];
    attributors: { id: string; name: string }[]
  } | null;
  selectedSource: string;
  modelName: string;
  selectedAttributor: string;

  configState: ConfigurationState;

  onSourceChange: (newValue: string) => void;
  onModelNameChange: (newValue: string) => void;
  onAttributorChange: (newValue: string) => void;
  onLoadConfiguration: () => void;
}

export default function ConfigurationPanel(props: ConfigurationPanelProps) {
  const [isSearchBarLoading, setIsSearchBarLoading] = useState(false);

  const { status, errorField, errorMessage, logs } = props.configState;
  const isRunning = status === 'running';

  // Helper (dynamic style for inputs in case of error)
  const getRowClasses = (fieldId: 'source' | 'model' | 'attributor') => {
    if (errorField === fieldId) {
      return "bg-red-900/30 flex items-center";
    }
    return "bg-neutral-600/30 flex items-center";
  };

  // Helper (inputs inline error message)
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
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'source' ? 'text-red-400' : 'text-neutral-400 uppercase'}`}>
            {getLabelText('source', 'select a source')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <select
              className="w-full text-sm font-mono font-medium bg-transparent text-white outline-none p-2 disabled:opacity-50"
              value={props.selectedSource}
              onChange={e => props.onSourceChange(e.target.value)}
              disabled={isRunning}
            >
              <option value="" disabled className="bg-neutral-800">Select a source...</option>
              {props.manifest?.sources.map(w => (
                <option key={w.id} value={w.id} className="bg-neutral-800">{w.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 2: MODEL NAME */}
        <div className={getRowClasses('model')}>
          {/* SX Label */}
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'model' ? 'text-red-400' : 'text-neutral-400 uppercase'}`}>
            {getLabelText('model', 'select a model')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <ModelSelector
              currentSource={props.selectedSource}
              currentModel={props.modelName}
              onModelSelect={props.onModelNameChange}
            />
          </div>
        </div>

        {/* 3: ATTRIBUTOR */}
        <div className={getRowClasses('attributor')}>
          {/* SX Label */}
          <div className={`flex-1 p-2 text-xs my-2 font-mono font-medium ${errorField === 'attributor' ? 'text-red-400' : 'text-neutral-400 uppercase'}`}>
            {getLabelText('attributor', 'select an attributor')}
          </div>
          {/* DX Input */}
          <div className="flex-1">
            <select
              className="w-full text-sm font-mono font-medium bg-transparent text-white outline-none p-2 disabled:opacity-50"
              value={props.selectedAttributor}
              onChange={e => props.onAttributorChange(e.target.value)}
              disabled={isRunning}
            >
              <option value="" disabled className="bg-neutral-800">Select an attributor...</option>
              {props.manifest?.attributors.map(w => (
                <option key={w.id} value={w.id} className="bg-neutral-800">{w.name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* LOAD CONFIGURATION BUTTON */}
        <div className="mt-5">
          <button
            className="bg-emerald-900/50 hover:bg-emerald-800/60 border border-emerald-700/50 text-emerald-400 w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            onClick={props.onLoadConfiguration}
            disabled={isRunning}
          >
            {isRunning ? (
              <><i className='bx bx-loader animate-spin text-lg'></i> Loading Configuration...</>
            ) : status === 'success' ? (
              <><i className='bx bx-check text-lg'></i> Configuration Loaded</>
            ) : (
              "Load Configuration"
            )}
          </button>

          {/* LOGS */}
          <div className="mt-4 font-mono text-neutral-500 text-xs min-h-15 flex flex-col gap-1">
            {logs.length === 0 && status !== 'error' && (
              <div>No configuration loaded. Please select a model source, a model name, and an attributor, then click "Load Configuration".</div>
            )}
            {logs.map((log, idx) => (
              <div key={idx} className={`${idx === logs.length - 1 && status === 'success' ? 'text-emerald-500' : ''}`}>
                {log}
              </div>
            ))}
            {errorField === 'general' && (
              <div className="text-red-500 mt-2">Critical Error: {errorMessage}</div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
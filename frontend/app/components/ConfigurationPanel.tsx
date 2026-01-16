import { useState } from "react";
import Tooltip from "./Tooltip";
import ModelSelector from "./ModelSelector";


interface ConfigurationPanelProps {
  manifest: {
    sources: { id: string; name: string; type: string }[];
    attributors: { id: string; name: string }[]
  } | null;
  selectedSource: string;
  modelName: string;
  isModelLoading: boolean;
  isModelLoadedSuccessfully: boolean;
  statusLoadModel: string;
  statusAttributor: string;
  selectedAttributor: string;
  setIsModelLoadedSuccessfully: (value: boolean) => void;
  onSourceChange: (newValue: string) => void;
  onModelNameChange: (newValue: string) => void;
  onLoadModelClick: () => void;
  onAttributorChange: (newValue: string) => void;
}


export default function ConfigurationPanel(props: ConfigurationPanelProps) {

  const [isSearchBarLoading, setIsSearchBarLoading] = useState(false);

  return (
    <div className={"mb-20"}>
      <div className="text-3xl font-semibold py-7 text-center">Configuration</div>
      <hr className="text-neutral-600 mx-5" />



      {/* 1: MODEL SOURCE */}
      <div className="flex items-center py-5">
        <div className="text-4xl font-semibold px-6 text-neutral-600">1.</div>
        <div className="w-1/2">
          <label className="text-xl font-semibold">Model Source</label>
          <select className="w-full p-2 border rounded mt-1 text-neutral-400"
            value={props.selectedSource}
            onChange={e => props.onSourceChange(e.target.value)}
          >
            <option value="" disabled>Select a source...</option>
            {props.manifest?.sources.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
          </select>
        </div>
        <div className="flex-1">
          {props.selectedSource === "" ? (
            <Tooltip iconName="" tooltipText="Please select a model source." />
          ) : props.manifest?.sources.some(w => w.id === props.selectedSource) ? (
            <Tooltip iconName="success" tooltipText="Model source selected successfully." />
          ) : (
            <Tooltip iconName="error" tooltipText="Invalid model source selected." />
          )}
        </div>
      </div>
      <hr className="text-neutral-600 mx-5" />



      {/* 2: MODEL NAME */}
      <div className="flex items-center py-5 w-full">
        <div className="text-4xl font-semibold px-6 text-neutral-600">2.</div>
        <div className="w-full pr-10">
          <div className="mb-5">
            <label className="text-xl font-semibold">Search model in {props.manifest?.sources.find(s => s.id === props.selectedSource)?.name || "-"}</label>
            <ModelSelector
              currentSource={props.selectedSource}
              currentModel={props.modelName}
              onModelSelect={props.onModelNameChange}
              setIsLoading={setIsSearchBarLoading}
              setIsModelLoadedSuccessfully={props.setIsModelLoadedSuccessfully}
            />
          </div>
          {/* Call /api/load */}
          <button
            className="bg-green-600 p-2 rounded text-white font-semibold cursor-pointer hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={props.modelName === "" || props.isModelLoading || props.isModelLoadedSuccessfully}
            onClick={() => { props.onLoadModelClick(); }}
          >
            <div className="line-clamp-1">{props.modelName ? `Load ${props.modelName}` : "Load model"}</div>
          </button>
          {/* Load model status */}
          <div className={"mt-5 text-neutral-500 flex"}>
            <i className={`bx ${props.isModelLoading ? "bx-loader animate-spin" : "bx-stop"} text-lg mr-1`}></i>
            <div className="font-mono text-sm whitespace-pre-line">{props.statusLoadModel}</div>
          </div>
        </div>
        <div className="flex-1">
          {props.selectedSource === "" ? (
            <Tooltip iconName="stop" tooltipText="No source selected." />
          ) : isSearchBarLoading ? (
            <Tooltip iconName="loading" tooltipText="Loading..." />
          ) : props.modelName !== "" ? (
            <div>
              {props.isModelLoadedSuccessfully ? (
                <Tooltip iconName="success" tooltipText="Model loaded successfully." />
              ) : (
                <Tooltip iconName="success-unverified" tooltipText="Load the model to check its availability." />
              )}
            </div>
          ) : (
            <Tooltip iconName="" tooltipText="Please select a model." />
          )
          }
        </div>
      </div>
      <hr className="text-neutral-600 mx-5" />



      {/* 3: ATTRIBUTOR */}
      <div className="flex items-center py-5">
        <div className="text-4xl font-semibold px-6 text-neutral-600">3.</div>
        <div className="w-1/2">
          <label className="text-xl font-semibold">Attributor</label>
          <select className="w-full p-2 border rounded mt-1 text-neutral-400"
            value={props.selectedAttributor}
            onChange={e => props.onAttributorChange(e.target.value)}
            disabled={!props.isModelLoadedSuccessfully}
          >
            <option value="" disabled>Select an attributor...</option>
            {props.manifest?.attributors.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
          </select>
          <div className="mt-5 text-neutral-500 font-mono text-sm whitespace-pre-line">
            {props.statusAttributor}
          </div>
        </div>
        <div className="flex-1">
          {props.selectedAttributor === "" ? (
            <Tooltip iconName="" tooltipText="Select a wrapper, then an attributor." />
          ) : props.manifest?.attributors.some(a => a.id === props.selectedAttributor) ? (
            <Tooltip iconName="success" tooltipText="Attributor selected successfully." />
          ) : (
            <Tooltip iconName="error" tooltipText="Invalid attributor selected." />
          )}
        </div>
      </div>
      <hr className="text-neutral-600 mx-5" />


    </div>
  );
}
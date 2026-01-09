import { log } from "console";
import Tooltip from "./Tooltip";


interface ConfigurationPanelProps {
  manifest: { wrappers: string[], attributors: string[] } | null;
  selectedWrapper: string;
  modelName: string;
  selectedAttributor: string;
  onWrapperChange: (newValue: string) => void;
  onModelNameChange: (newValue: string) => void;
  onLoadClick: () => void;
  onAttributorChange: (newValue: string) => void;
}


export default function ConfigurationPanel(props: ConfigurationPanelProps) {

  return (
    <div className={"mb-20"}>
      <div className="text-3xl font-semibold py-7 text-center">Configuration</div>
      <hr className="text-neutral-600 mx-5" />



      {/* 1: MODEL SOURCE / WRAPPER */}
      <div className="flex items-center py-5">
        <div className="text-4xl font-semibold px-6 text-neutral-600">1.</div>
        <div className="w-1/2">
          <label className="text-xl font-semibold">Model Source</label>
          <select className="w-full p-2 border rounded mt-1 text-neutral-400"
            value={props.selectedWrapper}
            onChange={e => props.onWrapperChange(e.target.value)}
          >
            <option value="" disabled>Select a source...</option>
            {props.manifest?.wrappers.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
        </div>
        <div className="flex-1">
          {props.selectedWrapper === "" ? (
            <Tooltip iconName="" tooltipText="Please select a model source." />
          ) : props.manifest?.wrappers.includes(props.selectedWrapper) ? (
            <Tooltip iconName="success" tooltipText="Model source selected successfully." />
          ) : (
            <Tooltip iconName="error" tooltipText="Invalid model source selected." />
          )}
        </div>
      </div>
      <hr className="text-neutral-600 mx-5" />



      {/* 2: MODEL NAME */}
      <div className="flex items-center py-5">
        <div className="text-4xl font-semibold px-6 text-neutral-600">2.</div>
        <div className="w-1/2">
          <div className="mb-5">
            <label className="text-xl font-semibold">Search model in -nome sorgente-</label>
            <input type="text" className="w-full p-2 border rounded mt-1 text-neutral-400" />
          </div>
          <button className="bg-green-600 p-2 rounded text-white font-semibold">Load Model</button>
        </div>
        <Tooltip iconName="" tooltipText="Message" />
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
          >
            <option value="" disabled>Select a source...</option>
            {props.manifest?.attributors.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
        </div>
        <div className="flex-1">
          {props.selectedAttributor === "" ? (
            <Tooltip iconName="" tooltipText="Please select an attributor." />
          ) : props.manifest?.attributors.includes(props.selectedAttributor) ? (
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
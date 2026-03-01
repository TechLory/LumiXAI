"use client";

import Navbar from "../app/components/layout/Navbar";
import Header from "../app/components/layout/Header"; // deprecated
import ConfigurationPanel from "../app/components/panels/ConfigurationPanel";
import InputPanel from "../app/components/panels/InputPanel";
import OutputPanel from "../app/components/panels/OutputPanel";

import { useSystemBoot } from "../app/hooks/useSystemBoot";
import { useModelManager } from "../app/hooks/useModelManager";
import { useInference } from "../app/hooks/useInference";

export default function Home() {
  const { systemState, bootLogs } = useSystemBoot();
  const {
    selectedSource, setSelectedSource,
    modelName, setModelName,
    selectedAttributor, setSelectedAttributor,
    configState, handleLoadConfiguration
  } = useModelManager();
  const {
    inputText, setInputText,
    inferenceState, handleExplain
  } = useInference();

  return (
    <div className="bg-neutral-950 text-white min-h-screen">

      <Navbar />

      <div className="m-auto min-h-screen relative w-[98%] xl:w-4/5 2xl:w-3/5">

        {/* BLOCK 1 */}
        <div className="bg-neutral-800 p-6 mb-2 font-mono font-medium">
          <div className="uppercase">System logs</div>
          {/* Banner */}
          <div className={`uppercase p-2 text-sm my-2 ${systemState.status === 'running'
            ? 'bg-blue-400/10 text-blue-300'
            : systemState.status === 'error'
              ? 'bg-red-400/10 text-red-300'
              : systemState.status === 'success'
                ? 'bg-green-400/10 text-green-300'
                : 'bg-neutral-400/10 text-neutral-300'
            }`}>
            {systemState.status === 'running'
              ? '// System is currently booting and initializing all required components...'
              : systemState.status === 'error'
                ? '// System encountered a critical error during initialization. Please check the logs below.'
                : systemState.status === 'success'
                  ? '// System ready.'
                  : '// System status is unknown. Please wait...'}
          </div>
          {/* Logs */}
          <div>
            {bootLogs.map((log, index) => (
              <div key={index} className="font-mono text-neutral-600 text-sm">{log}</div>
            ))}
          </div>
          {/* Error Details */}
          {systemState.status === 'error' && (
            <div className="text-red-500 font-mono text-sm mt-0">
              <div className="uppercase">Failed</div>
              <div className="font-bold mt-2">{systemState.error}</div>
            </div>
          )}
        </div>


        {/* BLOCK 2 */}
        <div className="bg-neutral-800 p-6 mb-2 min-h-60 relative">
          <div className="font-mono font-medium uppercase">Configuration panel</div>
          {systemState.status === 'running' ? (
            // Loading
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
              <i className='bx bx-loader animate-spin text-2xl text-neutral-600'></i>
            </div>
          ) : systemState.status === 'error' ? (
            // Error
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
              FAILED
            </div>
          ) : systemState.status === 'success' && systemState.data ? (
            // OK
            <div>
              <ConfigurationPanel
                manifest={systemState.data}
                selectedSource={selectedSource}
                modelName={modelName}
                selectedAttributor={selectedAttributor}
                onSourceChange={setSelectedSource}
                onModelNameChange={setModelName}
                onAttributorChange={setSelectedAttributor}
                configState={configState}
                onLoadConfiguration={handleLoadConfiguration}
              />
            </div>
          ) : null}
        </div>


        {/* BLOCK 3 */}
        <div className="bg-neutral-800 p-6 mb-2 min-h-60 relative">
          <div className="font-mono font-medium uppercase">Input panel</div>
          {systemState.status === 'running' ? (
            // Loading
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
              <i className='bx bx-loader animate-spin text-2xl text-neutral-500'></i>
            </div>
          ) : systemState.status === 'error' ? (
            // Error
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
              FAILED
            </div>
          ) : systemState.status === 'success' && systemState.data ? (
            // OK
            <div>
              <InputPanel
                inputText={inputText}
                setInputText={setInputText}
                onExplainClick={handleExplain}
                inferenceStatus={inferenceState.status}
                isConfigReady={configState.status === 'success'}
              />
            </div>
          ) : null}
        </div>

        {/* BLOCK 4 */}
        <div className="bg-neutral-800 p-6 mb-2 min-h-60 relative">
          <div className="font-mono font-medium uppercase">Output panel</div>
          {systemState.status === 'running' ? (
            // SYSTEM LOADING
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
              <i className='bx bx-loader animate-spin text-2xl text-neutral-500'></i>
            </div>
          ) : systemState.status === 'error' ? (
            // BOOT ERROR
            <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
              FAILED
            </div>
          ) : systemState.status === 'success' && systemState.data ? (
            // OK
            <div>
              <OutputPanel outputResult={inferenceState.data} />
            </div>
          ) : null}
        </div>

      </div>
      <div className="h-60"></div>
    </div>
  );
}
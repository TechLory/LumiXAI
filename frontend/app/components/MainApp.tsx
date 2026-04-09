"use client";

import Navbar from "./layout/Navbar";
import ConfigurationPanel from "./panels/ConfigurationPanel";
import InputPanel from "./panels/InputPanel";
import OutputPanel from "./panels/OutputPanel";

import { useSystemBoot } from "../hooks/useSystemBoot";
import { useModelManager } from "../hooks/useModelManager";
import { useInference } from "../hooks/useInference";
import { useJobsHistory } from "../hooks/useJobsHistory";

export default function MainApp() {
  const { systemState, bootLogs } = useSystemBoot();

  const {
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
  } = useModelManager();

  const {
    inputText, setInputText,
    inferenceState, handleExplain, loadPastJob
  } = useInference();

  const { jobs, fetchJobPayload } = useJobsHistory();

  const handleHistoryClick = async (jobId: string, status: string, prompt: string) => {
    if (status !== 'completed') return; // Ignora se fallito o in corso

    const fullJob = await fetchJobPayload(jobId);
    if (fullJob && fullJob.payload) {
      loadPastJob(fullJob.payload, prompt);
    }
  };



  return (
    <div className="flex flex-col h-screen overflow-hidden bg-neutral-950 text-white">

      <Navbar />

      <div className="flex flex-1 overflow-hidden gap-2 min-h-0 w-full xl:w-10/12 m-auto"> {/* FIX WIDTH */}

        {/* SIDE BAR */}
        <aside className="bg-neutral-800 p-6 w-[25%] shrink-0 overflow-y-auto">
          <div className="font-mono font-medium uppercase">Job History</div>
          <div className="mt-2 flex flex-col gap-2">

            {/* NO JOBS */}
            {jobs.length === 0 && (
              <div className="text-center text-neutral-600 text-sm mt-10 font-mono italic">No previous jobs found.</div>
            )}

            {jobs.map(job => (
              <div
                key={job.id}
                onClick={() => handleHistoryClick(job.id, job.status, job.prompt)}
                className={`bg-neutral-600/30 p-1.5
                ${job.status === 'running' ? 'border-blue-400/10 border-2 cursor-wait' :
                    job.status === 'failed' ? 'border-red-900/10 border-2' :
                      'border-green-400/10 hover:border-green-400/40 border-2 cursor-pointer'
                  }
                `}>
                <div className={`p-2 text-xs font-mono flex justify-between ${job.status === 'running'
                  ? 'bg-blue-400/10 text-blue-300 animate-pulse'
                  : job.status === 'failed'
                    ? 'bg-red-400/10 text-red-300'
                    : job.status === 'completed'
                      ? 'bg-green-400/10 text-green-300'
                      : 'bg-neutral-400/10 text-neutral-300'
                  }`}>
                  <div className="uppercase">// {job.status}</div>
                  <div>
                    {job.status === 'running' && <i className='bx bx-loader animate-spin text-blue-400'></i>}
                    {job.execution_time_sec && <div>{job.execution_time_sec}s</div>}
                  </div>
                </div>
                <div className="font-mono text-sm text-neutral-200 my-5 flex italic">
                  <div>"</div>
                  <div className="truncate">{job.prompt}</div>
                  <div>"</div>
                </div>
                <div className="flex justify-between text-xs font-mono w-full text-neutral-400">
                  <div className="truncate max-w-2/3">{job.model_name.split('/').pop()}</div>
                  <div>{new Date(job.created_at).toLocaleTimeString()}</div>
                </div>
              </div>
            ))}



          </div>

        </aside>

        {/* RIGHT COLUMN */}
        <main className="flex-1 overflow-y-auto">

          {/* BLOCK 1: SYSTEM LOGS */}
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


          {/* BLOCK 2: CONFIGURATION PANEL */}
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
                  onSourceChange={onSourceChange}
                  onModelNameChange={onModelNameChange}
                  onAttributorChange={onAttributorChange}
                  configState={configState}
                  isDirty={isDirty}
                  hasActiveConfiguration={hasActiveConfiguration}
                  hasResetTarget={!!lastLoadedConfiguration}
                  isInferenceRunning={inferenceState.status === 'running'}
                  onLoadConfiguration={handleLoadConfiguration}
                  onResetConfiguration={handleResetConfiguration}
                  onUnloadConfiguration={handleUnloadConfiguration}
                />
              </div>
            ) : null}
          </div>


          {/* BLOCK 3: INPUT PANEL */}
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
                  isConfigReady={hasActiveConfiguration}
                  activeAttributorId={activeAttributorId}
                />
              </div>
            ) : null}
          </div>

          {/* BLOCK 4: OUTPUT PANEL */}
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

        </main>


      </div>
    </div>
  );
}

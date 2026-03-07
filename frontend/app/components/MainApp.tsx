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
    selectedSource, setSelectedSource,
    modelName, setModelName,
    selectedAttributor, setSelectedAttributor,
    configState, handleLoadConfiguration
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

      <div className="flex flex-1 overflow-hidden gap-2 min-h-0 w-10/12 m-auto"> {/* FIX WIDTH */}

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
                className={`p-2 ${job.status === 'completed' ? ' bg-neutral-600/30 hover:bg-neutral-600/60 cursor-pointer' :
                  job.status === 'failed' ? 'bg-red-900/10 opacity-50' :
                    'bg-blue-900/10 cursor-wait animate-pulse'
                  }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="text-xs text-neutral-500 font-mono">
                    {new Date(job.created_at).toLocaleTimeString()}
                  </div>
                  <div>
                    {job.status === 'running' && <i className='bx bx-loader animate-spin text-blue-400'></i>}
                    {job.status === 'completed' && <i className='bx bx-check text-green-500'></i>}
                    {job.status === 'failed' && <i className='bx bx-x text-red-500'></i>}
                  </div>
                </div>
                <div className="font-mono text-sm line-clamp-2 text-neutral-200 mb-2">
                  "{job.prompt}"
                </div>
                <div className="flex justify-between items-center mt-2 text-[10px] uppercase font-bold text-neutral-500">
                  <span className="truncate max-w-30">{job.model_name.split('/').pop()}</span>
                  {job.execution_time_sec && <span>{job.execution_time_sec}s</span>}
                </div>
              </div>
            ))}


            {jobs.map(job => (
              <div className="bg-neutral-600/30 p-2 text-xs font-mono font-medium text-neutral-400 flex flex-col gap-1">
                <div className="flex gap-1 text-xs text-neutral-600">
                  <div>ID:</div>
                  <div>{job.id}</div>
                </div>

                <div className="p-3 bg-neutral-600/40 flex gap-4 uppercase">
                  <div>// started at</div>
                  <div>{new Date(job.created_at).toLocaleTimeString()}</div>
                </div>
                <div className="p-3 bg-neutral-600/40 italic line-clamp-1">"{job.prompt}"</div>

                <div className="flex p-3 bg-neutral-600/40 justify-between">
                  <div className="flex gap-2 uppercase font-semibold">
                    <div>//</div>
                    <div>
                      {job.status === 'running' && <span className="text-amber-600">Running...</span>}
                      {job.status === 'failed' && <span className="text-red-500">Failed</span>}
                      {job.status === 'completed' && <span className="text-green-500">Completed</span>}
                    </div>
                  </div>
                  {job.execution_time_sec && <div>{job.execution_time_sec}s</div>}
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
                  onSourceChange={setSelectedSource}
                  onModelNameChange={setModelName}
                  onAttributorChange={setSelectedAttributor}
                  configState={configState}
                  onLoadConfiguration={handleLoadConfiguration}
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
                  isConfigReady={configState.status === 'success'}
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
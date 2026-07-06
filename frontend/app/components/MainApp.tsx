"use client";

import { useState, useEffect } from "react";
import type { MouseEvent } from "react";

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
  const [pendingDeleteJobId, setPendingDeleteJobId] = useState<string | null>(null);

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
    seed, setSeed,
    inferenceState, handleExplain, loadPastJob, handleDeletedJob
  } = useInference();

  const { jobs, deletingJobIds, fetchJobPayload, deleteJob } = useJobsHistory();

  // --- COLLAPSE STATE (setup blocks fold away once they've done their job) ---
  const [logsOpen, setLogsOpen] = useState(true);
  const [configOpen, setConfigOpen] = useState(true);
  // Set while reviewing a past job (no live config is loaded in that case, so
  // hasActiveConfiguration can't drive the collapse). Holds the reviewed model.
  const [reviewedJob, setReviewedJob] = useState<{ modelName: string } | null>(null);

  // System Logs: expanded while booting or on error, auto-collapse once ready.
  // Effect only fires on status transitions, so manual toggles afterward stick.
  useEffect(() => {
    if (systemState.status === 'success') setLogsOpen(false);
    else setLogsOpen(true);
  }, [systemState.status]);

  // Configuration: collapse to a summary once a config is active, re-open when
  // unloaded. Loading a config also exits past-job review mode.
  useEffect(() => {
    setConfigOpen(!hasActiveConfiguration);
    if (hasActiveConfiguration) setReviewedJob(null);
  }, [hasActiveConfiguration]);

  // Status badge shown in the collapsed System Logs header.
  const statusMeta =
    systemState.status === 'running'
      ? { dot: 'bg-blue-400', text: 'System booting…', cls: 'text-blue-300' }
      : systemState.status === 'error'
        ? { dot: 'bg-red-400', text: 'System error', cls: 'text-red-300' }
        : systemState.status === 'success'
          ? { dot: 'bg-green-400', text: 'System ready', cls: 'text-green-300' }
          : { dot: 'bg-neutral-400', text: 'System status unknown', cls: 'text-neutral-300' };

  // Summary chip shown in the collapsed Configuration header.
  const activeAttributorName =
    systemState.data?.attributors?.find(a => a.id === selectedAttributor)?.name ?? selectedAttributor;
  const shortModelName = modelName?.split('/').pop();
  const configReady = systemState.status === 'success';
  const isReviewingPastJob = !hasActiveConfiguration && !!reviewedJob;
  const configCollapsible = configReady && (hasActiveConfiguration || isReviewingPastJob);
  const showConfigBody = !configCollapsible || configOpen;

  const handleHistoryClick = async (jobId: string, status: string, prompt: string, modelName: string) => {
    if (status !== 'completed') return; // Ignora se fallito o in corso

    const fullJob = await fetchJobPayload(jobId);
    if (fullJob && fullJob.payload) {
      loadPastJob(fullJob.payload, prompt);
      // Reviewing a past result: fold the setup away and show the reviewed model.
      setReviewedJob({ modelName });
      setConfigOpen(false);
    }
  };

  const suppressCardClick = (event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const requestDeleteJob = (event: MouseEvent<HTMLButtonElement>, jobId: string) => {
    suppressCardClick(event);
    setPendingDeleteJobId(currentJobId => currentJobId === jobId ? null : jobId);
  };

  const cancelDeleteJob = (event: MouseEvent<HTMLButtonElement>) => {
    suppressCardClick(event);
    setPendingDeleteJobId(null);
  };

  const handleDeleteJob = async (event: MouseEvent<HTMLButtonElement>, jobId: string) => {
    suppressCardClick(event);

    try {
      await deleteJob(jobId);
      handleDeletedJob(jobId);
      setPendingDeleteJobId(null);
    } catch (e: any) {
      window.alert(e.message || "Failed to delete job.");
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
                onClick={() => handleHistoryClick(job.id, job.status, job.prompt, job.model_name)}
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
                  <div
                    className="flex items-center gap-2"
                    onMouseDown={suppressCardClick}
                    onClick={suppressCardClick}
                  >
                    {job.status === 'running' && <i className='bx bx-loader animate-spin text-blue-400'></i>}
                    {job.execution_time_sec && <div>{job.execution_time_sec}s</div>}
                    {pendingDeleteJobId === job.id ? (
                      <>
                        <button
                          type="button"
                          onClick={(event) => handleDeleteJob(event, job.id)}
                          disabled={deletingJobIds.includes(job.id)}
                          className="px-2 py-1 border border-red-400/50 bg-red-400/10 text-red-200 hover:bg-red-400/20 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                          aria-label={`Confirm delete job ${job.id}`}
                        >
                          {deletingJobIds.includes(job.id) ? (
                            <i className='bx bx-loader animate-spin text-base'></i>
                          ) : (
                            "Delete"
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={cancelDeleteJob}
                          disabled={deletingJobIds.includes(job.id)}
                          className="px-2 py-1 border border-neutral-500/50 bg-neutral-700/20 text-neutral-200 hover:bg-neutral-700/40 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                          aria-label={`Cancel delete job ${job.id}`}
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        onClick={(event) => requestDeleteJob(event, job.id)}
                        disabled={deletingJobIds.includes(job.id)}
                        className="text-neutral-400 hover:text-red-300 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        aria-label={`Delete job ${job.id}`}
                      >
                        <i className={`bx ${deletingJobIds.includes(job.id) ? 'bx-loader animate-spin' : 'bx-trash'} text-base`}></i>
                      </button>
                    )}
                  </div>
                </div>
                {pendingDeleteJobId === job.id && (
                  <div
                    className="mt-1 px-2 py-1 text-[11px] font-mono uppercase bg-red-400/10 text-red-200 border border-red-400/20"
                    onMouseDown={suppressCardClick}
                    onClick={suppressCardClick}
                  >
                    This will permanently remove the job from history and the dataset.
                  </div>
                )}
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
        <main className="flex-1 overflow-y-auto flex flex-col gap-2">

          {/* BLOCK 1: SYSTEM LOGS (collapses to a status badge once ready) */}
          <div className="bg-neutral-800 p-6 font-mono font-medium shrink-0">
            <button
              type="button"
              onClick={() => setLogsOpen(open => !open)}
              className="w-full flex items-center justify-between gap-4 cursor-pointer"
            >
              <div className="uppercase">System Logs</div>
              <div className="flex items-center gap-3">
                {!logsOpen && (
                  <div className={`flex items-center gap-2 text-sm normal-case ${statusMeta.cls}`}>
                    <span className={`inline-block w-2 h-2 rounded-full ${statusMeta.dot} ${systemState.status === 'running' ? 'animate-pulse' : ''}`}></span>
                    {statusMeta.text}
                  </div>
                )}
                <i className={`bx ${logsOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-neutral-500`}></i>
              </div>
            </button>

            {logsOpen && (
              <>
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
              </>
            )}
          </div>


          {/* BLOCK 2: CONFIGURATION (collapses to a summary chip once loaded) */}
          <div className="bg-neutral-800 p-6 relative shrink-0">
            {configCollapsible ? (
              <button
                type="button"
                onClick={() => setConfigOpen(open => !open)}
                className="w-full flex items-center justify-between gap-4 cursor-pointer"
              >
                <div className="font-mono font-medium uppercase">Configuration</div>
                <div className="flex items-center gap-3">
                  {!configOpen && (
                    isReviewingPastJob ? (
                      <div className="flex items-center gap-2 text-sm font-mono text-neutral-300">
                        <i className="bx bx-history text-neutral-400 text-base"></i>
                        <span className="text-neutral-200">{reviewedJob?.modelName?.split('/').pop()}</span>
                        <span className="text-neutral-500 normal-case text-xs italic">reviewing</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm font-mono text-neutral-300">
                        <i className="bx bx-check text-emerald-400 text-base"></i>
                        <span className="text-neutral-200">{shortModelName}</span>
                        <span className="text-neutral-600">·</span>
                        <span className="text-neutral-400">{activeAttributorName}</span>
                      </div>
                    )
                  )}
                  <i className={`bx ${configOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-neutral-500`}></i>
                </div>
              </button>
            ) : (
              <div className="font-mono font-medium uppercase">Configuration</div>
            )}

            {systemState.status === 'running' ? (
              // Loading
              <div className="min-h-60 flex justify-center items-center">
                <i className='bx bx-loader animate-spin text-2xl text-neutral-600'></i>
              </div>
            ) : systemState.status === 'error' ? (
              // Error
              <div className="min-h-60 flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
                FAILED
              </div>
            ) : systemState.status === 'success' && systemState.data && showConfigBody ? (
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


          {/* WORKSPACE: INPUT + OUTPUT side by side on wide screens, stacked below 2xl */}
          <div className="flex flex-col 2xl:flex-row gap-2 2xl:items-start">

            {/* INPUT */}
            <div className="bg-neutral-800 p-6 min-h-60 relative flex-1 min-w-0 2xl:basis-1/2">
              <div className="font-mono font-medium uppercase">Input</div>
              {systemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-neutral-500'></i>
                </div>
              ) : systemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
                  FAILED
                </div>
              ) : systemState.status === 'success' && systemState.data ? (
                <div>
                  <InputPanel
                    inputText={inputText}
                    setInputText={setInputText}
                    seed={seed}
                    setSeed={setSeed}
                    onExplainClick={handleExplain}
                    inferenceStatus={inferenceState.status}
                    isConfigReady={hasActiveConfiguration}
                    activeAttributorId={activeAttributorId}
                  />
                </div>
              ) : null}
            </div>

            {/* OUTPUT */}
            <div className="bg-neutral-800 p-6 min-h-60 relative flex-1 min-w-0 2xl:basis-1/2">
              <div className="font-mono font-medium uppercase">Output</div>
              {systemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-neutral-500'></i>
                </div>
              ) : systemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-neutral-600 text-sm">
                  FAILED
                </div>
              ) : systemState.status === 'success' && systemState.data ? (
                <div>
                  <OutputPanel outputResult={inferenceState.data} />
                </div>
              ) : null}
            </div>

          </div>

        </main>


      </div>
    </div>
  );
}

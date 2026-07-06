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
import type { TutorialKind } from "../types";

type MainAppProps = {
  activeTutorial?: TutorialKind | null;
  onOpenWelcome?: () => void;
  onSelectTutorial?: (tutorial: TutorialKind) => void;
};

export default function MainApp({ activeTutorial = null, onOpenWelcome, onSelectTutorial }: MainAppProps) {
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
      ? { dot: 'bg-info', text: 'System booting…', cls: 'text-info' }
      : systemState.status === 'error'
        ? { dot: 'bg-danger', text: 'System error', cls: 'text-danger' }
        : systemState.status === 'success'
          ? { dot: 'bg-ok', text: 'System ready', cls: 'text-ok' }
          : { dot: 'bg-fg-faint', text: 'System status unknown', cls: 'text-fg-subtle' };

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
    <div className="flex flex-col h-screen overflow-hidden bg-page text-fg">

      <Navbar
        activeTutorial={activeTutorial}
        onOpenWelcome={onOpenWelcome}
        onSelectTutorial={onSelectTutorial}
      />

      <div className="flex flex-1 overflow-hidden gap-2 min-h-0 w-full xl:w-10/12 m-auto"> {/* FIX WIDTH */}

        {/* SIDE BAR */}
        <aside className="bg-surface p-6 w-[25%] shrink-0 overflow-y-auto">
          <div className="font-mono font-medium uppercase">Job History</div>
          <div className="mt-2 flex flex-col gap-2">

            {/* NO JOBS */}
            {jobs.length === 0 && (
              <div className="text-center text-fg-faint text-sm mt-10 font-mono italic">No previous jobs found.</div>
            )}

            {jobs.map(job => (
              <div
                key={job.id}
                onClick={() => handleHistoryClick(job.id, job.status, job.prompt, job.model_name)}
                className={`bg-fill p-1.5
                ${job.status === 'running' ? 'border-info-line border-2 cursor-wait' :
                    job.status === 'failed' ? 'border-danger-line border-2' :
                      'border-ok-line hover:border-ok border-2 cursor-pointer'
                  }
                `}>
                <div className={`p-2 text-xs font-mono flex justify-between ${job.status === 'running'
                  ? 'bg-info-soft text-info animate-pulse'
                  : job.status === 'failed'
                    ? 'bg-danger-soft text-danger'
                    : job.status === 'completed'
                      ? 'bg-ok-soft text-ok'
                      : 'bg-fill text-fg-subtle'
                  }`}>
                  <div className="uppercase">// {job.is_builtin_example ? "example" : job.status}</div>
                  <div
                    className="flex items-center gap-2"
                    onMouseDown={suppressCardClick}
                    onClick={suppressCardClick}
                  >
                    {job.status === 'running' && <i className='bx bx-loader animate-spin text-info'></i>}
                    {job.execution_time_sec && <div>{job.execution_time_sec}s</div>}
                    {job.is_builtin_example ? (
                      <div className="px-2 py-1 border border-info-line bg-info-soft text-info">Built-in</div>
                    ) : pendingDeleteJobId === job.id ? (
                      <>
                        <button
                          type="button"
                          onClick={(event) => handleDeleteJob(event, job.id)}
                          disabled={deletingJobIds.includes(job.id)}
                          className="px-2 py-1 border border-danger-line bg-danger-soft text-danger hover:bg-danger-hover transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
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
                          className="px-2 py-1 border border-border-strong bg-fill text-fg-muted hover:bg-fill-strong transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
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
                        className="text-fg-subtle hover:text-danger transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        aria-label={`Delete job ${job.id}`}
                      >
                        <i className={`bx ${deletingJobIds.includes(job.id) ? 'bx-loader animate-spin' : 'bx-trash'} text-base`}></i>
                      </button>
                    )}
                  </div>
                </div>
                {pendingDeleteJobId === job.id && (
                  <div
                    className="mt-1 px-2 py-1 text-[11px] font-mono uppercase bg-danger-soft text-danger border border-danger-line"
                    onMouseDown={suppressCardClick}
                    onClick={suppressCardClick}
                  >
                    This will permanently remove the job from history and the dataset.
                  </div>
                )}
                <div className="font-mono text-sm text-fg-muted my-5 flex italic">
                  <div>"</div>
                  <div className="truncate">{job.prompt}</div>
                  <div>"</div>
                </div>
                <div className="flex justify-between text-xs font-mono w-full text-fg-subtle">
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
          <div className="bg-surface p-6 font-mono font-medium shrink-0">
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
                <i className={`bx ${logsOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-fg-faint`}></i>
              </div>
            </button>

            {logsOpen && (
              <>
                {/* Banner */}
                <div className={`uppercase p-2 text-sm my-2 ${systemState.status === 'running'
                  ? 'bg-info-soft text-info'
                  : systemState.status === 'error'
                    ? 'bg-danger-soft text-danger'
                    : systemState.status === 'success'
                      ? 'bg-ok-soft text-ok'
                      : 'bg-fill text-fg-subtle'
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
                    <div key={index} className="font-mono text-fg-faint text-sm">{log}</div>
                  ))}
                </div>
                {/* Error Details */}
                {systemState.status === 'error' && (
                  <div className="text-danger font-mono text-sm mt-0">
                    <div className="uppercase">Failed</div>
                    <div className="font-bold mt-2">{systemState.error}</div>
                  </div>
                )}
              </>
            )}
          </div>


          {/* BLOCK 2: CONFIGURATION (collapses to a summary chip once loaded) */}
          <div className="bg-surface p-6 relative shrink-0">
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
                      <div className="flex items-center gap-2 text-sm font-mono text-fg-subtle">
                        <i className="bx bx-history text-fg-subtle text-base"></i>
                        <span className="text-fg-muted">{reviewedJob?.modelName?.split('/').pop()}</span>
                        <span className="text-fg-faint normal-case text-xs italic">reviewing</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm font-mono text-fg-subtle">
                        <i className="bx bx-check text-ok text-base"></i>
                        <span className="text-fg-muted">{shortModelName}</span>
                        <span className="text-fg-faint">·</span>
                        <span className="text-fg-subtle">{activeAttributorName}</span>
                      </div>
                    )
                  )}
                  <i className={`bx ${configOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-fg-faint`}></i>
                </div>
              </button>
            ) : (
              <div className="font-mono font-medium uppercase">Configuration</div>
            )}

            {systemState.status === 'running' ? (
              // Loading
              <div className="min-h-60 flex justify-center items-center">
                <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
              </div>
            ) : systemState.status === 'error' ? (
              // Error
              <div className="min-h-60 flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
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
            <div className="bg-surface p-6 min-h-60 relative flex-1 min-w-0 2xl:basis-1/2">
              <div className="font-mono font-medium uppercase">Input</div>
              {systemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
                </div>
              ) : systemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
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
            <div className="bg-surface p-6 min-h-60 relative flex-1 min-w-0 2xl:basis-1/2">
              <div className="font-mono font-medium uppercase">Output</div>
              {systemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
                </div>
              ) : systemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
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

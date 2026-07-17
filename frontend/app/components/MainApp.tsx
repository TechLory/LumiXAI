"use client";

import { useState, useEffect, useRef } from "react";
import type { MouseEvent } from "react";

import Navbar from "./layout/Navbar";
import ConfigurationPanel from "./panels/ConfigurationPanel";
import InputPanel from "./panels/InputPanel";
import OutputPanel from "./panels/OutputPanel";
import TutorialOverlay from "./tutorial/TutorialOverlay";

import { useSystemBoot } from "../hooks/useSystemBoot";
import { useModelManager } from "../hooks/useModelManager";
import { useBackendStatus } from "../hooks/useBackendStatus";
import { useInference } from "../hooks/useInference";
import { useJobsHistory } from "../hooks/useJobsHistory";
import { getTutorialExampleForKind, loadTutorialExamplePayload } from "../lib/tutorialExamples";
import { getTutorialSteps } from "../lib/tutorialGuide";
import { guessWrapperFromTask } from "../lib/taskToWrapper";
import type { OutputResult } from "./panels/OutputPanel";
import type { ResultMetadata, TutorialKind } from "../types";

type MainAppProps = {
  activeTutorial?: TutorialKind | null;
  onOpenWelcome?: () => void;
  onSelectTutorial?: (tutorial: TutorialKind) => void;
  onCloseTutorial?: () => void;
};

export default function MainApp({ activeTutorial = null, onOpenWelcome, onSelectTutorial, onCloseTutorial }: MainAppProps) {
  const { systemState, bootLogs } = useSystemBoot();
  const [pendingDeleteJobId, setPendingDeleteJobId] = useState<string | null>(null);

  const tutorialExample = activeTutorial ? getTutorialExampleForKind(activeTutorial) : null;
  const isTutorialActive = !!activeTutorial && !!tutorialExample;
  const tutorialConfig = tutorialExample?.config;
  const tutorialManifest = tutorialConfig ? {
    sources: [{ id: tutorialConfig.sourceId, name: tutorialConfig.sourceName, type: "remote" }],
    attributors: [{
      id: tutorialConfig.attributorId,
      name: tutorialConfig.attributorName,
      compatible_wrappers: [guessWrapperFromTask(tutorialConfig.detectedTask)].filter((w): w is string => !!w)
    }]
  } : null;
  const effectiveSystemState = isTutorialActive && tutorialManifest
    ? { status: 'success' as const, data: tutorialManifest, error: null }
    : systemState;

  const {
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
  } = useModelManager(effectiveSystemState.data?.attributors ?? []);

  // Tutorial mode fakes an active configuration without ever loading anything, so polling
  // the real backend there would immediately "correct" that state away.
  const backendStatus = useBackendStatus(!isTutorialActive);

  const {
    inputText, setInputText,
    inputImageBase64, setInputImageBase64,
    inputImageFileName, setInputImageFileName,
    seed, setSeed,
    maxNewTokens, setMaxNewTokens,
    inferenceState, resultMetadata, handleExplain, loadPastJob, resetInferenceState, handleDeletedJob
  } = useInference();

  const { jobs, deletingJobIds, fetchJobPayload, deleteJob, togglePinJob } = useJobsHistory();

  const tutorialSteps = activeTutorial ? getTutorialSteps(activeTutorial) : [];
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialPayload, setTutorialPayload] = useState<OutputResult | null>(null);

  // The example's output payload is real, previously-computed data shipped as a static
  // fixture; fetch it as soon as the tutorial activates so it's ready by the "result" step.
  useEffect(() => {
    if (!tutorialExample) {
      setTutorialPayload(null);
      return;
    }

    let cancelled = false;
    loadTutorialExamplePayload(tutorialExample).then((payload) => {
      if (!cancelled) setTutorialPayload(payload);
    });
    return () => {
      cancelled = true;
    };
  }, [tutorialExample?.id]);
  const currentTutorialStep = isTutorialActive ? tutorialSteps[Math.min(tutorialStepIndex, tutorialSteps.length - 1)] : null;
  const effectiveBootLogs = isTutorialActive
    ? ["Tutorial mode active.", "Using bundled prepared example data. Backend calls are skipped."]
    : bootLogs;

  useEffect(() => {
    setTutorialStepIndex(0);
  }, [activeTutorial]);

  useEffect(() => {
    if (!isTutorialActive || !tutorialExample || !tutorialConfig || !currentTutorialStep) {
      return;
    }

    const phase = currentTutorialStep.phase;
    const includeModel = phase !== "source";
    const includeAttributor = phase !== "source" && phase !== "model";
    const isLoaded = phase === "configuration" || phase === "input" || phase === "result";

    hydrateConfiguration({
      source: tutorialConfig.sourceId,
      sourceName: tutorialConfig.sourceName,
      modelName: includeModel ? tutorialConfig.modelName : "",
      attributor: includeAttributor ? tutorialConfig.attributorId : "",
      attributorName: tutorialConfig.attributorName,
      detectedTask: tutorialConfig.detectedTask,
    }, isLoaded);

    setPendingDeleteJobId(null);
    setReviewedJob(null);
    setSeed("");
    setLogsOpen(false);
    setConfigOpen(phase === "source" || phase === "model" || phase === "attributor" || phase === "configuration");

    if (phase === "input") {
      resetInferenceState(tutorialExample.prompt);
      if (tutorialConfig.detectedTask === "image-classification" && tutorialPayload?.input_image) {
        setInputImageBase64(tutorialPayload.input_image);
        setInputImageFileName(tutorialExample.prompt);
      }
      return;
    }

    if (phase === "result") {
      if (tutorialPayload) {
        loadPastJob(tutorialPayload, tutorialExample.prompt, {
          modelName: tutorialConfig.modelName,
          attributorName: tutorialConfig.attributorName,
        });
        if (tutorialConfig.detectedTask === "image-classification" && tutorialPayload.input_image) {
          setInputImageBase64(tutorialPayload.input_image);
          setInputImageFileName(tutorialExample.prompt);
        }
      }
      return;
    }

    resetInferenceState("");
  }, [activeTutorial, tutorialStepIndex, tutorialPayload]);

  // --- COLLAPSE STATE (setup blocks fold away once they've done their job) ---
  const [logsOpen, setLogsOpen] = useState(true);
  const [configOpen, setConfigOpen] = useState(true);
  // Set while reviewing a past job (no live config is loaded in that case, so
  // hasActiveConfiguration can't drive the collapse). Holds the reviewed model.
  const [reviewedJob, setReviewedJob] = useState<ResultMetadata | null>(null);

  // System Logs: expanded while booting or on error, auto-collapse once ready.
  // Effect only fires on status transitions, so manual toggles afterward stick.
  useEffect(() => {
    if (effectiveSystemState.status === 'success') setLogsOpen(false);
    else setLogsOpen(true);
  }, [effectiveSystemState.status]);

  // Configuration: collapse to a summary once a config is active, re-open when
  // unloaded. Loading a config also exits past-job review mode.
  useEffect(() => {
    if (isTutorialActive) return;

    setConfigOpen(!hasActiveConfiguration);
    if (hasActiveConfiguration) setReviewedJob(null);
  }, [hasActiveConfiguration, isTutorialActive]);

  // When the configuration went live, in client-clock terms. Compared against the poll's
  // own send time below to ignore replies that left before the model was loaded.
  const configActivatedAtRef = useRef(0);
  useEffect(() => {
    if (hasActiveConfiguration) configActivatedAtRef.current = Date.now();
  }, [hasActiveConfiguration]);

  // Give up the active configuration when the backend no longer holds the one this tab
  // loaded — the idle reaper reclaimed it, or another session took the model over. Either
  // way the UI must stop offering Explain against a configuration that isn't there.
  useEffect(() => {
    if (isTutorialActive || !backendStatus || !hasActiveConfiguration) return;
    if (backendStatus.requestedAt < configActivatedAtRef.current) return;

    if (!backendStatus.model_loaded) {
      const unloadedModel = backendStatus.last_unload?.model_name
        ? `'${backendStatus.last_unload.model_name}'`
        : "The model";
      notifyBackendUnload(
        backendStatus.last_unload?.reason === 'idle_timeout'
          ? `${unloadedModel} was unloaded to free memory after a period of inactivity. Click "Load Configuration" to bring it back.`
          : `${unloadedModel} is no longer loaded on the backend. Click "Load Configuration" to bring it back.`
      );
      return;
    }

    // A model is loaded, but not the one this tab configured: another session replaced it.
    // Compared by token, so an attributor change counts as a different configuration too.
    if (activeConfigId && backendStatus.config_id !== activeConfigId) {
      const activeModel = backendStatus.model_name ? `'${backendStatus.model_name}'` : "another configuration";
      notifyBackendUnload(
        `Another session loaded ${activeModel} on the backend, replacing your configuration. Click "Load Configuration" to take it back.`
      );
    }
  }, [backendStatus, hasActiveConfiguration, isTutorialActive, activeConfigId, notifyBackendUnload]);

  // Status badge shown in the collapsed System Logs header.
  const statusMeta =
    effectiveSystemState.status === 'running'
      ? { dot: 'bg-info', text: 'System booting…', cls: 'text-info' }
      : effectiveSystemState.status === 'error'
        ? { dot: 'bg-danger', text: 'System error', cls: 'text-danger' }
        : effectiveSystemState.status === 'success'
          ? { dot: 'bg-ok', text: 'System ready', cls: 'text-ok' }
          : { dot: 'bg-fg-faint', text: 'System status unknown', cls: 'text-fg-subtle' };

  // Summary chip shown in the collapsed Configuration header.
  const getAttributorName = (attributorId?: string | null) =>
    attributorId
      ? effectiveSystemState.data?.attributors?.find(a => a.id === attributorId)?.name ?? attributorId
      : "";
  const activeAttributorName = getAttributorName(selectedAttributor);
  const getLoadedResultMetadata = (): ResultMetadata | null => {
    if (!lastLoadedConfiguration) return null;

    return {
      modelName: lastLoadedConfiguration.modelName,
      attributorName: getAttributorName(lastLoadedConfiguration.attributor),
    };
  };
  const shortModelName = modelName?.split('/').pop();

  // Countdown to the backend's idle unload, so a model vanishing is never a surprise.
  // Refreshes on the status poll, hence the coarse rounding.
  const idleUnloadLabel = (() => {
    if (isTutorialActive || !hasActiveConfiguration || !backendStatus?.model_loaded) return null;

    const secondsLeft = backendStatus.seconds_until_unload;
    if (secondsLeft === null) return null;
    if (backendStatus.busy) return "in use";
    if (secondsLeft < 60) return "idle unload imminent";
    return `idle unload in ~${Math.round(secondsLeft / 60)}m`;
  })();
  const inputWrapperName = hasActiveConfiguration
    ? guessWrapperFromTask(lastLoadedConfiguration?.detectedTask)
    : detectedWrapperName;
  const configReady = effectiveSystemState.status === 'success';
  const isReviewingPastJob = !hasActiveConfiguration && !!reviewedJob;
  const configCollapsible = configReady && (hasActiveConfiguration || isReviewingPastJob);
  const showConfigBody = !configCollapsible || configOpen;

  const handleHistoryClick = async (jobId: string, status: string, prompt: string, modelName: string, attributorName: string) => {
    if (status !== 'completed') return; // Ignora se fallito o in corso

    const fullJob = await fetchJobPayload(jobId);
    if (fullJob && fullJob.payload) {
      const reviewedMetadata = {
        modelName: fullJob.model_name ?? modelName,
        attributorName: fullJob.attributor_name ?? attributorName,
      };
      if (isTutorialActive) {
        setTutorialStepIndex(0);
        onCloseTutorial?.();
      }
      loadPastJob(fullJob.payload, prompt, reviewedMetadata);
      // Reviewing a past result: fold the setup away and show the reviewed model.
      setReviewedJob(reviewedMetadata);
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

  const handleTogglePin = (event: MouseEvent<HTMLButtonElement>, jobId: string) => {
    suppressCardClick(event);
    togglePinJob(jobId);
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

  const closeTutorial = () => {
    setTutorialStepIndex(0);
    onCloseTutorial?.();
  };

  const goToNextTutorialStep = () => {
    if (!activeTutorial || tutorialSteps.length === 0) return;

    if (tutorialStepIndex >= tutorialSteps.length - 1) {
      setTutorialStepIndex(0);
      onCloseTutorial?.();
      return;
    }
    setTutorialStepIndex(currentIndex => currentIndex + 1);
  };

  const goToPreviousTutorialStep = () => {
    setTutorialStepIndex(currentIndex => Math.max(0, currentIndex - 1));
  };

  const handleTutorialLoadConfiguration = () => {
    if (!isTutorialActive) return;
    setTutorialStepIndex(currentIndex => Math.max(currentIndex, 4));
  };

  const handleTutorialExplain = () => {
    if (!isTutorialActive) return;
    setTutorialStepIndex(currentIndex => Math.max(currentIndex, 6));
  };

  const handleLiveExplain = (ignoreSpecialTokens: boolean, disableThinking: boolean) => {
    handleExplain(ignoreSpecialTokens, disableThinking, getLoadedResultMetadata(), activeConfigId, notifyBackendUnload);
  };

  const getTutorialTargetClass = (target: string) => (
    currentTutorialStep?.target === target ? " tutorial-highlight" : ""
  );

  const getTutorialFocusClass = (focusTarget: string) => (
    currentTutorialStep?.focusTarget === focusTarget ? " tutorial-inner-highlight" : ""
  );

  return (
    <div className="flex min-h-dvh flex-col overflow-y-auto bg-page text-fg lg:h-screen lg:overflow-hidden">

      <Navbar
        activeTutorial={activeTutorial}
        onOpenWelcome={onOpenWelcome}
        onSelectTutorial={onSelectTutorial}
      />

      <div className="flex w-full max-w-[2200px] flex-1 flex-col gap-2 px-2 pb-2 lg:min-h-0 lg:flex-row lg:overflow-hidden xl:px-4">

        {/* SIDE BAR */}
        <aside className={`bg-surface p-4 w-full max-h-48 shrink-0 overflow-y-auto sm:p-6 lg:w-72 lg:max-h-none xl:w-[25%] xl:max-w-sm${getTutorialTargetClass("history")}`}>
          <div className="font-mono font-medium uppercase">Job History</div>
          <div className="mt-2 flex flex-col gap-2">

            {/* NO JOBS */}
            {jobs.length === 0 && (
              <div className="text-center text-fg-faint text-sm mt-10 font-mono italic">No previous jobs found.</div>
            )}

            {jobs.map(job => (
              <div
                key={job.id}
                onClick={() => handleHistoryClick(job.id, job.status, job.prompt, job.model_name, job.attributor_name)}
                className={`bg-fill p-1.5
                ${job.status === 'running' ? 'border-info-line border-2 cursor-wait' :
                    job.status === 'failed' ? 'border-danger-line border-2' :
                      'border-ok-line hover:border-ok border-2 cursor-pointer'
                  }
                  ${job.tutorial_kind === activeTutorial ? getTutorialFocusClass("history-item") : ""}
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
                    {!job.is_builtin_example && job.status !== 'running' && (
                      <button
                        type="button"
                        onClick={(event) => handleTogglePin(event, job.id)}
                        className={`transition-colors cursor-pointer ${job.pinned ? "text-info" : "text-fg-subtle hover:text-info"}`}
                        aria-label={job.pinned ? `Unpin job ${job.id}` : `Pin job ${job.id}`}
                        title={job.pinned ? "Unpin" : "Pin to top"}
                      >
                        <i className={`bx ${job.pinned ? "bxs-pin" : "bx-pin"} text-base`}></i>
                      </button>
                    )}
                    {job.is_builtin_example ? (
                      <div className="flex items-center gap-2 px-2 py-1 border border-info-line bg-info-soft text-info">
                        <i className="bx bxs-pin text-base" title="Pinned"></i>
                        Built-in
                      </div>
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
        <main className="flex flex-1 min-w-0 flex-col gap-2 lg:min-h-0 lg:overflow-y-auto">

          {/* BLOCK 1: SYSTEM LOGS (collapses to a status badge once ready) */}
          <div className={`bg-surface p-6 font-mono font-medium shrink-0${getTutorialTargetClass("system")}`}>
            <button
              type="button"
              onClick={() => setLogsOpen(open => !open)}
              className="w-full flex items-center justify-between gap-4 cursor-pointer"
            >
              <div className="uppercase">System Logs</div>
              <div className="flex items-center gap-3">
                {!logsOpen && (
                  <div className={`flex items-center gap-2 text-sm normal-case ${statusMeta.cls}`}>
                    <span className={`inline-block w-2 h-2 rounded-full ${statusMeta.dot} ${effectiveSystemState.status === 'running' ? 'animate-pulse' : ''}`}></span>
                    {statusMeta.text}
                  </div>
                )}
                <i className={`bx ${logsOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-fg-faint`}></i>
              </div>
            </button>

            {logsOpen && (
              <>
                {/* Banner */}
                <div className={`uppercase p-2 text-sm my-2 ${effectiveSystemState.status === 'running'
                  ? 'bg-info-soft text-info'
                  : effectiveSystemState.status === 'error'
                    ? 'bg-danger-soft text-danger'
                    : effectiveSystemState.status === 'success'
                      ? 'bg-ok-soft text-ok'
                      : 'bg-fill text-fg-subtle'
                  }`}>
                  {effectiveSystemState.status === 'running'
                    ? '// System is currently booting and initializing all required components...'
                    : effectiveSystemState.status === 'error'
                      ? '// System encountered a critical error during initialization. Please check the logs below.'
                      : effectiveSystemState.status === 'success'
                        ? '// System ready.'
                        : '// System status is unknown. Please wait...'}
                </div>
                {/* Logs */}
                <div>
                  {effectiveBootLogs.map((log, index) => (
                    <div key={index} className="font-mono text-fg-faint text-sm">{log}</div>
                  ))}
                </div>
                {/* Error Details */}
                {effectiveSystemState.status === 'error' && (
                  <div className="text-danger font-mono text-sm mt-0">
                    <div className="uppercase">Failed</div>
                    <div className="font-bold mt-2">{systemState.error}</div>
                  </div>
                )}
              </>
            )}
          </div>


          {/* BLOCK 2: CONFIGURATION (collapses to a summary chip once loaded) */}
          <div className={`bg-surface p-6 relative shrink-0${getTutorialTargetClass("configuration")}`}>
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
                        {idleUnloadLabel && (
                          <>
                            <span className="text-fg-faint">·</span>
                            <span
                              className="text-fg-faint normal-case text-xs italic"
                              title="The backend frees the model's memory once it goes unused. Running a job resets the timer."
                            >
                              {idleUnloadLabel}
                            </span>
                          </>
                        )}
                      </div>
                    )
                  )}
                  <i className={`bx ${configOpen ? 'bx-chevron-up' : 'bx-chevron-down'} text-xl text-fg-faint`}></i>
                </div>
              </button>
            ) : (
              <div className="font-mono font-medium uppercase">Configuration</div>
            )}

            {effectiveSystemState.status === 'running' ? (
              // Loading
              <div className="min-h-60 flex justify-center items-center">
                <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
              </div>
            ) : effectiveSystemState.status === 'error' ? (
              // Error
              <div className="min-h-60 flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
                FAILED
              </div>
            ) : effectiveSystemState.status === 'success' && effectiveSystemState.data && showConfigBody ? (
              // OK
              <div>
                <ConfigurationPanel
                  manifest={effectiveSystemState.data}
                  selectedSource={selectedSource}
                  modelName={modelName}
                  selectedAttributor={selectedAttributor}
                  detectedTask={detectedTask}
                  detectedWrapperName={detectedWrapperName}
                  onSourceChange={onSourceChange}
                  onModelNameChange={onModelNameChange}
                  onAttributorChange={onAttributorChange}
                  configState={configState}
                  isDirty={isDirty}
                  hasActiveConfiguration={hasActiveConfiguration}
                  hasResetTarget={!!lastLoadedConfiguration}
                  isInferenceRunning={inferenceState.status === 'running'}
                  onLoadConfiguration={isTutorialActive ? handleTutorialLoadConfiguration : handleLoadConfiguration}
                  onResetConfiguration={handleResetConfiguration}
                  onUnloadConfiguration={handleUnloadConfiguration}
                  tutorialFocusTarget={currentTutorialStep?.focusTarget}
                />
              </div>
            ) : null}
          </div>


          {/* WORKSPACE: INPUT + OUTPUT side by side only when there is enough room. */}
          <div className="workspace-grid min-w-0">

            {/* INPUT */}
            <div className={`bg-surface p-4 min-h-60 relative min-w-0 sm:p-6${getTutorialTargetClass("input")}`}>
              <div className="font-mono font-medium uppercase">Input</div>
              {effectiveSystemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
                </div>
              ) : effectiveSystemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
                  FAILED
                </div>
              ) : effectiveSystemState.status === 'success' && effectiveSystemState.data ? (
                <div className="min-w-0">
                  <InputPanel
                    inputText={inputText}
                    setInputText={setInputText}
                    inputImageBase64={inputImageBase64}
                    setInputImageBase64={setInputImageBase64}
                    inputImageFileName={inputImageFileName}
                    setInputImageFileName={setInputImageFileName}
                    seed={seed}
                    setSeed={setSeed}
                    maxNewTokens={maxNewTokens}
                    setMaxNewTokens={setMaxNewTokens}
                    onExplainClick={isTutorialActive ? handleTutorialExplain : handleLiveExplain}
                    inferenceStatus={inferenceState.status}
                    isConfigReady={hasActiveConfiguration}
                    activeAttributorId={activeAttributorId}
                    activeWrapperName={inputWrapperName}
                    tutorialFocusTarget={currentTutorialStep?.focusTarget}
                  />
                </div>
              ) : null}
            </div>

            {/* OUTPUT */}
            <div className={`bg-surface p-4 min-h-60 relative min-w-0 sm:p-6${getTutorialTargetClass("output")}`}>
              <div className="font-mono font-medium uppercase">Output</div>
              {inferenceState.status === 'success' && resultMetadata && (
                <div className="mt-3 grid gap-2 text-xs font-mono sm:grid-cols-2">
                  <div className="bg-fill px-3 py-2 text-fg-subtle min-w-0">
                    <span className="font-bold text-fg-muted uppercase mr-2">// Model:</span>
                    <span className="normal-case text-fg break-all">{resultMetadata.modelName}</span>
                  </div>
                  <div className="bg-fill px-3 py-2 text-fg-subtle min-w-0">
                    <span className="font-bold text-fg-muted uppercase mr-2">// Attributor:</span>
                    <span className="normal-case text-fg break-words">{resultMetadata.attributorName}</span>
                  </div>
                </div>
              )}
              {effectiveSystemState.status === 'running' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center">
                  <i className='bx bx-loader animate-spin text-2xl text-fg-faint'></i>
                </div>
              ) : effectiveSystemState.status === 'error' ? (
                <div className="absolute top-0 left-0 w-full h-full flex justify-center items-center font-mono font-bold text-fg-faint text-sm">
                  FAILED
                </div>
              ) : effectiveSystemState.status === 'success' && effectiveSystemState.data ? (
                <div className="min-w-0">
                  <OutputPanel
                    outputResult={inferenceState.data}
                    tutorialInteraction={currentTutorialStep?.outputInteraction}
                    tutorialFocusTarget={currentTutorialStep?.focusTarget}
                  />
                </div>
              ) : null}
            </div>

          </div>

        </main>


      </div>

      {activeTutorial && currentTutorialStep && (
        <TutorialOverlay
          tutorialKind={activeTutorial}
          step={currentTutorialStep}
          stepIndex={Math.min(tutorialStepIndex, tutorialSteps.length - 1)}
          stepCount={tutorialSteps.length}
          onBack={goToPreviousTutorialStep}
          onNext={goToNextTutorialStep}
          onClose={closeTutorial}
        />
      )}
    </div>
  );
}

import { useState, useEffect, useRef } from "react";

import { buildApiUrl } from "../lib/api";
import { getTutorialExampleMeta, isTutorialExampleJobId, loadTutorialExamplePayload, tutorialExampleSummaries } from "../lib/tutorialExamples";
import type { JobHistoryItem } from "../types";

export type { JobHistoryItem } from "../types";

const getJobTimestamp = (job: JobHistoryItem) => {
  const timestamp = Date.parse(job.created_at);
  return Number.isNaN(timestamp) ? 0 : timestamp;
};

// Pinned jobs always sort above unpinned ones; within each group, newest first.
const sortJobs = (jobs: JobHistoryItem[]) =>
  [...jobs].sort((a, b) => {
    if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1;
    return getJobTimestamp(b) - getJobTimestamp(a);
  });

const mergeJobsWithTutorialExamples = (jobs: JobHistoryItem[]) => {
  const seenIds = new Set(jobs.map((job) => job.id));
  const builtInExamples = tutorialExampleSummaries.filter((job) => !seenIds.has(job.id));

  return sortJobs([...jobs, ...builtInExamples]);
};

// A single failed poll is expected noise (backend restart, sleep/wake, page unload);
// only a backend that stays unreachable this many polls in a row is worth reporting.
const FAILED_POLLS_BEFORE_WARNING = 3;

export function useJobsHistory() {
  const [jobs, setJobs] = useState<JobHistoryItem[]>(() => mergeJobsWithTutorialExamples([]));
  const [deletingJobIds, setDeletingJobIds] = useState<string[]>([]);
  const consecutivePollFailures = useRef(0);

  const fetchJobs = async () => {
    try {
      const res = await fetch(buildApiUrl("/api/jobs"));
      consecutivePollFailures.current = 0;
      if (res.ok) {
        const data = await res.json();
        setJobs(mergeJobsWithTutorialExamples(Array.isArray(data) ? data : []));
      }
    } catch (e) {
      consecutivePollFailures.current += 1;
      if (consecutivePollFailures.current === FAILED_POLLS_BEFORE_WARNING) {
        console.warn("Jobs history backend unreachable, keeping last known jobs", e);
      }
      setJobs((currentJobs) => mergeJobsWithTutorialExamples(currentJobs.filter((job) => !job.is_builtin_example)));
    }
  };

  useEffect(() => {
    fetchJobs();
    // Polling; paused while the tab is hidden, with a refresh when it becomes visible again.
    const interval = setInterval(() => {
      if (!document.hidden) fetchJobs();
    }, 5000);
    const handleVisibilityChange = () => {
      if (!document.hidden) fetchJobs();
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  // Download job payload for a specific job ID
  const fetchJobPayload = async (jobId: string) => {
    const tutorialExampleMeta = getTutorialExampleMeta(jobId);
    if (tutorialExampleMeta) {
      const { config, payloadUrl, ...jobSummary } = tutorialExampleMeta;
      const payload = await loadTutorialExamplePayload(tutorialExampleMeta);
      return { ...jobSummary, payload };
    }

    try {
      const res = await fetch(buildApiUrl(`/api/jobs/${jobId}`));
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error("Failed to fetch job details", e);
    }
    return null;
  };

  const deleteJob = async (jobId: string) => {
    if (isTutorialExampleJobId(jobId)) {
      throw new Error("Built-in tutorial examples are always available and cannot be deleted.");
    }

    setDeletingJobIds(prev => prev.includes(jobId) ? prev : [...prev, jobId]);
    setJobs(prev => prev.filter(job => job.id !== jobId));

    try {
      const res = await fetch(buildApiUrl(`/api/jobs/${jobId}`), {
        method: "DELETE"
      });

      if (!res.ok && res.status !== 404) {
        let errorDetail = "Failed to delete job.";
        try {
          const errorPayload = await res.json();
          errorDetail = errorPayload.detail || errorPayload.message || errorDetail;
        } catch {}
        throw new Error(errorDetail);
      }

      setJobs(prev => mergeJobsWithTutorialExamples(prev.filter(job => job.id !== jobId)));
    } catch (e) {
      await fetchJobs();
      throw e;
    } finally {
      setDeletingJobIds(prev => prev.filter(id => id !== jobId));
    }
  };

  const togglePinJob = async (jobId: string) => {
    if (isTutorialExampleJobId(jobId)) return; // built-in examples are always pinned

    const targetJob = jobs.find((job) => job.id === jobId);
    if (!targetJob) return;
    const nextPinned = !targetJob.pinned;

    setJobs(prev => sortJobs(prev.map(job => job.id === jobId ? { ...job, pinned: nextPinned } : job)));

    try {
      const res = await fetch(buildApiUrl(`/api/jobs/${jobId}/pin`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned: nextPinned }),
      });
      if (!res.ok) throw new Error("Failed to update pin state.");
    } catch (e) {
      console.error("Failed to update pin state", e);
      setJobs(prev => sortJobs(prev.map(job => job.id === jobId ? { ...job, pinned: !nextPinned } : job)));
    }
  };

  return { jobs, deletingJobIds, fetchJobs, fetchJobPayload, deleteJob, togglePinJob };
}

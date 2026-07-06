import { useState, useEffect } from "react";

import { buildApiUrl } from "../lib/api";
import { getTutorialExampleJob, isTutorialExampleJobId, tutorialExampleSummaries } from "../lib/tutorialExamples";
import type { JobHistoryItem } from "../types";

export type { JobHistoryItem } from "../types";

const getJobTimestamp = (job: JobHistoryItem) => {
  const timestamp = Date.parse(job.created_at);
  return Number.isNaN(timestamp) ? 0 : timestamp;
};

const mergeJobsWithTutorialExamples = (jobs: JobHistoryItem[]) => {
  const seenIds = new Set(jobs.map((job) => job.id));
  const builtInExamples = tutorialExampleSummaries.filter((job) => !seenIds.has(job.id));

  return [...jobs, ...builtInExamples].sort((a, b) => getJobTimestamp(b) - getJobTimestamp(a));
};

export function useJobsHistory() {
  const [jobs, setJobs] = useState<JobHistoryItem[]>(() => mergeJobsWithTutorialExamples([]));
  const [deletingJobIds, setDeletingJobIds] = useState<string[]>([]);

  const fetchJobs = async () => {
    try {
      const res = await fetch(buildApiUrl("/api/jobs"));
      if (res.ok) {
        const data = await res.json();
        setJobs(mergeJobsWithTutorialExamples(Array.isArray(data) ? data : []));
      }
    } catch (e) {
      console.error("Failed to fetch jobs history", e);
      setJobs((currentJobs) => mergeJobsWithTutorialExamples(currentJobs.filter((job) => !job.is_builtin_example)));
    }
  };

  useEffect(() => {
    fetchJobs();
    // Polling
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  // Download job payload for a specific job ID
  const fetchJobPayload = async (jobId: string) => {
    const tutorialExample = getTutorialExampleJob(jobId);
    if (tutorialExample) {
      const { payload, ...jobSummary } = tutorialExample;
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

  return { jobs, deletingJobIds, fetchJobs, fetchJobPayload, deleteJob };
}

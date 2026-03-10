import { useState, useEffect } from "react";

export interface JobHistoryItem {
  id: string;
  status: 'running' | 'completed' | 'failed';
  prompt: string;
  source_name: string;
  model_name: string;
  attributor_name: string;
  created_at: string;
  execution_time_sec: number | null;
}

export function useJobsHistory() {
  const isAppLocal = true;
  const ipAddress = isAppLocal ? "localhost" : "192.168.1.23";
  const [jobs, setJobs] = useState<JobHistoryItem[]>([]);

  const fetchJobs = async () => {
    try {
      const res = await fetch(`http://${ipAddress}:8000/api/jobs`);
      if (res.ok) {
        const data = await res.json();
        setJobs(data);
      }
    } catch (e) {
      console.error("Failed to fetch jobs history", e);
    }
  };

  useEffect(() => {
    fetchJobs();
    // Polling
    const interval = setInterval(fetchJobs, 2000);
    return () => clearInterval(interval);
  }, []);

  // Download job payload for a specific job ID
  const fetchJobPayload = async (jobId: string) => {
    try {
      const res = await fetch(`http://${ipAddress}:8000/api/jobs/${jobId}`);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error("Failed to fetch job details", e);
    }
    return null;
  };

  return { jobs, fetchJobs, fetchJobPayload };
}
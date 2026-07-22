import { useEffect, useState } from "react";
import { apiFetch } from "../lib/api";

const STATUS_POLL_INTERVAL_MS = 10000;

export interface BackendUnloadInfo {
  reason: "manual" | "idle_timeout" | "model_switch";
  model_name: string | null;
  at: number;
}

export interface BackendStatus {
  model_loaded: boolean;
  model_name: string | null;
  source: string | null;
  wrapper: string | null;
  device: string | null;
  docker_gpu_mode: boolean;
  nvidia_visible_devices: string | null;
  cuda_available: boolean;
  cuda_device_count: number;
  attributor_id: string | null;
  busy: boolean;
  idle_seconds: number;
  idle_timeout_sec: number | null;
  seconds_until_unload: number | null;
  last_unload: BackendUnloadInfo | null;
  /** Token for the live configuration; differs from ours once someone else loads. */
  config_id: string | null;
  /** Whether this tab is the session that loaded what's live. */
  owned_by_you: boolean;
  /** Whether another session's claim currently blocks us from loading. */
  held_by_other_session: boolean;
  lease_seconds_remaining: number | null;
  /**
   * Client clock reading from when this poll was *sent*, not when it arrived. A response
   * can only describe state at or after its request, so callers use this to tell a genuine
   * "nothing is loaded" from an in-flight poll that predates their own load.
   */
  requestedAt: number;
}

/**
 * Polls /api/status so the UI reflects what the backend actually holds in memory.
 * The backend releases an idle model on its own, which no user action tells us about.
 *
 * Returns null until the first successful poll, and keeps the last known status if a
 * poll fails: a blip in connectivity is not evidence that the model went away.
 */
export function useBackendStatus(enabled = true) {
  const [status, setStatus] = useState<BackendStatus | null>(null);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    const fetchStatus = async () => {
      const requestedAt = Date.now();

      try {
        const res = await apiFetch("/api/status");
        if (!res.ok) return;

        const data = await res.json();
        if (!cancelled) setStatus({ ...data, requestedAt });
      } catch {
        // Keep the previous status; the next tick retries.
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, STATUS_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [enabled]);

  // While disabled, report "unknown" rather than the last status polled before that.
  return enabled ? status : null;
}

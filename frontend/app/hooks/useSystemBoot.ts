import { useState, useEffect } from "react";
import { buildApiUrl } from "../lib/api";
import { isExamplesOnlyMode } from "../lib/examplesOnlyMode";
import { BootLog, Manifest, AsyncState } from "../types";

const offlineManifest: Manifest = {
  sources: [],
  attributors: [],
};

const offlineBootLogs = [
  "Examples-only mode active.",
  "Using bundled prepared example data. Live inference calls are skipped.",
];

export function useSystemBoot(enabled = true) {
  const effectiveEnabled = enabled && !isExamplesOnlyMode;
  const [bootLogs, setBootLogs] = useState<string[]>(() => effectiveEnabled ? [] : offlineBootLogs);
  const [systemState, setSystemState] = useState<AsyncState<Manifest>>(() => (
    effectiveEnabled
      ? {
          status: 'running', // Running by default since we start booting immediately
          data: null,
          error: null
        }
      : {
          status: 'success',
          data: offlineManifest,
          error: null
        }
  ));

  useEffect(() => {
    if (!effectiveEnabled) {
      setBootLogs(offlineBootLogs);
      setSystemState({ status: 'success', data: offlineManifest, error: null });
      return;
    }

    const initSystem = async () => {
      try {
        setBootLogs([]);
        setSystemState({ status: 'running', data: null, error: null });
        setBootLogs(prev => [...prev, BootLog.SYSTEM_BOOTING, BootLog.CHECK_SERVER_CONNECTION]);

        const server = await fetch(buildApiUrl("/"));
        if (!server.ok) throw new Error(BootLog.ERROR_SERVER_CONNECTION);

        setBootLogs(prev => [...prev, BootLog.LOADING_MANIFEST]);
        const res = await fetch(buildApiUrl("/api/manifest"));
        if (!res.ok) throw new Error(BootLog.ERROR_LOADING_MANIFEST);
        const data = await res.json();

        setBootLogs(prev => [...prev, BootLog.MANIFEST_LOADED]);
        setSystemState({ status: 'success', data: data, error: null });

      } catch (err: unknown) {
        const errorMessage = err instanceof Error && err.message ? err.message : "Boot failed";
        setSystemState({ status: 'error', data: null, error: errorMessage });
      }
    };

    initSystem();
  }, [effectiveEnabled]);

  return { systemState, bootLogs };
}

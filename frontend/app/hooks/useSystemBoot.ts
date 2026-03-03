import { useState, useEffect } from "react";
import { BootLog, Manifest, AsyncState } from "../types";

export function useSystemBoot() {
  const isAppLocal = false
  const ipAddress = isAppLocal ? "localhost" : "192.168.1.23";
  const [bootLogs, setBootLogs] = useState<string[]>([]);
  const [systemState, setSystemState] = useState<AsyncState<Manifest>>({
    status: 'running', // Running by default since we start booting immediately
    data: null,
    error: null
  });

  useEffect(() => {
    const initSystem = async () => {
      try {
        setBootLogs(prev => [...prev, BootLog.SYSTEM_BOOTING, BootLog.CHECK_SERVER_CONNECTION]);

        const server = await fetch(`http://${ipAddress}:8000`);
        if (!server.ok) throw new Error(BootLog.ERROR_SERVER_CONNECTION);

        setBootLogs(prev => [...prev, BootLog.LOADING_MANIFEST]);
        const res = await fetch(`http://${ipAddress}:8000/api/manifest`);
        if (!res.ok) throw new Error(BootLog.ERROR_LOADING_MANIFEST);
        const data = await res.json();

        setBootLogs(prev => [...prev, BootLog.MANIFEST_LOADED]);
        setSystemState({ status: 'success', data: data, error: null });

      } catch (err: any) {
        setSystemState({ status: 'error', data: null, error: err.message || "Boot failed" });
      }
    };

    initSystem();
  }, []);

  return { systemState, bootLogs };
}
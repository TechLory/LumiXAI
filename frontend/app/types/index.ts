
export enum BootLog {
  SYSTEM_BOOTING = "System booting...",
  CHECK_SERVER_CONNECTION = "Checking connection to server...",
  ERROR_SERVER_CONNECTION = "Error connecting to server. Please ensure the backend is running.",
  LOADING_MANIFEST = "Loading manifest...",
  MANIFEST_LOADED = "Manifest loaded.",
  ERROR_LOADING_MANIFEST = "Error loading manifest."
}

export type Manifest = {
  sources: { id: string; name: string; type: string }[];
  attributors: { id: string; name: string }[];
};

export type ProcessStatus = 'idle' | 'running' | 'success' | 'error';

export interface AsyncState<T = any> {
  status: ProcessStatus;
  data: T | null;
  error: string | null;
}
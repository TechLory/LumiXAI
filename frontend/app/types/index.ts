
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
  attributors: { id: string; name: string; compatible_wrappers: string[] }[];
};

export type ProcessStatus = 'idle' | 'running' | 'success' | 'error';

export interface AsyncState<T = any> {
  status: ProcessStatus;
  data: T | null;
  error: string | null;
}

export interface ResultMetadata {
  modelName: string;
  attributorName: string;
}

export type TutorialKind = 'text-classification' | 'text-generation' | 'txt2img-generation';

export interface JobHistoryItem {
  id: string;
  status: "running" | "completed" | "failed";
  prompt: string;
  source_name: string;
  model_name: string;
  attributor_name: string;
  created_at: string;
  execution_time_sec: number | null;
  is_builtin_example?: boolean;
  tutorial_kind?: TutorialKind;
  pinned?: boolean;
}

export type TutorialOutputInteraction = {
  classificationTokenIndex?: number;
  textGenerationSelection?: {
    selectedType: "input" | "output";
    selectedIndex: number;
  };
  imageSelection?: {
    selectedTokenIndices?: number[];
    hoveredCell?: { x: number; y: number };
  };
};

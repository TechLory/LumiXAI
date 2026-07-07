/**
 * Mirrors the task-detection `match` in `backend/main.py`'s `/api/load` handler.
 *
 * This is a deliberate, small duplication of the backend's mapping: it exists only to
 * give an early, pre-load guess at which wrapper a model will use (from the `task` field
 * already returned by `/api/search`), so the attributor dropdown can start filtering
 * before the user clicks "Load Configuration". The backend's `/api/load` response and
 * `/api/set_attributor` validation remain the authoritative source of truth — this is UX
 * only, not a compatibility guarantee.
 */
const TASK_TO_WRAPPER: Record<string, string> = {
  "text-classification": "hf_text_classification",
  "fill-mask": "hf_text_classification",
  "token-classification": "hf_text_classification",
  "text-generation": "hf_text_generation",
  "text2text-generation": "hf_text_generation",
  "translation": "hf_text_generation",
  "summarization": "hf_text_generation",
  "text-to-image": "hf_image",
  "image-classification": "hf_image_classification",
};

export function guessWrapperFromTask(task?: string | null): string | null {
  if (!task) return null;
  return TASK_TO_WRAPPER[task] ?? null;
}

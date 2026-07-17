const SESSION_STORAGE_KEY = "lumixai_session_id";

/**
 * Identifies this browser tab to the backend, which holds a single model shared by every
 * client. The id lets the backend tell "the session that loaded this model" from everyone
 * else, so one user's load cannot silently replace another's configuration.
 *
 * Scoped to sessionStorage rather than localStorage: two tabs are two independent
 * configurations, which is how the app is actually demoed side by side. Returns "" during
 * SSR, where there is no tab to identify yet.
 */
export function getSessionId(): string {
  if (typeof window === "undefined") return "";

  let sessionId = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }

  return sessionId;
}

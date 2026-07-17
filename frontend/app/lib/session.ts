const SESSION_STORAGE_KEY = "lumixai_session_id";

// crypto.randomUUID exists only in secure contexts (https:// or localhost), and the app
// is served over plain http:// in production. Uniqueness is all the id needs — it's a
// session handle, not a credential — so fall back to getRandomValues (available in
// insecure contexts) and, failing even that, Math.random.
function generateSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }

  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

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
    sessionId = generateSessionId();
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }

  return sessionId;
}

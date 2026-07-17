import { useEffect, useState } from "react";
import { getSessionId } from "./session";

const DEFAULT_API_PORT = "8000";
const DEFAULT_DOCS_PORT = "8001";
export const SESSION_HEADER = "X-LumiXAI-Session";

function normalizeBrowserHostname(hostname: string) {
  if (!hostname || hostname === "0.0.0.0" || hostname === "::" || hostname === "[::]") {
    return "localhost";
  }

  return hostname;
}

export function getApiBaseUrl() {
  const configuredBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }

  if (typeof window === "undefined") {
    return `http://localhost:${DEFAULT_API_PORT}`;
  }

  const browserHostname = normalizeBrowserHostname(window.location.hostname);
  return `http://${browserHostname}:${DEFAULT_API_PORT}`;
}

export function buildApiUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

/**
 * fetch() against the backend with this tab's session id attached. Use it for anything
 * that loads, changes or inspects the active configuration: the backend needs to know who
 * is asking to decide whether the model is theirs to replace.
 */
export function apiFetch(path: string, init: RequestInit = {}) {
  return fetch(buildApiUrl(path), {
    ...init,
    headers: {
      ...init.headers,
      [SESSION_HEADER]: getSessionId(),
    },
  });
}

// The docs are exposed at /docs on the app's own origin by the production reverse
// proxy (e.g. http://lumixai.islab.di.unimi.it/docs). In local development the app
// runs on a dev port (3000/3001) with no proxy and the docs are a sibling service on
// :8001 of the same host, so fall back to that when a non-default port is in use.
// Override either case with NEXT_PUBLIC_DOCS_BASE_URL (e.g. http://<docs-host>:8001).
//
// `useBrowserHost` defaults to false so the first client render (hydration) matches
// the server-rendered "/docs" value exactly; callers pass true only after mount to
// resolve the dev-server fallback, avoiding a hydration mismatch on the anchor's href.
export function getDocsBaseUrl(useBrowserHost = false) {
  const configuredBaseUrl = process.env.NEXT_PUBLIC_DOCS_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }

  // SSR / first paint: same-origin relative path (stable across hydration).
  if (!useBrowserHost || typeof window === "undefined") {
    return "/docs";
  }

  // Local dev: app served on a port (3000/3001) → docs are a sibling on :8001.
  const { origin, hostname, port } = window.location;
  if (port && port !== "80" && port !== "443") {
    return `http://${normalizeBrowserHostname(hostname)}:${DEFAULT_DOCS_PORT}`;
  }

  // Reverse-proxied deployment (default port): docs live at /docs on this origin.
  return `${origin}/docs`;
}

// Renders same-origin "/docs" on first paint (matching SSR) then, post-mount, swaps
// in the :8001 sibling when running on a dev port — so the href never mismatches
// during hydration.
export function useDocsUrl() {
  const [docsUrl, setDocsUrl] = useState(() => getDocsBaseUrl(false));

  useEffect(() => {
    setDocsUrl(getDocsBaseUrl(true));
  }, []);

  return docsUrl;
}

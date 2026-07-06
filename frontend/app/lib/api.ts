import { useEffect, useState } from "react";

const DEFAULT_API_PORT = "8000";
const DEFAULT_DOCS_PORT = "8001";

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

// `useBrowserHost` defaults to false so the initial client render (hydration) matches
// the server-rendered "localhost" fallback exactly; callers pass true only after mount
// to swap in the real hostname, avoiding a hydration mismatch on the anchor's href.
export function getDocsBaseUrl(useBrowserHost = false) {
  const configuredBaseUrl = process.env.NEXT_PUBLIC_DOCS_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }

  if (!useBrowserHost || typeof window === "undefined") {
    return `http://localhost:${DEFAULT_DOCS_PORT}`;
  }

  const browserHostname = normalizeBrowserHostname(window.location.hostname);
  return `http://${browserHostname}:${DEFAULT_DOCS_PORT}`;
}

// Renders the "localhost" fallback on first paint (matching SSR) then swaps in the
// real browser hostname post-mount, so the link's href never mismatches during hydration.
export function useDocsUrl() {
  const [docsUrl, setDocsUrl] = useState(() => getDocsBaseUrl(false));

  useEffect(() => {
    setDocsUrl(getDocsBaseUrl(true));
  }, []);

  return docsUrl;
}

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

export function getDocsBaseUrl() {
  const configuredBaseUrl = process.env.NEXT_PUBLIC_DOCS_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/+$/, "");
  }

  if (typeof window === "undefined") {
    return `http://localhost:${DEFAULT_DOCS_PORT}`;
  }

  const browserHostname = normalizeBrowserHostname(window.location.hostname);
  return `http://${browserHostname}:${DEFAULT_DOCS_PORT}`;
}

const CLIENT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
// Server-only origin (never exposed to the browser; NEXT_PUBLIC_API_BASE may
// differ, e.g. a public URL in production).
const SERVER_API_BASE = process.env.API_BASE ?? "http://localhost:8000";

async function request(path: string, init?: RequestInit) {
  return fetch(`${CLIENT_API_BASE}${path}`, { ...init, credentials: "include" });
}

function errorMessage(data: unknown, status: number) {
  const d = data as { detail?: string; error?: string } | null;
  return d?.detail ?? d?.error ?? `Request failed (${status})`;
}

export async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await request(path, init);
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) throw new Error(errorMessage(data, res.status));
  return data as T;
}

export async function apiStream(path: string, init?: RequestInit) {
  const res = await request(path, init);
  if (!res.ok || !res.body) {
    let data: unknown = null;
    try {
      data = await res.json();
    } catch {
      /* still no body */
    }
    throw new Error(errorMessage(data, res.status));
  }
  return res.body;
}

async function serverRequest(path: string, session: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  headers.set("cookie", `session=${session}`);
  return fetch(`${SERVER_API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}

/**
 * Server-side fetch: forwards the session cookie read from the incoming request
 * (there is no browser cookie jar in Node). Returns null instead of throwing so
 * callers like `getServerUser` can treat "no session / not ok / network error"
 * uniformly.
 */
export async function serverApi<T = unknown>(
  path: string,
  session: string | undefined,
  init?: RequestInit,
): Promise<T | null> {
  if (!session) return null;
  try {
    const res = await serverRequest(path, session, init);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

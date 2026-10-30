// Thin client for the backend API. Every failure becomes an ApiError with a user-facing
// message and optional technical detail.

export class ApiError extends Error {
  constructor(message, { code = "error", detail = null, status = 0 } = {}) {
    super(message);
    this.code = code;
    this.detail = detail;
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    throw new ApiError("Can't reach the DocTutor server. Check that it is running.", { detail: String(err) });
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON body (e.g. proxy error page) handled below.
  }
  if (!response.ok) {
    const error = body?.error;
    throw new ApiError(error?.message || `Request failed (HTTP ${response.status}).`, {
      code: error?.code,
      detail: error?.detail,
      status: response.status,
    });
  }
  return body;
}

const json = (method, payload) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: payload === undefined ? undefined : JSON.stringify(payload),
});

export const api = {
  health: () => request("/api/health"),
  uploadDocument(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/documents", { method: "POST", body: form });
  },
  getDocument: (id) => request(`/api/documents/${encodeURIComponent(id)}`),
  createSession: (payload) => request("/api/sessions", json("POST", payload)),
  getSession: (id) => request(`/api/sessions/${encodeURIComponent(id)}`),
  nextQuestion: (id) => request(`/api/sessions/${encodeURIComponent(id)}/next`, json("POST")),
  submitAnswer: (id, payload) => request(`/api/sessions/${encodeURIComponent(id)}/answer`, json("POST", payload)),
  finishSession: (id) => request(`/api/sessions/${encodeURIComponent(id)}/finish`, json("POST")),
  summary: (id) => request(`/api/sessions/${encodeURIComponent(id)}/summary`),
};

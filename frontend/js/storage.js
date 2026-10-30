// Remembers the active session so a page reload resumes it. Storage may be unavailable
// (private mode, blocked cookies); every access is guarded.

const KEY = "doctutor.activeSession";

export function rememberSession(sessionId) {
  try {
    localStorage.setItem(KEY, sessionId);
  } catch {
    /* storage unavailable: resuming is a convenience only */
  }
}

export function recalledSession() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function forgetSession() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

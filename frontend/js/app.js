// Application controller: owns the flow upload -> processing -> setup -> question ->
// feedback -> ... -> summary. Views are pure renderers; all API calls happen here.

import { api } from "./api.js";
import { mount } from "./dom.js";
import { forgetSession, recalledSession, rememberSession } from "./storage.js";
import { button, errorBlock, loading, sessionBar } from "./views/common.js";
import { renderFeedback } from "./views/feedback.js";
import { renderProcessing } from "./views/processing.js";
import { renderQuestion } from "./views/question.js";
import { renderSetup } from "./views/setup.js";
import { renderSummary } from "./views/summary.js";
import { renderUpload } from "./views/upload.js";

const POLL_INTERVAL_MS = 800;
const ALL_TYPES = ["multiple_choice", "true_false", "fill_in_blank", "short_answer", "conceptual", "calculation"];

const root = document.getElementById("app");
const bar = document.getElementById("session-bar");
const noticeBox = document.getElementById("notice");

const state = {
  availableTypes: ALL_TYPES,
  maxUploadMb: undefined,
  document: null,
  reused: false,
  sessionId: null,
};

function setNotice(text) {
  noticeBox.hidden = !text;
  noticeBox.textContent = text || "";
}

function showBar(progress) {
  bar.hidden = !progress;
  if (progress) mount(bar, sessionBar(progress));
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ----- upload & processing -----------------------------------------------------------------

function showUpload(error = null) {
  state.document = null;
  state.sessionId = null;
  forgetSession();
  showBar(null);
  renderUpload(root, { error, maxUploadMb: state.maxUploadMb, onFile: uploadFile });
}

async function uploadFile(file) {
  mount(root, loading(`Uploading ${file.name}…`));
  try {
    const { document: doc, reused } = await api.uploadDocument(file);
    state.reused = reused;
    await waitForProcessing(doc);
  } catch (error) {
    showUpload(error);
  }
}

async function waitForProcessing(doc) {
  while (doc.status === "processing") {
    renderProcessing(root, { document: doc, onRestart: () => showUpload() });
    await sleep(POLL_INTERVAL_MS);
    doc = await api.getDocument(doc.document_id);
  }
  state.document = doc;
  if (doc.status === "failed") {
    renderProcessing(root, { document: doc, onRestart: () => showUpload() });
    return;
  }
  showSetup();
}

// ----- setup -------------------------------------------------------------------------------

function showSetup(error = null, busy = false) {
  showBar(null);
  renderSetup(root, {
    document: state.document,
    availableTypes: state.availableTypes,
    reused: state.reused,
    error,
    busy,
    onStart: startSession,
    onRestart: () => showUpload(),
  });
}

async function startSession(options) {
  showSetup(null, true);
  try {
    const session = await api.createSession({ document_id: state.document.document_id, ...options });
    state.sessionId = session.session_id;
    rememberSession(session.session_id);
    await loadQuestion();
  } catch (error) {
    showSetup(error);
  }
}

// ----- question loop -----------------------------------------------------------------------

/** Loading message with a running seconds counter, so a slow model doesn't look frozen. */
function timedLoading(message) {
  const view = loading(message);
  const counter = document.createElement("span");
  counter.className = "muted";
  view.append(counter);
  mount(root, view);
  const started = Date.now();
  const timer = setInterval(() => {
    if (!view.isConnected) return clearInterval(timer);
    counter.textContent = `${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);
}

async function loadQuestion() {
  timedLoading("Retrieving a passage and writing your question, then checking it against your document…");
  let session;
  try {
    session = await api.nextQuestion(state.sessionId);
  } catch (error) {
    showBar(null);
    mount(
      root,
      errorBlock(error, [
        button("Try again", loadQuestion),
        button("End session and see summary", finishSession, { variant: "secondary" }),
      ]),
    );
    return;
  }
  showQuestion(session);
}

function showQuestion(session) {
  showBar(session.progress);
  setNotice(session.engine_notice);
  const view = renderQuestion(root, {
    question: session.current_question,
    progress: session.progress,
    onFinish: finishSession,
    onSubmit: async (answer) => {
      view.setBusy(true);
      view.setError(null);
      try {
        const result = await api.submitAnswer(state.sessionId, {
          question_id: session.current_question.question_id,
          answer,
        });
        showBar(result.progress);
        renderFeedback(root, {
          result,
          onNext: result.session_completed ? showSummary : loadQuestion,
        });
      } catch (error) {
        view.setBusy(false);
        view.setError(error);
      }
    },
  });
}

// ----- summary -----------------------------------------------------------------------------

async function finishSession() {
  mount(root, loading("Ending the session…"));
  try {
    await api.finishSession(state.sessionId);
  } catch (error) {
    mount(root, errorBlock(error, [button("Back to start", () => showUpload())]));
    return;
  }
  await showSummary();
}

async function showSummary() {
  mount(root, loading("Putting together your summary…"));
  try {
    const summary = await api.summary(state.sessionId);
    forgetSession();
    showBar(null);
    setNotice(null);
    renderSummary(root, {
      summary,
      onRestartSameDocument: async () => {
        try {
          state.document = await api.getDocument(summary.document_id);
          state.reused = true;
          showSetup();
        } catch (error) {
          showUpload(error);
        }
      },
      onNewDocument: () => showUpload(),
    });
    window.scrollTo({ top: 0 });
  } catch (error) {
    mount(root, errorBlock(error, [button("Back to start", () => showUpload())]));
  }
}

// ----- startup -----------------------------------------------------------------------------

async function resumeSession(sessionId) {
  const session = await api.getSession(sessionId);
  state.sessionId = sessionId;
  state.document = await api.getDocument(session.document_id);
  if (session.progress.status === "completed") return showSummary();
  if (session.current_question) return showQuestion(session);
  return loadQuestion();
}

async function init() {
  try {
    const health = await api.health();
    state.availableTypes = health.question_types_available;
    state.maxUploadMb = health.max_upload_mb;
    setNotice(health.notice);
  } catch (error) {
    mount(root, errorBlock(error, [button("Retry", () => window.location.reload())]));
    return;
  }
  const saved = recalledSession();
  if (saved) {
    try {
      await resumeSession(saved);
      return;
    } catch {
      forgetSession();
    }
  }
  showUpload();
}

init();

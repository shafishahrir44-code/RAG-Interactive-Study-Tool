import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import { installDom } from "./setup-dom.js";

const root = installDom();
const { renderQuestion } = await import("../js/views/question.js");
const { renderFeedback } = await import("../js/views/feedback.js");
const { renderSetup } = await import("../js/views/setup.js");
const { renderUpload } = await import("../js/views/upload.js");

const progress = { answered: 0, target: 5, score: 0, difficulty: "easy" };
const mcq = {
  question_id: "q1",
  number: 1,
  question: "Where does glycolysis occur?",
  question_type: "multiple_choice",
  type_label: "Multiple choice",
  difficulty: "easy",
  options: ["Nucleus", "Cytoplasm", "Stroma", "Golgi"],
  topic: "respiration",
  rationale: "Covering a new topic.",
  generated_by: "extractive",
};

beforeEach(() => root.replaceChildren());

test("submit stays disabled until an option is chosen, then sends that option", () => {
  const submitted = [];
  renderQuestion(root, { question: mcq, progress, onSubmit: (a) => submitted.push(a), onFinish() {} });
  const submit = root.querySelector("#submit-btn");
  assert.equal(submit.disabled, true);

  const option = root.querySelectorAll(".option input")[1];
  option.checked = true;
  option.dispatchEvent(new window.Event("change"));
  assert.equal(submit.disabled, false);

  root.querySelector("form").dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.deepEqual(submitted, ["Cytoplasm"]);
});

test("fill-in-the-blank submits on Enter and ignores blank input", () => {
  const submitted = [];
  const question = { ...mcq, question_type: "fill_in_blank", options: [], question: "Glycolysis occurs in the _____." };
  renderQuestion(root, { question, progress, onSubmit: (a) => submitted.push(a), onFinish() {} });
  const input = root.querySelector("input.answer-input");

  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.deepEqual(submitted, []);

  input.value = "  cytoplasm ";
  input.dispatchEvent(new window.Event("input"));
  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter" }));
  assert.deepEqual(submitted, ["cytoplasm"]);
});

test("busy state blocks double submission and errors keep the typed answer", () => {
  const submitted = [];
  const question = { ...mcq, question_type: "short_answer", options: [] };
  const view = renderQuestion(root, { question, progress, onSubmit: (a) => submitted.push(a), onFinish() {} });
  const textarea = root.querySelector("textarea");
  textarea.value = "an answer";
  textarea.dispatchEvent(new window.Event("input"));

  view.setBusy(true);
  assert.equal(root.querySelector("#submit-btn").disabled, true);
  root.querySelector("form").dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.deepEqual(submitted, []);

  view.setBusy(false);
  view.setError({ message: "The AI model timed out.", detail: "POST /chat: timeout" });
  assert.equal(root.querySelector("textarea").value, "an answer");
  assert.match(root.querySelector(".error-message").textContent, /timed out/);
  assert.match(root.querySelector(".error-detail pre").textContent, /timeout/);
});

test("feedback shows verdict, both answers, explanation, page and highlighted evidence", () => {
  let next = 0;
  renderFeedback(root, {
    result: {
      user_answer: "Nucleus",
      evaluation: { verdict: "incorrect", score: 0, feedback: "Not the nucleus.", correct_points: [], missing_points: [], incorrect_points: [] },
      question: {
        question: "Where does glycolysis occur?",
        question_type: "multiple_choice",
        expected_answer: "Cytoplasm",
        explanation: "Page 2 says glycolysis is in the cytoplasm.",
        evidence: "Glycolysis takes place in the cytoplasm.",
        sources: [{ page_number: 2, chunk_id: "c1", excerpt: "Intro. Glycolysis takes place in the cytoplasm. More." }],
      },
      next_step: "Next: Reinforcing respiration.",
      session_completed: false,
    },
    onNext: () => next++,
  });
  assert.equal(root.querySelector(".verdict h1").textContent, "Not quite");
  const answers = [...root.querySelectorAll(".answers dd")].map((d) => d.textContent);
  assert.deepEqual(answers, ["Nucleus", "Cytoplasm"]);
  assert.equal(root.querySelector(".clip-tab").textContent, "p. 2");
  assert.equal(root.querySelector(".clip mark").textContent, "Glycolysis takes place in the cytoplasm.");
  assert.match(root.querySelector(".next-step").textContent, /Reinforcing/);
  root.querySelector("#next-btn").click();
  assert.equal(next, 1);
});

test("user text is rendered as text, never as HTML", () => {
  renderFeedback(root, {
    result: {
      user_answer: "<img src=x onerror=alert(1)>",
      evaluation: { verdict: "correct", score: 1, feedback: "ok" },
      question: { question: "q", question_type: "short_answer", expected_answer: "a", explanation: "e", evidence: "", sources: [] },
      next_step: "",
      session_completed: true,
    },
    onNext() {},
  });
  assert.equal(root.querySelector("img"), null);
  assert.equal(root.querySelector("#next-btn").textContent, "See my summary");
});

test("setup only enables available question types and returns chosen options", () => {
  let payload = null;
  renderSetup(root, {
    document: { filename: "a.pdf", page_count: 3, pages_with_text: 3, chunk_count: 6, topics: [], warnings: [] },
    availableTypes: ["multiple_choice", "true_false", "fill_in_blank"],
    reused: false,
    onStart: (p) => (payload = p),
    onRestart() {},
  });
  assert.equal(root.querySelector("input[value=short_answer]").disabled, true);
  root.querySelector("input[value=true_false]").checked = false;
  root.querySelector("input[name=num_questions][value='5']").checked = true;
  root.querySelector("form").dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.deepEqual(payload, {
    num_questions: 5,
    start_difficulty: "easy",
    adaptive: true,
    question_types: ["multiple_choice", "fill_in_blank"],
  });
});

test("setup requires at least one question type", () => {
  let started = false;
  renderSetup(root, {
    document: { filename: "a.pdf", page_count: 1, pages_with_text: 1, chunk_count: 1, topics: [], warnings: [] },
    availableTypes: ["multiple_choice"],
    reused: false,
    onStart: () => (started = true),
    onRestart() {},
  });
  root.querySelector("input[value=multiple_choice]").checked = false;
  root.querySelector("form").dispatchEvent(new window.Event("submit", { cancelable: true }));
  assert.equal(started, false);
  assert.match(root.querySelector(".field-error").textContent, /at least one/);
});

test("upload rejects non-PDF files before contacting the server", () => {
  const chosen = [];
  renderUpload(root, { onFile: (f) => chosen.push(f) });
  const input = root.querySelector("#pdf-input");
  Object.defineProperty(input, "files", { value: [{ name: "notes.txt", type: "text/plain", size: 10 }], configurable: true });
  input.dispatchEvent(new window.Event("change"));
  assert.deepEqual(chosen, []);
  assert.match(root.querySelector(".field-error").textContent, /isn't a PDF/);
});

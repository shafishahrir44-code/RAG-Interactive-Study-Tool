import { h, mount } from "../dom.js";
import { answerMode, engineLabel } from "../format.js";
import { button, difficultyMeter, errorBlock } from "./common.js";

const MAX_ANSWER_CHARS = 4000;

/** Builds the answer control for the question type. Returns { element, getValue }. */
export function answerControl(question, onChange, onSubmitShortcut) {
  const mode = answerMode(question.question_type);
  if (mode === "choice" || mode === "boolean") {
    const options = mode === "boolean" ? ["True", "False"] : question.options;
    const letters = "ABCDE";
    const element = h(
      "div",
      { class: `options ${mode === "boolean" ? "options-boolean" : ""}`, role: "radiogroup", "aria-label": "Answer options" },
      options.map((option, i) =>
        h(
          "label",
          { class: "option" },
          h("input", { type: "radio", name: "answer", value: option, onchange: onChange }),
          mode === "choice" ? h("span", { class: "option-key", "aria-hidden": "true" }, letters[i]) : null,
          h("span", { class: "option-text" }, option),
        ),
      ),
    );
    const getValue = () => element.querySelector("input:checked")?.value ?? "";
    return { element, getValue };
  }

  const attrs = { name: "answer", maxlength: MAX_ANSWER_CHARS, oninput: onChange, "aria-label": "Your answer", autocomplete: "off" };
  const field =
    mode === "line"
      ? h("input", { ...attrs, type: "text", class: "answer-input", placeholder: "Type the missing word or phrase" })
      : h("textarea", { ...attrs, class: "answer-input", rows: 5, placeholder: "Write your answer in your own words" });
  field.addEventListener("keydown", (event) => {
    const submitKey = mode === "line" ? event.key === "Enter" : event.key === "Enter" && (event.ctrlKey || event.metaKey);
    if (submitKey) {
      event.preventDefault();
      onSubmitShortcut();
    }
  });
  const hint = mode === "line" ? null : h("p", { class: "hint" }, "Ctrl + Enter to submit");
  return { element: h("div", {}, field, hint), getValue: () => field.value.trim() };
}

/**
 * Renders the question and returns a small controller so the app can show progress or an
 * error without re-rendering (which would wipe what the learner typed).
 */
export function renderQuestion(container, { question, progress, notice = null, onSubmit, onFinish }) {
  let busy = false;
  let control;
  const submit = button("Submit answer", null, { type: "submit", disabled: true, id: "submit-btn" });
  const errorSlot = h("div", { class: "error-slot" });
  const refresh = () => {
    submit.disabled = busy || control.getValue() === "";
    submit.textContent = busy ? "Checking…" : "Submit answer";
  };
  const trySubmit = () => {
    const value = control.getValue();
    if (value && !busy) onSubmit(value);
  };
  control = answerControl(question, refresh, trySubmit);

  const form = h(
    "form",
    { class: "answer-form", onsubmit: (event) => (event.preventDefault(), trySubmit()) },
    control.element,
    errorSlot,
    h("div", { class: "actions" }, submit, h("button", { class: "link", type: "button", onclick: onFinish }, "End session now")),
  );

  mount(
    container,
    h(
      "article",
      { class: "question-card", "aria-labelledby": "question-text" },
      h(
        "div",
        { class: "question-meta" },
        h("span", { class: "q-number" }, `Question ${question.number} of ${progress.target}`),
        h("span", { class: "tag" }, question.type_label),
        difficultyMeter(question.difficulty),
      ),
      h("p", { class: "topic-line" }, "Topic: ", h("span", {}, question.topic)),
      h("h1", { id: "question-text", class: "question-text" }, question.question),
      form,
      h(
        "footer",
        { class: "question-foot" },
        question.rationale ? h("p", {}, h("strong", {}, "Why this question: "), question.rationale) : null,
        h("p", { class: "muted" }, engineLabel(question.generated_by)),
        notice ? h("p", { class: "muted" }, notice) : null,
      ),
    ),
  );
  refresh();
  container.querySelector(".answer-input, .option input")?.focus({ preventScroll: true });

  return {
    setBusy(value) {
      busy = value;
      refresh();
    },
    setError(error) {
      mount(errorSlot, error ? errorBlock(error) : null);
    },
  };
}

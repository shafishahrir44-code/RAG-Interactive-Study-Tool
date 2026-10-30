import { h, mount } from "../dom.js";
import { QUESTION_TYPES, pageList } from "../format.js";
import { button, errorBlock } from "./common.js";

const QUESTION_COUNTS = [5, 10, 15, 20];
const DIFFICULTIES = ["easy", "medium", "hard"];

/** Reads the setup form into the createSession payload (exported for tests). */
export function readSetupForm(form) {
  const data = new FormData(form);
  return {
    num_questions: Number(data.get("num_questions")),
    start_difficulty: data.get("start_difficulty"),
    adaptive: data.get("adaptive") === "on",
    question_types: data.getAll("question_types"),
  };
}

export function renderSetup(container, { document: doc, availableTypes, reused, error = null, busy = false, onStart, onRestart }) {
  const typeOptions = QUESTION_TYPES.map((type) => {
    const available = availableTypes.includes(type.value);
    return h(
      "label",
      { class: `check ${available ? "" : "is-disabled"}`, title: available ? null : "Needs an AI model (see the notice above)" },
      h("input", { type: "checkbox", name: "question_types", value: type.value, checked: available, disabled: !available }),
      h("span", {}, type.label),
    );
  });

  const typeError = h("p", { class: "field-error", role: "alert" });
  const form = h(
    "form",
    { class: "setup-form", novalidate: true },
    fieldset("How many questions?", QUESTION_COUNTS.map((n) => radio("num_questions", n, n === 10, String(n)))),
    fieldset("Start at", DIFFICULTIES.map((d) => radio("start_difficulty", d, d === "easy", d))),
    h(
      "label",
      { class: "check toggle" },
      h("input", { type: "checkbox", name: "adaptive", checked: true }),
      h("span", {}, "Adapt to my answers", h("small", {}, "Harder after strong answers, easier and more review after mistakes.")),
    ),
    fieldset("Question types", typeOptions),
    typeError,
    error ? errorBlock(error) : null,
    h("div", { class: "actions" }, button(busy ? "Starting…" : "Start studying", null, { type: "submit", disabled: busy, id: "start-btn" })),
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = readSetupForm(form);
    if (payload.question_types.length === 0) {
      typeError.textContent = "Pick at least one question type.";
      return;
    }
    onStart(payload);
  });

  mount(
    container,
    h(
      "section",
      { class: "panel" },
      h("h2", { class: "doc-name" }, doc.title || doc.filename),
      h(
        "p",
        { class: "muted" },
        `${doc.pages_with_text} of ${doc.page_count} pages have text, split into ${doc.chunk_count} searchable passages.`,
        reused ? " This PDF was already indexed, so the existing index was reused." : "",
      ),
      doc.warnings?.length ? h("ul", { class: "warnings" }, doc.warnings.map((w) => h("li", {}, w))) : null,
      h("h3", { class: "subhead" }, "Topics found"),
      h(
        "ul",
        { class: "topics" },
        doc.topics.map((t) => h("li", {}, h("span", {}, t.label), h("small", {}, pageList(t.pages)))),
      ),
    ),
    h("section", { class: "panel" }, h("h2", {}, "Set up your session"), form),
    h("p", { class: "center" }, h("button", { class: "link", type: "button", onclick: onRestart }, "Use a different PDF")),
  );
}

function fieldset(legend, children) {
  return h("fieldset", {}, h("legend", {}, legend), h("div", { class: "choices" }, children));
}

function radio(name, value, checked, label) {
  return h("label", { class: "chip" }, h("input", { type: "radio", name, value, checked }), h("span", {}, label));
}

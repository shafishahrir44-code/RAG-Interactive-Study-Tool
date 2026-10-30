import { h } from "../dom.js";
import { DIFFICULTY_LEVEL, percent } from "../format.js";

/** Readable error with collapsible technical detail and optional actions. */
export function errorBlock(error, actions = []) {
  return h(
    "div",
    { class: "error-block", role: "alert" },
    h("p", { class: "error-message" }, error.message || "Something went wrong."),
    error.detail
      ? h("details", { class: "error-detail" }, h("summary", {}, "Technical details"), h("pre", {}, String(error.detail)))
      : null,
    actions.length ? h("div", { class: "actions" }, actions) : null,
  );
}

export function button(label, onClick, { variant = "primary", type = "button", disabled = false, id } = {}) {
  return h("button", { class: `btn btn-${variant}`, type, disabled, id, onclick: onClick }, label);
}

export function difficultyMeter(difficulty) {
  const level = DIFFICULTY_LEVEL[difficulty] || 1;
  return h(
    "span",
    { class: "difficulty", title: `Difficulty: ${difficulty}`, "aria-label": `Difficulty: ${difficulty}` },
    h("span", { class: "pips", "aria-hidden": "true" }, [1, 2, 3].map((i) => h("i", { class: i <= level ? "on" : "" }))),
    h("span", { class: "difficulty-name" }, difficulty),
  );
}

/** Header bar shown during a session: progress, score and current difficulty. */
export function sessionBar(progress) {
  const done = progress.answered;
  const ratio = progress.target ? done / progress.target : 0;
  return [
    h(
      "div",
      { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": progress.target, "aria-valuenow": done, "aria-label": "Questions answered" },
      h("span", { class: "progress-fill", style: `width:${Math.min(100, ratio * 100)}%` }),
    ),
    h("span", { class: "stat" }, h("strong", {}, `${done}/${progress.target}`), " answered"),
    h("span", { class: "stat" }, h("strong", {}, done ? percent(progress.score) : "–"), " score"),
    difficultyMeter(progress.difficulty),
  ];
}

export function loading(message) {
  return h("div", { class: "loading", role: "status" }, h("span", { class: "spinner", "aria-hidden": "true" }), message);
}

import { h, mount } from "../dom.js";
import { VERDICTS, pageList, percent } from "../format.js";
import { button } from "./common.js";

function bar(label, score, detail) {
  return h(
    "li",
    { class: "bar-row" },
    h("span", { class: "bar-label" }, label),
    h("span", { class: "bar-track", "aria-hidden": "true" }, h("span", { class: "bar-fill", style: `width:${Math.round(score * 100)}%` })),
    h("span", { class: "bar-value" }, `${percent(score)}`, h("small", {}, detail)),
  );
}

function breakdown(title, rows) {
  if (!rows.length) return null;
  return h("section", { class: "panel" }, h("h2", {}, title), h("ul", { class: "bars" }, rows));
}

export function renderSummary(container, { summary, onRestartSameDocument, onNewDocument }) {
  const counts = [
    ["correct", summary.correct],
    ["partially_correct", summary.partially_correct],
    ["incorrect", summary.incorrect],
  ];
  mount(
    container,
    h(
      "section",
      { class: "panel summary-head" },
      h("h1", {}, "Session complete"),
      h(
        "p",
        { class: "score-line" },
        h("strong", { class: "score-big" }, percent(summary.score)),
        ` across ${summary.answered} question${summary.answered === 1 ? "" : "s"}`,
      ),
      h(
        "ul",
        { class: "verdict-counts" },
        counts.map(([key, n]) => h("li", { class: `tone-${VERDICTS[key].tone}` }, h("strong", {}, n), VERDICTS[key].label)),
      ),
      h(
        "p",
        { class: "muted" },
        `Difficulty path: ${summary.difficulty_path.join(", ") || "none"}. Finished at ${summary.final_difficulty}.`,
      ),
    ),
    summary.weak_topics.length || summary.pages_to_review.length
      ? h(
          "section",
          { class: "panel review-callout" },
          h("h2", {}, "What to review"),
          summary.weak_topics.length
            ? h("ul", {}, summary.weak_topics.map((t) => h("li", {}, `${t.label}: ${pageList(t.pages)}`)))
            : null,
          summary.pages_to_review.length
            ? h("p", {}, `Re-read ${pageList(summary.pages_to_review).toLowerCase()}, where your missed answers came from.`)
            : null,
        )
      : null,
    breakdown("By topic", summary.by_topic.map((t) => bar(t.label, t.score, ` ${t.attempts} q`))),
    breakdown("By question type", summary.by_type.map((t) => bar(t.label, t.score, ` ${t.attempts} q`))),
    breakdown("By difficulty", summary.by_difficulty.map((t) => bar(t.label, t.score, ` ${t.attempts} q`))),
    summary.adaptation_log.length
      ? h(
          "section",
          { class: "panel" },
          h("h2", {}, "How the session adapted"),
          h("ol", { class: "timeline" }, summary.adaptation_log.map((e) => h("li", {}, h("strong", {}, `After question ${e.after_question}: `), e.reason))),
        )
      : null,
    h(
      "section",
      { class: "panel" },
      h("h2", {}, "Question review"),
      h(
        "ol",
        { class: "review" },
        summary.review.map((r) =>
          h(
            "li",
            { class: `tone-${VERDICTS[r.verdict].tone}` },
            h("p", { class: "review-q" }, r.question),
            h("p", {}, h("span", { class: "muted" }, "You: "), r.user_answer),
            r.verdict === "correct" ? null : h("p", {}, h("span", { class: "muted" }, "Answer: "), r.expected_answer),
            h("p", { class: "muted small" }, `${VERDICTS[r.verdict].label}, ${r.question_type.toLowerCase()}, ${r.difficulty}, ${pageList(r.source_pages).toLowerCase()}`),
          ),
        ),
      ),
    ),
    h(
      "div",
      { class: "actions center" },
      button("Study this PDF again", onRestartSameDocument),
      button("Upload another PDF", onNewDocument, { variant: "secondary" }),
    ),
  );
}

import { h, mount } from "../dom.js";
import { VERDICTS, highlightSegments, percent } from "../format.js";
import { button } from "./common.js";

/** A source passage with the evidence sentence highlighted and a page tab. */
export function evidenceClip(source, evidence) {
  return h(
    "figure",
    { class: "clip" },
    h("span", { class: "clip-tab" }, `p. ${source.page_number}`),
    h(
      "blockquote",
      {},
      highlightSegments(source.excerpt, evidence).map((segment) => (segment.mark ? h("mark", {}, segment.text) : segment.text)),
    ),
  );
}

function pointList(title, points, tone) {
  if (!points?.length) return null;
  return h("div", { class: `points points-${tone}` }, h("h3", {}, title), h("ul", {}, points.map((p) => h("li", {}, p))));
}

export function renderFeedback(container, { result, onNext }) {
  const { evaluation, question } = result;
  const verdict = VERDICTS[evaluation.verdict];
  const isFreeForm = !["multiple_choice", "true_false", "fill_in_blank"].includes(question.question_type);
  const [primary, ...others] = question.sources;

  mount(
    container,
    h(
      "article",
      { class: `feedback feedback-${verdict.tone}` },
      h(
        "header",
        { class: "verdict" },
        h("h1", { tabindex: "-1", id: "verdict" }, verdict.label),
        isFreeForm ? h("span", { class: "verdict-score" }, `${percent(evaluation.score)} match`) : null,
      ),
      h("p", { class: "verdict-feedback" }, evaluation.feedback),
      h("p", { class: "question-recap" }, question.question),
      h(
        "dl",
        { class: "answers" },
        h("div", {}, h("dt", {}, "Your answer"), h("dd", {}, result.user_answer)),
        h("div", {}, h("dt", {}, "Correct answer"), h("dd", {}, question.expected_answer)),
      ),
      pointList("What you got right", evaluation.correct_points, "good"),
      pointList("What was missing", evaluation.missing_points, "partial"),
      pointList("What was incorrect", evaluation.incorrect_points, "bad"),
      h("section", { class: "explanation" }, h("h2", {}, "Explanation"), h("p", {}, question.explanation)),
      primary
        ? h(
            "section",
            { class: "sources" },
            h("h2", {}, "From your document"),
            evidenceClip(primary, question.evidence),
            others.length
              ? h(
                  "details",
                  { class: "more-sources" },
                  h("summary", {}, `Other passages used (${others.length})`),
                  others.map((s) => evidenceClip(s, question.evidence)),
                )
              : null,
          )
        : null,
      h(
        "footer",
        { class: "next" },
        h("p", { class: "next-step" }, result.next_step),
        button(result.session_completed ? "See my summary" : "Next question", onNext, { id: "next-btn" }),
      ),
    ),
  );
  container.querySelector("#verdict")?.focus({ preventScroll: false });
}

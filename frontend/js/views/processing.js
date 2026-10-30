import { h, mount } from "../dom.js";
import { STAGE_LABELS, formatBytes } from "../format.js";
import { button, errorBlock } from "./common.js";

const STAGE_ORDER = ["extracting", "chunking", "embedding", "indexing"];

export function renderProcessing(container, { document: doc, onRestart }) {
  const current = STAGE_ORDER.indexOf(doc.stage);
  const failed = doc.status === "failed";
  const steps = STAGE_ORDER.map((stage, index) => {
    let state = "todo";
    if (doc.stage === "done" || index < current) state = "done";
    else if (index === current) state = failed ? "failed" : "active";
    return h("li", { class: `step step-${state}`, "aria-current": state === "active" ? "step" : null }, STAGE_LABELS[stage]);
  });

  mount(
    container,
    h(
      "section",
      { class: "panel" },
      h("h2", { class: "doc-name" }, doc.filename),
      h("p", { class: "muted" }, doc.page_count ? `${formatBytes(doc.size_bytes)}, ${doc.page_count} pages` : formatBytes(doc.size_bytes)),
      h("ol", { class: "steps" }, steps),
      failed
        ? errorBlock({ message: doc.error_message, detail: doc.error_detail }, [button("Upload a different PDF", onRestart)])
        : h(
            "div",
            { class: "progress progress-lg", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": Math.round(doc.progress * 100), "aria-label": "Processing progress" },
            h("span", { class: "progress-fill", style: `width:${Math.round(doc.progress * 100)}%` }),
          ),
    ),
  );
}

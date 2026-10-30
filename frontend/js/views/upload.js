import { h, mount } from "../dom.js";
import { MAX_UPLOAD_MB, validatePdfFile } from "../format.js";
import { errorBlock } from "./common.js";

export function renderUpload(container, { error = null, maxUploadMb = MAX_UPLOAD_MB, onFile }) {
  const input = h("input", { type: "file", accept: "application/pdf,.pdf", id: "pdf-input", class: "visually-hidden" });
  const hint = h("p", { class: "field-error", role: "alert" });

  const choose = (file) => {
    const problem = validatePdfFile(file, maxUploadMb);
    if (problem) {
      hint.textContent = problem;
      return;
    }
    hint.textContent = "";
    onFile(file);
  };

  input.addEventListener("change", () => choose(input.files[0]));
  const zone = h(
    "label",
    { class: "dropzone", for: "pdf-input" },
    h("span", { class: "dropzone-title" }, "Drop a PDF here, or choose one"),
    h("span", { class: "dropzone-hint" }, `Text-based PDFs up to ${maxUploadMb} MB. Scanned pages without a text layer are skipped.`),
  );
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    choose(event.dataTransfer.files[0]);
  });

  mount(
    container,
    h(
      "section",
      { class: "intro" },
      h("h1", {}, "Study a PDF, one question at a time."),
      h(
        "p",
        { class: "lede" },
        "Upload your notes or a textbook chapter. Every question is written from passages retrieved from your document, and each answer comes back with the page it came from.",
      ),
    ),
    input,
    zone,
    hint,
    error ? errorBlock(error) : null,
  );
}

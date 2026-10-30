// Pure formatting helpers (no DOM), unit-tested in tests/format.test.js.

export const MAX_UPLOAD_MB = 25; // default; the server reports its configured limit

export const VERDICTS = {
  correct: { label: "Correct", tone: "good" },
  partially_correct: { label: "Partly correct", tone: "partial" },
  incorrect: { label: "Not quite", tone: "bad" },
};

export const STAGE_LABELS = {
  queued: "Waiting to start",
  extracting: "Reading pages",
  chunking: "Splitting into passages",
  embedding: "Building the search index",
  indexing: "Grouping topics",
  done: "Ready",
};

export const QUESTION_TYPES = [
  { value: "multiple_choice", label: "Multiple choice" },
  { value: "true_false", label: "True / false" },
  { value: "fill_in_blank", label: "Fill in the blank" },
  { value: "short_answer", label: "Short answer" },
  { value: "conceptual", label: "Conceptual" },
  { value: "calculation", label: "Calculation" },
];

export const DIFFICULTY_LEVEL = { easy: 1, medium: 2, hard: 3 };

export function percent(fraction) {
  return `${Math.round((Number(fraction) || 0) * 100)}%`;
}

export function pageList(pages) {
  if (!pages || pages.length === 0) return "";
  const sorted = [...new Set(pages)].sort((a, b) => a - b);
  return `${sorted.length === 1 ? "Page" : "Pages"} ${sorted.join(", ")}`;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Client-side pre-check; the server validates again. Returns an error message or null. */
export function validatePdfFile(file, maxMb = MAX_UPLOAD_MB) {
  if (!file) return "Choose a PDF file to upload.";
  const looksLikePdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name || "");
  if (!looksLikePdf) return "That file isn't a PDF. Choose a .pdf file.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxMb * 1024 * 1024) return `PDFs must be ${maxMb} MB or smaller.`;
  return null;
}

/**
 * Split `text` into segments marking where `needle` occurs (case/whitespace-insensitive),
 * so the evidence sentence can be highlighted inside its source passage.
 */
export function highlightSegments(text, needle) {
  const clean = (needle || "").replace(/^[“"]|[”"]$/g, "").trim();
  if (!clean) return [{ text, mark: false }];
  const escaped = clean.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  const match = new RegExp(escaped, "i").exec(text);
  if (!match) return [{ text, mark: false }];
  return [
    { text: text.slice(0, match.index), mark: false },
    { text: match[0], mark: true },
    { text: text.slice(match.index + match[0].length), mark: false },
  ].filter((segment) => segment.text.length > 0);
}

/** Whether a question type is answered with free text (vs. picking an option). */
export function answerMode(questionType) {
  if (questionType === "multiple_choice") return "choice";
  if (questionType === "true_false") return "boolean";
  if (questionType === "fill_in_blank") return "line";
  return "text";
}

export function engineLabel(generatedBy) {
  if (!generatedBy) return "";
  if (generatedBy === "extractive") return "Built from a document sentence";
  return `Written by ${generatedBy.replace(/^llm:/, "").replace(/^(ollama|openai_compatible):/, "")}`;
}

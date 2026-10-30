import assert from "node:assert/strict";
import { test } from "node:test";

import { answerMode, engineLabel, highlightSegments, pageList, percent, validatePdfFile } from "../js/format.js";

test("pageList de-duplicates, sorts and pluralises", () => {
  assert.equal(pageList([4, 2, 4]), "Pages 2, 4");
  assert.equal(pageList([7]), "Page 7");
  assert.equal(pageList([]), "");
});

test("percent rounds fractions", () => {
  assert.equal(percent(0.666), "67%");
  assert.equal(percent(undefined), "0%");
});

test("validatePdfFile rejects wrong types, empty and oversized files", () => {
  assert.equal(validatePdfFile({ name: "a.pdf", type: "application/pdf", size: 10 }), null);
  assert.match(validatePdfFile({ name: "a.docx", type: "application/msword", size: 10 }), /isn't a PDF/);
  assert.match(validatePdfFile({ name: "a.pdf", type: "application/pdf", size: 0 }), /empty/);
  assert.match(validatePdfFile({ name: "a.pdf", type: "", size: 99 * 1024 * 1024 }), /25 MB/);
  assert.match(validatePdfFile(null), /Choose/);
});

test("highlightSegments marks the evidence regardless of case and spacing", () => {
  const text = "Intro. Glycolysis is the  first stage. Outro.";
  const segments = highlightSegments(text, "glycolysis is the first stage.");
  assert.deepEqual(
    segments.map((s) => [s.text, s.mark]),
    [["Intro. ", false], ["Glycolysis is the  first stage.", true], [" Outro.", false]],
  );
});

test("highlightSegments leaves text unmarked when evidence is absent or regex-hostile", () => {
  assert.deepEqual(highlightSegments("plain", "missing"), [{ text: "plain", mark: false }]);
  assert.deepEqual(highlightSegments("a (b) [c]", "(b) [c"), [
    { text: "a ", mark: false },
    { text: "(b) [c", mark: true },
    { text: "]", mark: false },
  ]);
});

test("answerMode maps question types to controls", () => {
  assert.equal(answerMode("multiple_choice"), "choice");
  assert.equal(answerMode("true_false"), "boolean");
  assert.equal(answerMode("fill_in_blank"), "line");
  assert.equal(answerMode("conceptual"), "text");
});

test("engineLabel tells the learner what produced the question", () => {
  assert.equal(engineLabel("extractive"), "Built from a document sentence");
  assert.equal(engineLabel("llm:ollama:qwen2.5:7b-instruct"), "Written by qwen2.5:7b-instruct");
});

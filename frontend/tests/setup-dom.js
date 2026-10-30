// Installs a jsdom window as the global DOM so view modules can be tested in Node.
import { JSDOM } from "jsdom";

export function installDom() {
  const dom = new JSDOM("<!doctype html><html><body><main id='app'></main></body></html>");
  const { window } = dom;
  for (const name of ["window", "document", "Node", "FormData", "HTMLElement", "Event", "KeyboardEvent"]) {
    globalThis[name] = name === "window" ? window : window[name];
  }
  return window.document.getElementById("app");
}

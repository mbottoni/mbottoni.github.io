// In-browser code runner for `{.run}` code blocks.
//
// Python runs on Pyodide (loaded lazily from the jsDelivr CDN, like mermaid),
// JavaScript runs as-is. Both execute inside a Web Worker so a slow or
// infinite snippet never freezes the page; "Stop" terminates the worker.
// Packages are resolved from the snippet's imports (numpy, scipy, ...).
(function () {
  "use strict";

  var PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";

  var WORKER_SRC = [
    "var pyodide = null, PY_URL = " + JSON.stringify(PYODIDE_URL) + ";",
    "function out(kind, text) { postMessage({ type: 'out', kind: kind, text: String(text) }); }",
    "async function getPyodide() {",
    "  if (pyodide) return pyodide;",
    "  postMessage({ type: 'status', text: 'Loading Python runtime…' });",
    "  importScripts(PY_URL + 'pyodide.js');",
    "  pyodide = await loadPyodide({ indexURL: PY_URL });",
    "  pyodide.setStdout({ batched: function (t) { out('stdout', t); } });",
    "  pyodide.setStderr({ batched: function (t) { out('stderr', t); } });",
    "  return pyodide;",
    "}",
    "async function runPython(code) {",
    "  var py = await getPyodide();",
    "  postMessage({ type: 'status', text: 'Resolving packages…' });",
    "  await py.loadPackagesFromImports(code);",
    "  postMessage({ type: 'status', text: 'Running…' });",
    "  var result = await py.runPythonAsync(code);",
    "  if (result !== undefined && result !== null) out('stdout', result);",
    "}",
    "function runJs(code) {",
    "  var log = function (kind) { return function () {",
    "    out(kind, Array.prototype.map.call(arguments, function (a) {",
    "      return typeof a === 'string' ? a : (function () { try { return JSON.stringify(a); } catch (e) { return String(a); } })();",
    "    }).join(' '));",
    "  }; };",
    "  var console = { log: log('stdout'), info: log('stdout'), warn: log('stderr'), error: log('stderr') };",
    "  postMessage({ type: 'status', text: 'Running…' });",
    "  var result = (new Function('console', code))(console);",
    "  return Promise.resolve(result).then(function (r) { if (r !== undefined) out('stdout', r); });",
    "}",
    "onmessage = async function (e) {",
    "  var t0 = Date.now();",
    "  try {",
    "    if (e.data.lang === 'python') await runPython(e.data.code); else await runJs(e.data.code);",
    "    postMessage({ type: 'done', ms: Date.now() - t0 });",
    "  } catch (err) {",
    "    out('stderr', err && err.message ? err.message : err);",
    "    postMessage({ type: 'done', ms: Date.now() - t0, error: true });",
    "  }",
    "};",
  ].join("\n");

  var workerUrl = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
  // One worker per language: Pyodide is heavy, so it is shared across blocks
  // and its state persists between runs (like a notebook kernel).
  var workers = {};
  var busy = null;

  function getWorker(lang) {
    if (!workers[lang]) workers[lang] = new Worker(workerUrl);
    return workers[lang];
  }

  function killWorker(lang) {
    if (workers[lang]) { workers[lang].terminate(); delete workers[lang]; }
  }

  function setup(fig) {
    var lang = fig.getAttribute("data-lang");
    var source = fig.querySelector(".run-source");
    var pre = fig.querySelector("pre:not(.run-output)");
    var runBtn = fig.querySelector(".run-btn");
    var editBtn = fig.querySelector(".run-edit");
    var status = fig.querySelector(".run-status");
    var output = fig.querySelector(".run-output");
    if (!source || !runBtn) return;

    var editing = false;
    editBtn.addEventListener("click", function () {
      editing = !editing;
      source.hidden = !editing;
      if (pre) pre.hidden = editing;
      editBtn.textContent = editing ? "View" : "Edit";
      if (editing) {
        source.style.height = "auto";
        source.style.height = source.scrollHeight + 4 + "px";
        source.focus();
      }
    });
    source.addEventListener("keydown", function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
      if (e.key === "Tab") {
        e.preventDefault();
        var s = source.selectionStart, t = source.selectionEnd;
        source.value = source.value.slice(0, s) + "    " + source.value.slice(t);
        source.selectionStart = source.selectionEnd = s + 4;
      }
    });

    function append(kind, text) {
      var span = document.createElement("span");
      span.className = "run-" + kind;
      span.textContent = text + (text.endsWith("\n") ? "" : "\n");
      output.appendChild(span);
      output.hidden = false;
    }

    function run() {
      if (busy) {
        if (busy.fig === fig) { stop(); return; }
        status.textContent = "Another snippet is running.";
        return;
      }
      output.textContent = "";
      output.hidden = true;
      status.textContent = "Starting…";
      runBtn.textContent = "Stop";
      fig.classList.add("running");
      var worker = getWorker(lang);
      busy = { fig: fig, lang: lang };
      worker.onmessage = function (e) {
        var m = e.data;
        if (m.type === "status") status.textContent = m.text;
        else if (m.type === "out") append(m.kind, m.text);
        else if (m.type === "done") {
          status.textContent = (m.error ? "Error" : "Done") + " · " + (m.ms / 1000).toFixed(2) + "s";
          finish();
          if (output.hidden && !m.error) append("stdout", "(no output)");
        }
      };
      worker.onerror = function (e) {
        append("stderr", e.message || "Worker error");
        status.textContent = "Error";
        finish();
      };
      worker.postMessage({ lang: lang, code: source.value });
    }

    function stop() {
      killWorker(lang);
      status.textContent = "Stopped";
      finish();
    }

    function finish() {
      busy = null;
      runBtn.textContent = "Run";
      fig.classList.remove("running");
    }

    runBtn.addEventListener("click", run);
  }

  function init() {
    document.querySelectorAll("figure.code-block.runnable").forEach(setup);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

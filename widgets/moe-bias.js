// Load balancing for a top-k Mixture-of-Experts router, live.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-moe-bias
//   :::
//
// Every step a batch of tokens is routed to K of E experts. Each token's
// affinity for expert i is s_i = sigmoid(mu_i + w_i + noise), where mu_i is a
// fixed popularity skew (the "LM wants expert 1" part) and w_i is a learnable
// router offset. Three regimes:
//
//   none  - plain top-k on s. The skew goes straight into the load.
//   aux   - an auxiliary loss alpha * sum_i f_i P_i (the DeepSeek-V3
//           sequence-wise form) is differentiated w.r.t. w_i and descended,
//           against a weak pull of w back to 0 that stands in for the language
//           modelling gradient. Balance is bought by bending the gate scores.
//   bias  - DeepSeek-V3's auxiliary-loss-free rule: a per-expert bias b_i is
//           added to s_i for the top-k *selection only*; gating weights still
//           come from s. After each step b_i += gamma * sign(mean - load_i).
//
// "Gate distortion" is the mean absolute difference between the gating
// weights actually used and the ones the un-shifted affinities would give on
// the same selected experts. By construction it is exactly 0 for the bias rule.
(function () {
  "use strict";

  var E = 16;            // routed experts
  var K = 4;             // experts per token
  var B = 1024;          // tokens per step
  var HIST = 320;        // MaxVio samples kept for the trace
  var PULL = 0.02;       // stand-in for the LM gradient pulling w back to 0
  var AUX_LR = 10;       // step size on the auxiliary-loss gradient
  var SKEW = 1.4;        // mu runs linearly from +SKEW to -SKEW

  var MODES = [
    { key: "none", label: "No balancing" },
    { key: "aux", label: "Aux loss" },
    { key: "bias", label: "Bias update" },
  ];

  var MU = new Float64Array(E);
  for (var m0 = 0; m0 < E; m0++) MU[m0] = SKEW - 2 * SKEW * m0 / (E - 1);

  function randn() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function cssVar(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  // Scoped styles for the bits main.css does not already provide.
  function injectStyles() {
    if (document.getElementById("moe-bias-styles")) return;
    var s = document.createElement("style");
    s.id = "moe-bias-styles";
    var w = '.widget[data-widget="moe-bias"] ';
    s.textContent =
      w + ".moe-modes{display:flex;flex-wrap:wrap;gap:0.35rem;margin-bottom:0.9rem}" +
      w + ".moe-modes .widget-btn{flex:1 1 auto;padding:0.35rem 0.5rem}" +
      w + ".moe-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:0.4rem;margin:0.2rem 0 0.9rem}" +
      w + ".widget-stat{border:1px solid var(--border);border-radius:8px;padding:0.4rem 0.5rem;background:var(--block-bg);min-width:0}" +
      w + ".widget-stat .moe-stat-name{display:block;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.07em;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}" +
      w + ".widget-stat b{display:block;font-family:'JetBrains Mono',monospace;font-weight:400;font-size:0.86rem;color:var(--fg);margin-top:0.15rem}" +
      w + ".moe-trace{display:block;width:100%;height:76px}";
    document.head.appendChild(s);
  }

  function build(mount) {
    injectStyles();

    var mode = "none";
    var gamma = 0.003, alpha = 0.3;
    var b = new Float64Array(E);      // selection-only bias
    var w = new Float64Array(E);      // router offset moved by the aux loss
    var load = new Float64Array(E);   // last batch
    var loadEma = new Float64Array(E);
    var sArr = new Float64Array(B * E);
    var z0Arr = new Float64Array(B * E);
    var hist = [];                    // { v: MaxVio ema, mode }
    var vioEma = null, distEma = 0, step = 0;
    var running = true, raf = null;

    // --- chrome ------------------------------------------------------------
    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "MoE load balancing";
    el("div", "widget-meta", head).textContent =
      E + " experts · top-" + K + " · " + B + " tokens/step";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Per-expert load bars against the balanced ideal, with the offsets each "
      + "balancing method applies");

    var side = el("div", "widget-side", body);
    el("div", "widget-label", side).textContent = "Balancing";
    var modeRow = el("div", "moe-modes", side);
    var modeBtns = MODES.map(function (md) {
      var bt = el("button", "widget-btn", modeRow);
      bt.type = "button";
      bt.textContent = md.label;
      bt.addEventListener("click", function () { setMode(md.key); });
      return { key: md.key, btn: bt };
    });

    function slider(label, lo, hi, initial, toVal, fmt, onChange) {
      var row = el("div", "widget-slider-row", side);
      el("span", "moe-slider-name", row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range";
      input.min = String(lo); input.max = String(hi); input.step = "1";
      input.value = String(initial);
      input.setAttribute("aria-label", label);
      var val = el("span", "widget-slider-val", row);
      val.textContent = fmt(toVal(initial));
      input.addEventListener("input", function () {
        var v = toVal(Number(input.value));
        val.textContent = fmt(v);
        onChange(v);
      });
    }

    // gamma on a log scale, 1e-4 .. 3e-2; slider position 0..100.
    var gToVal = function (p) { return Math.pow(10, -4 + 2.5 * p / 100); };
    var gInit = Math.round((Math.log10(gamma) + 4) / 2.5 * 100);
    slider("γ (bias)", 0, 100, gInit, gToVal,
      function (v) { return v < 0.001 ? v.toExponential(1) : v.toFixed(4); },
      function (v) { gamma = v; });
    slider("α (aux)", 0, 100, Math.round(alpha / 3 * 100),
      function (p) { return 3 * p / 100; },
      function (v) { return v.toFixed(2); },
      function (v) { alpha = v; });

    el("div", "widget-label", side).textContent = "Readout";
    var stats = el("div", "moe-stats", side);
    function stat(name) {
      var d = el("div", "widget-stat", stats);
      // Spans need an attribute: the page unwraps bare <span>s after load.
      el("span", "moe-stat-name", d).textContent = name;
      return el("b", null, d);
    }
    var stStep = stat("step");
    var stVio = stat("MaxVio");
    var stDist = stat("gate dist.");

    el("div", "widget-label", side).textContent = "MaxVio over time";
    var trace = el("canvas", "moe-trace", side);

    var note = el("div", "widget-note", mount);
    var controls = el("div", "widget-controls", mount);
    function button(text, fn, primary) {
      var bt = el("button", "widget-btn", controls);
      bt.type = "button";
      bt.textContent = text;
      if (primary) bt.setAttribute("data-primary", "");
      bt.addEventListener("click", fn);
      return bt;
    }
    var pauseBtn = button("Pause", function () {
      running = !running;
      pauseBtn.textContent = running ? "Pause" : "Run";
      if (running) loop();
    }, true);
    button("Reset", function () { reset(); });

    // --- simulation --------------------------------------------------------
    function simStep() {
      load.fill(0);
      var useBias = mode === "bias";
      var dsum = 0;
      var idx = new Int32Array(E), key = new Float64Array(E);
      for (var t = 0; t < B; t++) {
        var o = t * E;
        for (var i = 0; i < E; i++) {
          var z0 = MU[i] + randn();
          z0Arr[o + i] = z0;
          var s = 1 / (1 + Math.exp(-(z0 + w[i])));
          sArr[o + i] = s;
          key[i] = useBias ? s + b[i] : s;
          idx[i] = i;
        }
        // partial selection sort: the top K of E by key
        for (var a = 0; a < K; a++) {
          var best = a;
          for (var c = a + 1; c < E; c++) if (key[idx[c]] > key[idx[best]]) best = c;
          var tmp = idx[a]; idx[a] = idx[best]; idx[best] = tmp;
        }
        var g = 0, g0 = 0;
        for (var k = 0; k < K; k++) {
          var e = idx[k];
          load[e]++;
          g += sArr[o + e];
          g0 += 1 / (1 + Math.exp(-z0Arr[o + e]));
        }
        for (var k2 = 0; k2 < K; k2++) {
          var e2 = idx[k2];
          dsum += Math.abs(sArr[o + e2] / g - 1 / (1 + Math.exp(-z0Arr[o + e2])) / g0);
        }
      }

      var mean = B * K / E, mx = 0;
      for (var j = 0; j < E; j++) {
        if (load[j] > mx) mx = load[j];
        loadEma[j] = step === 0 ? load[j] : 0.8 * loadEma[j] + 0.2 * load[j];
      }
      var vio = mx / mean - 1;
      vioEma = vioEma === null ? vio : 0.9 * vioEma + 0.1 * vio;
      distEma = 0.9 * distEma + 0.1 * dsum / (B * K);

      if (mode === "bias") {
        // The integral controller: accumulate the sign of the load error.
        for (var j2 = 0; j2 < E; j2++) {
          b[j2] += gamma * (load[j2] < mean ? 1 : load[j2] > mean ? -1 : 0);
        }
      }

      // Gradient of alpha * sum_i f_i P_i w.r.t. w_i, f held constant
      // (it comes from a hard top-k). P_i is the mean of s_i / sum_j s_j.
      var grad = new Float64Array(E);
      if (mode === "aux" && alpha > 0) {
        var f = new Float64Array(E);
        for (var j3 = 0; j3 < E; j3++) f[j3] = load[j3] * E / (K * B);
        for (var t2 = 0; t2 < B; t2++) {
          var o2 = t2 * E, S = 0, fs = 0;
          for (var i2 = 0; i2 < E; i2++) S += sArr[o2 + i2];
          for (var i3 = 0; i3 < E; i3++) fs += f[i3] * sArr[o2 + i3] / S;
          for (var i4 = 0; i4 < E; i4++) {
            var si = sArr[o2 + i4];
            grad[i4] += si * (1 - si) / S * (f[i4] - fs) / B;
          }
        }
      }
      for (var j4 = 0; j4 < E; j4++) {
        w[j4] -= AUX_LR * alpha * grad[j4] + PULL * w[j4];
      }

      step++;
      hist.push({ v: vioEma, mode: mode });
      if (hist.length > HIST) hist.shift();
    }

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var tctx = trace.getContext("2d");

    function palette() {
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        link: cssVar(mount, "--link", "#2156a5"),
        line: cssVar(mount, "--border", "#e2e2dd"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        block: cssVar(mount, "--block-bg", "#f3f3f2"),
      };
    }

    function drawBars() {
      var W = stage.clientWidth || 300;
      var H = Math.round(W * 0.9);
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      var p = palette();
      ctx.font = '10px "JetBrains Mono", monospace';

      var left = 4, right = W - 4;
      var topH = H * 0.66, top0 = 14, base = topH;
      var mean = B * K / E, ymax = 3;
      var slot = (right - left) / E, bw = slot * 0.68;
      var yOf = function (r) { return base - (base - top0) * Math.min(r, ymax) / ymax; };

      // ideal line
      ctx.strokeStyle = p.muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(left, yOf(1)); ctx.lineTo(right, yOf(1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = p.muted;
      ctx.textAlign = "right";
      ctx.fillText("ideal", right, yOf(1) - 4);
      ctx.textAlign = "left";
      ctx.fillText("load / ideal (max " + ymax + "×)", left, 10);

      for (var i = 0; i < E; i++) {
        var r = loadEma[i] / mean;
        var x = left + i * slot + (slot - bw) / 2;
        ctx.fillStyle = r > 1.25 ? p.accent : p.link;
        ctx.fillRect(x, yOf(r), bw, base - yOf(r));
      }
      ctx.strokeStyle = p.line;
      ctx.beginPath();
      ctx.moveTo(left, base + 0.5); ctx.lineTo(right, base + 0.5);
      ctx.stroke();

      // expert indices
      ctx.fillStyle = p.muted;
      ctx.textAlign = "center";
      for (var i2 = 0; i2 < E; i2 += 3) {
        ctx.fillText(String(i2 + 1), left + i2 * slot + slot / 2, base + 12);
      }

      // offsets panel: b_i (score units) or w_i (logit units)
      var offTop = base + 40, offBot = H - 4, mid = (offTop + offBot) / 2;
      var half = (offBot - offTop) / 2;
      var arr, scale, title;
      if (mode === "bias") { arr = b; scale = 0.5; title = "bias b_i (selection only)"; }
      else if (mode === "aux") { arr = w; scale = 2; title = "router shift w_i (bends gates)"; }
      else { arr = null; title = "no offset applied"; }
      ctx.textAlign = "left";
      ctx.fillText(title, left, offTop - 6);
      ctx.strokeStyle = p.line;
      ctx.beginPath();
      ctx.moveTo(left, mid + 0.5); ctx.lineTo(right, mid + 0.5);
      ctx.stroke();
      if (arr) {
        for (var j = 0; j < E; j++) {
          var v = Math.max(-1, Math.min(1, arr[j] / scale));
          var xx = left + j * slot + (slot - bw) / 2;
          ctx.fillStyle = v >= 0 ? p.link : p.accent;
          var hpx = Math.abs(v) * half;
          ctx.fillRect(xx, v >= 0 ? mid - hpx : mid, bw, Math.max(1, hpx));
        }
      }
    }

    function drawTrace() {
      var W = side.clientWidth || 260, H = 76;
      var dpr = window.devicePixelRatio || 1;
      trace.width = Math.round(W * dpr);
      trace.height = Math.round(H * dpr);
      tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      tctx.clearRect(0, 0, W, H);
      var p = palette();
      var vmax = 2;
      var x = function (k) { return (W - 2) * k / (HIST - 1) + 1; };
      var y = function (v) { return H - 4 - (H - 10) * Math.min(v, vmax) / vmax; };

      tctx.strokeStyle = p.line;
      tctx.lineWidth = 1;
      tctx.setLineDash([2, 3]);
      [0.5, 1, 1.5].forEach(function (g) {
        tctx.beginPath(); tctx.moveTo(0, y(g)); tctx.lineTo(W, y(g)); tctx.stroke();
      });
      tctx.setLineDash([]);
      tctx.fillStyle = p.muted;
      tctx.font = '9px "JetBrains Mono", monospace';
      tctx.fillText("1.0", 2, y(1) - 2);

      // mode-change markers
      for (var k = 1; k < hist.length; k++) {
        if (hist[k].mode !== hist[k - 1].mode) {
          tctx.strokeStyle = p.muted;
          tctx.beginPath(); tctx.moveTo(x(k), 0); tctx.lineTo(x(k), H); tctx.stroke();
        }
      }
      if (hist.length < 2) return;
      tctx.strokeStyle = p.accent;
      tctx.lineWidth = 1.75;
      tctx.lineJoin = "round";
      tctx.beginPath();
      for (var k2 = 0; k2 < hist.length; k2++) {
        if (k2 === 0) tctx.moveTo(x(k2), y(hist[k2].v));
        else tctx.lineTo(x(k2), y(hist[k2].v));
      }
      tctx.stroke();
    }

    function readout() {
      stStep.textContent = String(step);
      stVio.textContent = vioEma === null ? "–" : vioEma.toFixed(2);
      stDist.textContent = (100 * distEma).toFixed(2) + "%";
      modeBtns.forEach(function (m) {
        if (m.key === mode) m.btn.setAttribute("data-primary", "");
        else m.btn.removeAttribute("data-primary");
        m.btn.setAttribute("aria-pressed", m.key === mode ? "true" : "false");
      });
    }

    function redraw() { drawBars(); drawTrace(); readout(); }

    var NOTES = {
      none: "Plain top-" + K + ": the router's preference for the low-numbered "
        + "experts goes straight into the load. MaxVio is how far the busiest "
        + "expert sits above the balanced load (0 is perfect).",
      aux: "An auxiliary loss pushes the router's scores themselves. It does "
        + "balance, but only by bending the gating weights away from what the "
        + "model wanted (gate distortion), and it fights the main objective. "
        + "Raise α and watch both numbers move.",
      bias: "Each step, overloaded experts get b_i −= γ and underloaded ones "
        + "b_i += γ. The bias only decides who is picked, so gate distortion "
        + "stays at exactly zero. Large γ converges fast but chatters; small γ "
        + "is smooth but slow.",
    };

    function setMode(k) {
      if (k === mode) return;
      if (k === "bias") b.fill(0);
      mode = k;
      note.textContent = NOTES[k];
      readout();
      if (!running) redraw();
    }

    function reset() {
      b.fill(0); w.fill(0); loadEma.fill(0);
      hist.length = 0; vioEma = null; distEma = 0; step = 0;
      simStep();
      redraw();
    }

    function loop() {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(function tick() {
        if (!running) { raf = null; return; }
        simStep();
        redraw();
        raf = requestAnimationFrame(tick);
      });
    }

    // Pause when off-screen so the page does not burn CPU in the background.
    var visible = true;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible && running && !raf) loop();
        if (!visible && raf) { cancelAnimationFrame(raf); raf = null; }
      }).observe(mount);
    }

    note.textContent = NOTES[mode];
    reset();
    if (visible) loop();

    window.addEventListener("resize", redraw);
    new MutationObserver(redraw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="moe-bias"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

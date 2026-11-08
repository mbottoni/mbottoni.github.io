// Toy Models of Superposition (Elhage et al., 2022), trained live in the page.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-superposition
//   :::
//
// The model is the paper's ReLU output model with n = 5 features squeezed into
// m = 2 hidden dimensions:
//   h = W x,   x' = ReLU(W^T h + b),   W in R^{2x5}
// Each feature is zero with probability S (the sparsity) and uniform on [0, 1]
// otherwise; feature i has importance I_i = decay^i, and the loss is the
// importance-weighted squared error  sum_i I_i (x_i - x'_i)^2, minimised with
// Adam on fresh mini-batches. Nothing is scripted: the arrows are the columns
// of W as the optimiser moves them.
//
// Checked offline with this exact code (node, 5 seeds, decay 0.9, 2500 steps):
// S = 0 stores two features orthogonally in every seed (dims/feature 1.00);
// S = 0.6-0.75 stores four as two antipodal pairs in every seed (0.50);
// S ~ 0.84-0.9 is the boundary (seeds split between pairs and pentagon);
// S >= 0.94 stores all five as a pentagon in every seed (0.29-0.32).
(function () {
  "use strict";

  var NF = 5, M = 2;
  var STEPS = 2500;          // Adam steps per training run
  var BATCH = 256;
  var PER_FRAME = 50;        // optimiser steps per animation frame
  var LR = 0.01;
  var RMAX = 1.35;           // plot radius in weight units

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function cssVar(node, name, fallback) {
    var v = getComputedStyle(node).getPropertyValue(name).trim();
    return v || fallback;
  }

  function rgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (isNaN(n)) return hex;
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + a + ")";
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // --- the model -----------------------------------------------------------
  // W is stored row-major as W[d * NF + i]: column i is feature i's direction.
  function Trainer(S, decay, seed) {
    this.rnd = mulberry32(seed);
    this.S = S;
    this.I = new Float64Array(NF);
    for (var i = 0; i < NF; i++) this.I[i] = Math.pow(decay, i);
    this.W = new Float64Array(M * NF);
    this.b = new Float64Array(NF);
    for (var k = 0; k < M * NF; k++) this.W[k] = (this.rnd() * 2 - 1) * 0.5;
    var P = M * NF + NF;
    this.m1 = new Float64Array(P);
    this.m2 = new Float64Array(P);
    this.g = new Float64Array(P);
    this.t = 0;
    this.loss = NaN;
    this.x = new Float64Array(NF);
    this.h = new Float64Array(M);
    this.dh = new Float64Array(M);
  }

  Trainer.prototype.step = function () {
    var W = this.W, b = this.b, I = this.I, g = this.g, x = this.x, h = this.h;
    var dh = this.dh, rnd = this.rnd, S = this.S;
    var gb = M * NF;            // offset of the bias gradient in g
    g.fill(0);
    var loss = 0;
    for (var s = 0; s < BATCH; s++) {
      for (var i = 0; i < NF; i++) x[i] = rnd() < S ? 0 : rnd();
      for (var d = 0; d < M; d++) {
        var a = 0;
        for (var j = 0; j < NF; j++) a += W[d * NF + j] * x[j];
        h[d] = a;
      }
      dh[0] = 0; dh[1] = 0;
      for (var i2 = 0; i2 < NF; i2++) {
        var p = b[i2];
        for (var d2 = 0; d2 < M; d2++) p += W[d2 * NF + i2] * h[d2];
        var out = p > 0 ? p : 0;
        var e = out - x[i2];
        loss += I[i2] * e * e;
        if (p <= 0) continue;   // ReLU gate
        var gi = 2 * I[i2] * e / BATCH;
        g[gb + i2] += gi;
        for (var d3 = 0; d3 < M; d3++) {
          g[d3 * NF + i2] += gi * h[d3];        // through W^T
          dh[d3] += gi * W[d3 * NF + i2];
        }
      }
      for (var d4 = 0; d4 < M; d4++) {
        for (var j2 = 0; j2 < NF; j2++) g[d4 * NF + j2] += dh[d4] * x[j2]; // through W
      }
    }
    this.loss = loss / BATCH;

    // Adam
    this.t++;
    var b1 = 0.9, b2 = 0.999;
    var c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (var k = 0; k < g.length; k++) {
      this.m1[k] = b1 * this.m1[k] + (1 - b1) * g[k];
      this.m2[k] = b2 * this.m2[k] + (1 - b2) * g[k] * g[k];
      var up = LR * (this.m1[k] / c1) / (Math.sqrt(this.m2[k] / c2) + 1e-8);
      if (k < gb) W[k] -= up; else b[k - gb] -= up;
    }
  };

  // Per-feature norms, the paper's superposition score
  //   sum_{j != i} (W_i_hat . W_j)^2,
  // and dimensions per feature m / ||W||_F^2.
  function analyse(W) {
    var norms = [], sup = [], fro = 0;
    for (var i = 0; i < NF; i++) {
      var nx = W[i], ny = W[NF + i];
      norms.push(Math.sqrt(nx * nx + ny * ny));
      fro += nx * nx + ny * ny;
    }
    for (var a = 0; a < NF; a++) {
      var s = 0;
      for (var c = 0; c < NF; c++) {
        if (c === a || norms[a] < 1e-9) continue;
        var dp = (W[a] * W[c] + W[NF + a] * W[NF + c]) / norms[a];
        s += dp * dp;
      }
      sup.push(s);
    }
    var stored = 0;
    for (var k = 0; k < NF; k++) if (norms[k] > 0.5) stored++;
    var antipodal = 0;
    for (var p = 0; p < NF; p++) {
      for (var q = p + 1; q < NF; q++) {
        if (norms[p] < 0.5 || norms[q] < 0.5) continue;
        var cos = (W[p] * W[q] + W[NF + p] * W[NF + q]) / (norms[p] * norms[q]);
        if (cos < -0.95) antipodal++;
      }
    }
    // A feature counts as "in superposition" only if it is actually stored:
    // for a near-zero column the normalised overlap is meaningless.
    var shared = [];
    for (var f = 0; f < NF; f++) shared.push(norms[f] > 0.5 && sup[f] > 0.25);
    return {
      norms: norms, sup: sup, shared: shared, stored: stored,
      antipodal: antipodal,
      dims: fro > 1e-9 ? M / fro : 0,
    };
  }

  // Sparsity slider is log-scaled in the feature probability 1 - S, so the
  // interesting region near S = 1 gets most of the travel.
  function sparsityOf(t) { return 1 - Math.pow(10, -2 * t); }

  function build(mount) {
    var tS = 0, decay = 0.9, seed = 1;
    var tr = null, raf = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Toy model of superposition";
    el("div", "widget-meta", head).textContent =
      NF + " features → " + M + " dims · trained live";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Columns of W drawn as arrows in the two-dimensional hidden space");

    var side = el("div", "widget-side", body);

    function slider(label, min, max, step, value, fmt, onInput) {
      var row = el("div", "widget-slider-row", side);
      el("span", null, row).textContent = label;
      var s = el("input", "widget-slider", row);
      s.type = "range";
      s.min = String(min); s.max = String(max); s.step = String(step);
      s.value = String(value);
      s.setAttribute("aria-label", label);
      var val = el("span", "widget-slider-val", row);
      function sync() { val.textContent = fmt(Number(s.value)); }
      s.addEventListener("input", function () { sync(); onInput(Number(s.value)); });
      sync();
      return { input: s, sync: sync };
    }

    el("div", "widget-label", side).textContent = "Data";
    var sS = slider("sparsity S", 0, 1, 0.005, tS, function (t) {
      return sparsityOf(t).toFixed(3);
    }, function (t) { tS = t; restart(); });
    var sD = slider("importance", 0.6, 1, 0.01, decay, function (d) {
      return d.toFixed(2) + "ⁱ";
    }, function (d) { decay = d; restart(); });

    el("div", "widget-label", side).textContent = "‖Wᵢ‖ per feature";
    var bars = [];
    for (var i = 0; i < NF; i++) {
      var row = el("div", "widget-row", side);
      el("span", "widget-row-name", row).textContent = String(i + 1);
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      var val = el("span", "widget-row-val", row);
      bars.push({ fill: fill, val: val });
    }

    function stat(label) {
      var r = el("div", "widget-energy", side);
      r.style.marginTop = "0.45rem";
      el("div", "widget-label", r).textContent = label;
      return el("b", null, r);
    }
    var stStored = stat("features stored");
    var stDims = stat("dimensions / feature");
    var stLoss = stat("loss · step");

    var note = el("div", "widget-note", mount);
    var controls = el("div", "widget-controls", mount);

    function button(text, fn, primary) {
      var b = el("button", "widget-btn", controls);
      b.type = "button";
      b.textContent = text;
      if (primary) b.setAttribute("data-primary", "");
      b.addEventListener("click", fn);
      return b;
    }

    function preset(t) {
      tS = t; sS.input.value = String(t); sS.sync(); restart();
    }
    button("Retrain (new seed)", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      restart();
    }, true);
    button("Dense · S = 0", function () { preset(0); });
    button("S = 0.6", function () { preset(0.2); });
    button("S = 0.97", function () { preset(0.76); });

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function draw() {
      var w = stage.clientWidth || 300;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = w + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);

      var accent = cssVar(mount, "--accent", "#ba3925");
      var link = cssVar(mount, "--link", "#2156a5");
      var border = cssVar(mount, "--border", "#e2e2dd");
      var muted = cssVar(mount, "--muted", "#6a6a6a");
      var fg = cssVar(mount, "--fg", "#1a1a1a");

      var c = w / 2, R = (w / 2 - 14) / RMAX;

      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(8, c); ctx.lineTo(w - 8, c);
      ctx.moveTo(c, 8); ctx.lineTo(c, w - 8);
      ctx.stroke();
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.arc(c, c, R, 0, 6.2832);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = muted;
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("circle: ‖w‖ = 1", 8, 8);
      ctx.textBaseline = "alphabetic";

      if (!tr) return;
      var info = analyse(tr.W);
      var W = tr.W;
      var dropped = [];
      // least important first so the important arrows sit on top
      for (var i = NF - 1; i >= 0; i--) {
        var x = W[i], y = W[NF + i];
        var px = c + x * R, py = c - y * R;
        var col = info.shared[i] ? accent : link;
        var alpha = 0.4 + 0.6 * tr.I[i] / tr.I[0];
        ctx.strokeStyle = rgba(col, alpha);
        ctx.fillStyle = rgba(col, alpha);
        ctx.lineWidth = 2.4;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(c, c); ctx.lineTo(px, py);
        ctx.stroke();
        var len = Math.hypot(px - c, py - c);
        if (len > 6) {
          var ux = (px - c) / len, uy = (py - c) / len, hs = 8;
          ctx.beginPath();
          ctx.moveTo(px + ux * 2, py + uy * 2);
          ctx.lineTo(px - ux * hs - uy * hs * 0.5, py - uy * hs + ux * hs * 0.5);
          ctx.lineTo(px - ux * hs + uy * hs * 0.5, py - uy * hs - ux * hs * 0.5);
          ctx.closePath();
          ctx.fill();
        }
        // label just beyond the tip; dropped features are listed instead
        if (info.norms[i] < 0.5) { dropped.unshift(i + 1); continue; }
        ctx.fillStyle = fg;
        ctx.font = "600 11px 'Open Sans', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), px + (px - c) / len * 11,
          py + (py - c) / len * 11);
        ctx.textBaseline = "alphabetic";
      }
      if (dropped.length) {
        ctx.fillStyle = muted;
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText("dropped: " + dropped.join(", "), 8, w - 8);
      }
    }

    function readout() {
      if (!tr) return;
      var info = analyse(tr.W);
      var accent = cssVar(mount, "--accent", "#ba3925");
      var link = cssVar(mount, "--link", "#2156a5");
      for (var i = 0; i < NF; i++) {
        var n = info.norms[i];
        bars[i].fill.style.width = (100 * Math.min(1, n / RMAX)).toFixed(1) + "%";
        bars[i].fill.style.background = info.shared[i] ? accent : link;
        bars[i].val.textContent = n.toFixed(2);
      }
      stStored.textContent = info.stored + " / " + NF;
      stDims.textContent = info.dims.toFixed(2);
      stLoss.textContent = tr.loss.toFixed(4) + " · " + tr.t;
      return info;
    }

    function describe(info) {
      var S = sparsityOf(tS);
      if (info.stored > M && info.dims > 0.9) {
        return "Dense regime with near-equal importances: no single pair of "
          + "features is the obvious one to keep, so the model spreads two "
          + "dimensions' worth of capacity across a rotated basis "
          + "(dims/feature ≈ 1). Still no real superposition.";
      }
      if (info.dims > 0.9) {
        return "Dense regime (S = " + S.toFixed(2) + "). The model stores the "
          + info.stored + " most important features on orthogonal axes and "
          + "simply drops the rest — the same answer PCA would give. Blue means "
          + "no interference.";
      }
      if (info.stored === 4 && info.antipodal >= 2) {
        return "Antipodal superposition. Four features now share two dimensions "
          + "as two opposite pairs (dims/feature ≈ 0.5). A pair can't both be "
          + "active without interfering, but at this sparsity they rarely are, "
          + "and the ReLU plus a negative bias filters the small cross-talk.";
      }
      if (info.stored === NF && info.dims < 0.36) {
        return "Pentagon. All five features are packed into two dimensions, "
          + "72° apart (dims/feature ≈ 0.3). Red bars mark features that "
          + "overlap with others: every feature is in superposition, which is "
          + "exactly why individual neurons in real models are hard to read.";
      }
      return "An intermediate geometry (" + info.stored + " features stored). "
        + "Near a phase boundary the optimiser can settle into either solution; "
        + "try Retrain to see another seed.";
    }

    // --- training loop -----------------------------------------------------
    function restart() {
      if (raf) cancelAnimationFrame(raf);
      tr = new Trainer(sparsityOf(tS), decay, seed);
      note.textContent = "Training…";
      raf = requestAnimationFrame(frame);
    }

    function frame() {
      for (var k = 0; k < PER_FRAME && tr.t < STEPS; k++) tr.step();
      draw();
      var info = readout();
      if (tr.t < STEPS) {
        note.textContent = "Training with Adam… step " + tr.t + " of " + STEPS
          + ". The arrows are the columns of W.";
        raf = requestAnimationFrame(frame);
      } else {
        raf = null;
        note.textContent = describe(info);
      }
    }

    function redraw() { draw(); readout(); }
    window.addEventListener("resize", redraw);
    new MutationObserver(redraw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });

    restart();
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="superposition"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

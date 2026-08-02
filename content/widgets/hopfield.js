// Interactive Hopfield network: draw on the grid, corrupt it, and watch the
// network relax back into one of its stored memories.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-hopfield
//   :::
//
// The dynamics here are the real thing, not a canned animation: Hebbian
// weights, asynchronous sign updates in random order, and the Lyapunov energy
// E = -1/2 sum_ij w_ij s_i s_j, which is guaranteed to be non-increasing.
(function () {
  "use strict";

  var SIDE = 12;              // grid is SIDE x SIDE
  var N = SIDE * SIDE;
  var PER_FRAME = 5;          // neuron updates per animation frame
  var HIST = 260;             // energy samples kept for the sparkline

  // The glyphs are deliberately thick: they sit near 50% density and have low
  // mutual correlation (max |overlap| 0.39), which is what makes them stable
  // attractors. Thin, sparse letters share a strong negative mean, and under the
  // plain Hebbian rule they stop being fixed points at all.
  var PATTERNS = {
    T: [
      "############",
      "############",
      "############",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
      "....####....",
    ],
    O: [
      "...######...",
      "..########..",
      ".####..####.",
      "####....####",
      "###......###",
      "###......###",
      "###......###",
      "###......###",
      "####....####",
      ".####..####.",
      "..########..",
      "...######...",
    ],
    X: [
      "###......###",
      "####....####",
      ".####..####.",
      "..########..",
      "...######...",
      "....####....",
      "....####....",
      "...######...",
      "..########..",
      ".####..####.",
      "####....####",
      "###......###",
    ],
  };

  var NAMES = Object.keys(PATTERNS);

  function toVector(rows) {
    var v = new Float32Array(N);
    for (var r = 0; r < SIDE; r++) {
      for (var c = 0; c < SIDE; c++) {
        v[r * SIDE + c] = rows[r].charAt(c) === "#" ? 1 : -1;
      }
    }
    return v;
  }

  var STORED = NAMES.map(function (k) { return toVector(PATTERNS[k]); });

  // Hebbian outer-product rule, zero diagonal.
  var W = new Float32Array(N * N);
  (function buildWeights() {
    for (var m = 0; m < STORED.length; m++) {
      var p = STORED[m];
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          if (i !== j) W[i * N + j] += (p[i] * p[j]) / N;
        }
      }
    }
  })();

  function energy(s) {
    var e = 0;
    for (var i = 0; i < N; i++) {
      var row = i * N;
      for (var j = 0; j < N; j++) e -= W[row + j] * s[i] * s[j];
    }
    return e / 2;
  }

  function overlap(s, p) {
    var m = 0;
    for (var i = 0; i < N; i++) m += s[i] * p[i];
    return m / N;
  }

  function cssVar(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function shuffled(n) {
    var a = new Int32Array(n);
    for (var i = 0; i < n; i++) a[i] = i;
    for (var k = n - 1; k > 0; k--) {
      var j = Math.floor(Math.random() * (k + 1));
      var t = a[k]; a[k] = a[j]; a[j] = t;
    }
    return a;
  }

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function rect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
      ctx.fill();
    } else {
      ctx.fillRect(x, y, w, h);
    }
  }

  function build(mount) {
    var state = new Float32Array(STORED[0]);
    var flash = new Float32Array(N);   // recently flipped neurons, for a fade
    var hist = [];                     // energy trace for the sparkline
    var running = false;
    var queue = null, qi = 0, sweepChanged = false, updates = 0, raf = null;

    // --- chrome ------------------------------------------------------------
    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Hopfield network";
    el("div", "widget-meta", head).textContent =
      N + " neurons · " + STORED.length + " stored memories";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.setAttribute("aria-label",
      "Network state grid: click or drag to flip neurons");

    var side = el("div", "widget-side", body);

    el("div", "widget-label", side).textContent = "Stored memories";
    var chipRow = el("div", "widget-chips", side);
    var chips = NAMES.map(function (name, idx) {
      var chip = el("button", "widget-chip", chipRow);
      chip.type = "button";
      chip.title = "Load stored pattern " + name;
      var c = el("canvas", null, chip);
      c.width = 40; c.height = 40;
      var lbl = el("span", "widget-chip-name", chip);
      lbl.textContent = name;
      chip.addEventListener("click", function () { loadPattern(idx); });
      return { chip: chip, canvas: c };
    });

    el("div", "widget-label", side).textContent = "Overlap with current state";
    var rows = NAMES.map(function (name) {
      var row = el("div", "widget-row", side);
      el("span", "widget-row-name", row).textContent = name;
      var meter = el("div", "widget-meter", row);
      var fill = el("span", "widget-meter-fill", meter);
      var val = el("span", "widget-row-val", row);
      return { fill: fill, val: val };
    });

    var eRow = el("div", "widget-energy", side);
    el("div", "widget-label", eRow).textContent = "Energy";
    var eVal = el("b", null, eRow);
    var spark = el("canvas", "widget-spark", side);

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

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var sctx = spark.getContext("2d");

    function palette() {
      return {
        on: cssVar(mount, "--accent", "#ba3925"),
        off: cssVar(mount, "--block-bg", "#f3f3f2"),
        line: cssVar(mount, "--border", "#e2e2dd"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        bg: cssVar(mount, "--card-bg", "#fff"),
      };
    }

    function drawGrid() {
      var w = stage.clientWidth || 300;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = w + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);

      var p = palette();
      var cell = w / SIDE;
      var pad = Math.max(1, cell * 0.06);
      var r = Math.max(1.5, cell * 0.16);

      for (var i = 0; i < N; i++) {
        var cx = (i % SIDE) * cell + pad;
        var cy = Math.floor(i / SIDE) * cell + pad;
        var s = cell - pad * 2;
        ctx.fillStyle = state[i] > 0 ? p.on : p.off;
        rect(ctx, cx, cy, s, s, r);
        if (flash[i] > 0.01) {
          ctx.save();
          ctx.globalAlpha = flash[i] * 0.55;
          ctx.fillStyle = p.bg;
          rect(ctx, cx, cy, s, s, r);
          ctx.restore();
          flash[i] *= 0.82;
        }
      }
    }

    function drawSpark() {
      var w = side.clientWidth || 260;
      var h = 44;
      var dpr = window.devicePixelRatio || 1;
      spark.width = Math.round(w * dpr);
      spark.height = Math.round(h * dpr);
      sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      sctx.clearRect(0, 0, w, h);

      var p = palette();
      if (hist.length < 2) {
        sctx.strokeStyle = p.line;
        sctx.lineWidth = 1;
        sctx.beginPath();
        sctx.moveTo(0, h - 6);
        sctx.lineTo(w, h - 6);
        sctx.stroke();
        return;
      }
      var lo = Math.min.apply(null, hist);
      var hi = Math.max.apply(null, hist);
      if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
      var span = hi - lo;
      var x = function (k) { return (w - 2) * k / (hist.length - 1) + 1; };
      var y = function (v) { return h - 5 - (h - 12) * (v - lo) / span; };

      sctx.strokeStyle = p.line;
      sctx.lineWidth = 1;
      sctx.setLineDash([2, 3]);
      sctx.beginPath();
      sctx.moveTo(0, y(lo)); sctx.lineTo(w, y(lo));
      sctx.stroke();
      sctx.setLineDash([]);

      sctx.strokeStyle = p.on;
      sctx.lineWidth = 1.75;
      sctx.lineJoin = "round";
      sctx.beginPath();
      for (var k = 0; k < hist.length; k++) {
        var px = x(k), py = y(hist[k]);
        if (k === 0) sctx.moveTo(px, py); else sctx.lineTo(px, py);
      }
      sctx.stroke();

      sctx.fillStyle = p.on;
      sctx.beginPath();
      sctx.arc(x(hist.length - 1), y(hist[hist.length - 1]), 2.6, 0, 6.284);
      sctx.fill();
    }

    function drawChips() {
      var p = palette();
      chips.forEach(function (c, idx) {
        var cc = c.canvas.getContext("2d");
        var dpr = window.devicePixelRatio || 1;
        c.canvas.width = 40 * dpr;
        c.canvas.height = 40 * dpr;
        c.canvas.style.width = "40px";
        c.canvas.style.height = "40px";
        cc.setTransform(dpr, 0, 0, dpr, 0, 0);
        cc.clearRect(0, 0, 40, 40);
        var cell = 40 / SIDE;
        var pat = STORED[idx];
        for (var i = 0; i < N; i++) {
          cc.fillStyle = pat[i] > 0 ? p.on : p.off;
          cc.fillRect((i % SIDE) * cell, Math.floor(i / SIDE) * cell,
            cell + 0.5, cell + 0.5);
        }
      });
    }

    function readout() {
      var best = -Infinity, bestIdx = -1;
      for (var i = 0; i < STORED.length; i++) {
        var m = overlap(state, STORED[i]);
        rows[i].val.textContent = (m < 0 ? "−" : "+") + Math.abs(m).toFixed(2);
        var half = Math.abs(m) * 50;
        rows[i].fill.style.left = (m >= 0 ? 50 : 50 - half) + "%";
        rows[i].fill.style.width = half + "%";
        if (m >= 0) rows[i].fill.setAttribute("data-pos", "");
        else rows[i].fill.removeAttribute("data-pos");
        if (m > best) { best = m; bestIdx = i; }
        if (m > 0.999) chips[i].chip.setAttribute("data-active", "");
        else chips[i].chip.removeAttribute("data-active");
      }
      eVal.textContent = energy(state).toFixed(3);
      return { best: best, bestIdx: bestIdx };
    }

    function record() {
      hist.push(energy(state));
      if (hist.length > HIST) hist.shift();
    }

    function refresh() {
      drawGrid();
      drawSpark();
      return readout();
    }

    function reset(msg) {
      stop();
      hist.length = 0;
      updates = 0;
      flash.fill(0);
      record();
      refresh();
      note.textContent = msg;
    }

    // --- dynamics ----------------------------------------------------------
    function stepOnce() {
      var i = queue[qi++];
      var h = 0, row = i * N;
      for (var j = 0; j < N; j++) h += W[row + j] * state[j];
      var next = h >= 0 ? 1 : -1;
      if (next !== state[i]) { state[i] = next; flash[i] = 1; sweepChanged = true; }
      updates++;
    }

    function frame() {
      for (var k = 0; k < PER_FRAME && running; k++) {
        if (qi >= queue.length) {
          if (!sweepChanged) { settle(); return; }
          queue = shuffled(N); qi = 0; sweepChanged = false;
        }
        stepOnce();
      }
      if (!running) return;
      record();
      refresh();
      note.textContent = "Relaxing… " + updates + " asynchronous neuron updates. "
        + "The energy can only fall.";
      raf = requestAnimationFrame(frame);
    }

    function settle() {
      running = false;
      recallBtn.disabled = false;
      record();
      var r = refresh();
      if (r.best > 0.999) {
        note.textContent = "Settled into stored pattern " + NAMES[r.bestIdx]
          + " after " + updates + " updates — a fixed point of the dynamics.";
      } else if (r.best < -0.999) {
        note.textContent = "Settled into the inverse of pattern "
          + NAMES[r.bestIdx] + ". Every stored memory has a mirror image that "
          + "is equally stable.";
      } else {
        note.textContent = "Settled into a spurious state after " + updates
          + " updates — stable, but not one of the stored memories.";
      }
    }

    function stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      recallBtn.disabled = false;
    }

    function loadPattern(idx) {
      state.set(STORED[idx]);
      reset("Stored pattern " + NAMES[idx] + " loaded. It is already a fixed "
        + "point, so recall leaves it untouched — corrupt it first.");
    }

    // --- interaction -------------------------------------------------------
    var painting = 0;

    function cellAt(ev) {
      var r = canvas.getBoundingClientRect();
      var c = Math.floor((ev.clientX - r.left) / (r.width / SIDE));
      var y = Math.floor((ev.clientY - r.top) / (r.height / SIDE));
      if (y < 0 || y >= SIDE || c < 0 || c >= SIDE) return -1;
      return y * SIDE + c;
    }

    canvas.addEventListener("pointerdown", function (ev) {
      stop();
      var i = cellAt(ev);
      if (i < 0) return;
      ev.preventDefault();
      canvas.setPointerCapture(ev.pointerId);
      painting = state[i] > 0 ? -1 : 1;
      state[i] = painting;
      hist.length = 0;
      record();
      refresh();
      note.textContent = "Drawing. Press Recall to let the network settle from "
        + "whatever you draw.";
    });

    canvas.addEventListener("pointermove", function (ev) {
      if (!painting) return;
      var i = cellAt(ev);
      if (i < 0 || state[i] === painting) return;
      state[i] = painting;
      record();
      refresh();
    });

    function endPaint() { painting = 0; }
    canvas.addEventListener("pointerup", endPaint);
    canvas.addEventListener("pointercancel", endPaint);

    var recallBtn = button("Recall", function () {
      if (running) return;
      running = true;
      recallBtn.disabled = true;
      queue = shuffled(N);
      qi = 0;
      sweepChanged = false;
      updates = 0;
      hist.length = 0;
      record();
      raf = requestAnimationFrame(frame);
    }, true);

    button("Corrupt 30%", function () {
      var flips = shuffled(N);
      var n = Math.round(0.3 * N);
      for (var k = 0; k < n; k++) state[flips[k]] = -state[flips[k]];
      reset("30% of the neurons flipped. The memory is damaged, but the basin "
        + "of attraction around it is wide — press Recall.");
    });

    button("Randomize", function () {
      for (var i = 0; i < N; i++) state[i] = Math.random() < 0.5 ? 1 : -1;
      reset("Pure noise. From here the network usually still falls into one of "
        + "its three memories — or, sometimes, a spurious mixture of them.");
    });

    // --- lifecycle ---------------------------------------------------------
    var flips0 = shuffled(N);
    for (var k0 = 0; k0 < Math.round(0.3 * N); k0++) {
      state[flips0[k0]] = -state[flips0[k0]];
    }
    drawChips();
    reset("Three patterns are stored in the weights. This is pattern T with "
      + "30% of its neurons flipped — press Recall to watch it come back.");

    function redraw() { drawChips(); refresh(); }
    window.addEventListener("resize", redraw);
    new MutationObserver(redraw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="hopfield"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

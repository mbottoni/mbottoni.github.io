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

  function build(mount) {
    var state = new Float32Array(STORED[0]);
    var running = false;
    var queue = null;
    var qi = 0;
    var sweepChanged = false;
    var updates = 0;
    var raf = null;

    // --- DOM ---------------------------------------------------------------
    var body = document.createElement("div");
    body.className = "widget-body";

    var stage = document.createElement("div");
    stage.className = "widget-stage";
    var canvas = document.createElement("canvas");
    canvas.className = "widget-canvas";
    canvas.setAttribute("aria-label",
      "Hopfield network state grid: click or drag to flip neurons");
    stage.appendChild(canvas);

    var side = document.createElement("div");
    side.className = "widget-side";
    var overlapLabel = document.createElement("div");
    overlapLabel.className = "widget-label";
    overlapLabel.textContent = "Overlap with each stored memory";
    side.appendChild(overlapLabel);

    var bars = NAMES.map(function (name) {
      var stat = document.createElement("div");
      stat.className = "widget-stat";
      var lab = document.createElement("span");
      lab.className = "widget-stat-name";
      lab.textContent = "pattern " + name;
      var val = document.createElement("b");
      val.textContent = "0.00";
      stat.appendChild(lab);
      stat.appendChild(val);
      var bar = document.createElement("div");
      bar.className = "widget-bar";
      var fill = document.createElement("span");
      fill.style.width = "0%";
      bar.appendChild(fill);
      side.appendChild(stat);
      side.appendChild(bar);
      return { val: val, bar: bar, fill: fill };
    });

    var estat = document.createElement("div");
    estat.className = "widget-stat";
    var elab = document.createElement("span");
    elab.className = "widget-stat-name";
    elab.textContent = "energy";
    var eval_ = document.createElement("b");
    estat.appendChild(elab);
    estat.appendChild(eval_);
    side.appendChild(estat);

    var note = document.createElement("div");
    note.className = "widget-note";
    side.appendChild(note);

    body.appendChild(stage);
    body.appendChild(side);
    mount.appendChild(body);

    var controls = document.createElement("div");
    controls.className = "widget-controls";
    mount.appendChild(controls);

    function button(text, fn, primary) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "widget-btn";
      b.textContent = text;
      if (primary) b.setAttribute("data-primary", "");
      b.addEventListener("click", fn);
      controls.appendChild(b);
      return b;
    }

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function draw() {
      var cssW = stage.clientWidth || 320;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = cssW + "px";
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssW * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var bg = cssVar(mount, "--card-bg", "#fff");
      var grid = cssVar(mount, "--border", "#e2e2dd");
      var on = cssVar(mount, "--accent", "#ba3925");

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, cssW, cssW);

      var cell = cssW / SIDE;
      for (var r = 0; r < SIDE; r++) {
        for (var c = 0; c < SIDE; c++) {
          if (state[r * SIDE + c] > 0) {
            ctx.fillStyle = on;
            ctx.fillRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2);
          }
        }
      }
      ctx.strokeStyle = grid;
      ctx.lineWidth = 1;
      for (var k = 0; k <= SIDE; k++) {
        var p = Math.round(k * cell) + 0.5;
        ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, cssW); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(cssW, p); ctx.stroke();
      }
    }

    function readout() {
      var best = -Infinity, bestIdx = -1;
      for (var i = 0; i < STORED.length; i++) {
        var m = overlap(state, STORED[i]);
        bars[i].val.textContent = (m >= 0 ? " " : "") + m.toFixed(2);
        bars[i].fill.style.width = (Math.abs(m) * 100).toFixed(1) + "%";
        if (m > best) { best = m; bestIdx = i; }
      }
      for (var j = 0; j < bars.length; j++) {
        if (j === bestIdx && best > 0.95) bars[j].bar.setAttribute("data-win", "");
        else bars[j].bar.removeAttribute("data-win");
      }
      eval_.textContent = energy(state).toFixed(3);
      return { best: best, bestIdx: bestIdx };
    }

    function refresh() {
      draw();
      return readout();
    }

    function say(msg) { note.textContent = msg; }

    // --- dynamics ----------------------------------------------------------
    function step() {
      // One asynchronous update: s_i <- sign(sum_j w_ij s_j).
      var i = queue[qi++];
      var h = 0;
      var row = i * N;
      for (var j = 0; j < N; j++) h += W[row + j] * state[j];
      var next = h >= 0 ? 1 : -1;
      if (next !== state[i]) { state[i] = next; sweepChanged = true; }
      updates++;
    }

    function frame() {
      for (var k = 0; k < PER_FRAME && running; k++) {
        if (qi >= queue.length) {
          if (!sweepChanged) { settle(); return; }
          queue = shuffled(N); qi = 0; sweepChanged = false;
        }
        step();
      }
      if (!running) return;
      var r = refresh();
      say("Relaxing… " + updates + " neuron updates, energy falling toward the "
        + "nearest attractor.");
      raf = requestAnimationFrame(frame);
      void r;
    }

    function settle() {
      running = false;
      recallBtn.disabled = false;
      var r = refresh();
      if (r.best > 0.95) {
        say("Settled into stored pattern " + NAMES[r.bestIdx] + " after "
          + updates + " updates — a fixed point of the dynamics.");
      } else if (r.best < -0.95) {
        say("Settled into the inverse of pattern " + NAMES[r.bestIdx]
          + ". Every stored memory has a mirror image that is equally stable.");
      } else {
        say("Settled into a spurious state after " + updates + " updates — a "
          + "stable mixture that is not one of the stored memories.");
      }
    }

    function stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      recallBtn.disabled = false;
    }

    // --- interaction -------------------------------------------------------
    var painting = 0;

    function cellAt(ev) {
      var rect = canvas.getBoundingClientRect();
      var c = Math.floor((ev.clientX - rect.left) / (rect.width / SIDE));
      var r = Math.floor((ev.clientY - rect.top) / (rect.height / SIDE));
      if (r < 0 || r >= SIDE || c < 0 || c >= SIDE) return -1;
      return r * SIDE + c;
    }

    canvas.addEventListener("pointerdown", function (ev) {
      stop();
      var i = cellAt(ev);
      if (i < 0) return;
      ev.preventDefault();
      canvas.setPointerCapture(ev.pointerId);
      painting = state[i] > 0 ? -1 : 1;
      state[i] = painting;
      refresh();
      say("Drawing. Hit Recall to let the network settle from here.");
    });

    canvas.addEventListener("pointermove", function (ev) {
      if (!painting) return;
      var i = cellAt(ev);
      if (i < 0 || state[i] === painting) return;
      state[i] = painting;
      refresh();
    });

    function endPaint() { painting = 0; }
    canvas.addEventListener("pointerup", endPaint);
    canvas.addEventListener("pointercancel", endPaint);

    NAMES.forEach(function (name, idx) {
      button("Load " + name, function () {
        stop();
        state.set(STORED[idx]);
        updates = 0;
        refresh();
        say("Stored pattern " + name + " loaded. It is already a fixed point, "
          + "so recall leaves it unchanged — corrupt it first.");
      });
    });

    button("Corrupt 30%", function () {
      stop();
      var flips = shuffled(N);
      var n = Math.round(0.3 * N);
      for (var k = 0; k < n; k++) state[flips[k]] = -state[flips[k]];
      updates = 0;
      refresh();
      say("30% of the neurons flipped. The memory is damaged but the basin of "
        + "attraction is wide — press Recall.");
    });

    var recallBtn = button("Recall", function () {
      if (running) return;
      running = true;
      recallBtn.disabled = true;
      queue = shuffled(N);
      qi = 0;
      sweepChanged = false;
      updates = 0;
      raf = requestAnimationFrame(frame);
    }, true);

    button("Randomize", function () {
      stop();
      for (var i = 0; i < N; i++) state[i] = Math.random() < 0.5 ? 1 : -1;
      updates = 0;
      refresh();
      say("Pure noise. From here the network usually still falls into one of "
        + "its three memories — or, sometimes, a spurious mixture of them.");
    });

    // --- lifecycle ---------------------------------------------------------
    var initial = new Float32Array(STORED[0]);
    var flips0 = shuffled(N);
    for (var k0 = 0; k0 < Math.round(0.3 * N); k0++) {
      initial[flips0[k0]] = -initial[flips0[k0]];
    }
    state.set(initial);
    refresh();
    say("Three patterns are stored in the weights. This is pattern T with 30% "
      + "of its neurons flipped — press Recall to watch it come back.");

    window.addEventListener("resize", draw);
    new MutationObserver(draw).observe(document.documentElement, {
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

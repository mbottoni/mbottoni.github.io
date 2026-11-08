// One learnable KAN edge, phi(x) = w_b * silu(x) + sum_m c_m B_m(x): drag the
// B-spline coefficients and watch the curve and its basis functions respond.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-kan-spline
//   :::
//
// The basis is the real thing: Cox-de Boor recursion on a uniform grid of G
// intervals over [-1, 1], extended by k knots on each side, giving G + k basis
// functions of order k. Each control point sits at its Greville abscissa
// (the mean of the k interior knots of its basis function), which is where the
// control polygon of a B-spline naturally lives. Grid refinement G -> 2G is a
// least-squares refit; because the coarse knots are a subset of the fine ones
// the refit is exact on [-1, 1] up to floating point (the coarse spline space
// is a subspace of the fine one), while coarsening
// is lossy.
(function () {
  "use strict";

  var LO = -1, HI = 1;           // grid domain
  var VIEW = 1.6;                // plotted x window is at least [-VIEW, VIEW]
  var YMIN = -1.6, YMAX = 1.6;
  var G0 = 5, K0 = 3, GMAX = 40, GMIN = 2;
  var PTS = 360;
  var FIT_N = 400;

  function target(x) { return Math.sin(4 * x); }
  function silu(x) { return x / (1 + Math.exp(-x)); }

  function knots(G, k) {
    var h = (HI - LO) / G, t = [];
    for (var j = -k; j <= G + k; j++) t.push(LO + h * j);
    return t;
  }

  // Cox-de Boor. Returns all G + k basis values of order k at x.
  function basis(x, t, k) {
    var n0 = t.length - 1, B = new Float64Array(n0), i, p;
    for (i = 0; i < n0; i++) B[i] = (x >= t[i] && x < t[i + 1]) ? 1 : 0;
    for (p = 1; p <= k; p++) {
      for (i = 0; i < n0 - p; i++) {
        var a = (x - t[i]) / (t[i + p] - t[i]);
        var b = (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]);
        B[i] = a * B[i] + b * B[i + 1];
      }
    }
    return B.subarray(0, t.length - k - 1);
  }

  function greville(t, k) {
    var n = t.length - k - 1, g = [];
    for (var m = 0; m < n; m++) {
      var s = 0;
      for (var j = 1; j <= k; j++) s += t[m + j];
      g.push(k > 0 ? s / k : 0.5 * (t[m] + t[m + 1]));
    }
    return g;
  }

  // Least squares: coefficients on grid (G, k) that best match f on [LO, HI].
  function fit(f, G, k) {
    var t = knots(G, k), n = G + k, A = [], r, c, i;
    for (r = 0; r < n; r++) { A.push(new Float64Array(n + 1)); }
    for (i = 0; i < FIT_N; i++) {
      var x = LO + (HI - LO) * (i + 0.5) / FIT_N;
      var B = basis(x, t, k), y = f(x);
      for (r = 0; r < n; r++) {
        if (!B[r]) continue;
        for (c = 0; c < n; c++) A[r][c] += B[r] * B[c];
        A[r][n] += B[r] * y;
      }
    }
    for (r = 0; r < n; r++) A[r][r] += 1e-10;
    // Gaussian elimination with partial pivoting.
    for (c = 0; c < n; c++) {
      var piv = c;
      for (r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      var tmp = A[c]; A[c] = A[piv]; A[piv] = tmp;
      for (r = c + 1; r < n; r++) {
        var m = A[r][c] / A[c][c];
        if (!m) continue;
        for (var q = c; q <= n; q++) A[r][q] -= m * A[c][q];
      }
    }
    var out = new Float64Array(n);
    for (r = n - 1; r >= 0; r--) {
      var s = A[r][n];
      for (c = r + 1; c < n; c++) s -= A[r][c] * out[c];
      out[r] = s / A[r][r];
    }
    return Array.prototype.slice.call(out);
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

  function rgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    if (!/^[0-9a-fA-F]{6}$/.test(h)) return hex;
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + a + ")";
  }

  function initial(G, k) {
    return fit(function (x) { return 0.9 * Math.sin(2.4 * x) + 0.35 * x * x - 0.2; }, G, k);
  }

  function build(mount) {
    var S = { G: G0, k: K0, wb: 0, c: null, t: null, gr: null,
              showTarget: false, active: -1 };

    function setGrid(G, k, coeffs) {
      S.G = G; S.k = k; S.t = knots(G, k); S.gr = greville(S.t, k);
      S.c = coeffs;
    }
    setGrid(G0, K0, initial(G0, K0));

    // --- chrome ------------------------------------------------------------
    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "One KAN edge: a learnable B-spline";
    var meta = el("div", "widget-meta", head);
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.setAttribute("aria-label",
      "Plot of the edge function phi(x) and its B-spline basis; drag the control points up and down");

    var sliders = el("div", null, mount);
    sliders.style.marginTop = "0.8rem";
    function slider(label, min, max, step, value, fmt, fn) {
      var row = el("div", "widget-slider-row", sliders);
      el("span", null, row).textContent = label;
      var s = el("input", "widget-slider", row);
      s.type = "range"; s.min = min; s.max = max; s.step = step; s.value = value;
      var v = el("span", "widget-slider-val", row);
      v.textContent = fmt(value);
      s.addEventListener("input", function () {
        v.textContent = fmt(+s.value); fn(+s.value);
      });
      return { input: s, val: v, set: function (x) { s.value = x; v.textContent = fmt(x); } };
    }
    var kSlider = slider("Spline order k", 1, 3, 1, K0,
      function (x) { return ["", "linear", "quadratic", "cubic"][x]; },
      function (k) {
        // Refit the current spline part on the new order, same grid.
        var old = spline;
        setGrid(S.G, k, fit(old, S.G, k));
        S.active = -1;
        say("Order k = " + k + ": each basis function is a piecewise polynomial of "
          + "degree " + k + " spanning " + (k + 1) + " grid intervals, and there are "
          + "G + k = " + (S.G + k) + " of them.");
        refresh();
      });
    var wbSlider = slider("Base weight w_b", -1, 1, 0.05, 0,
      function (x) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2); },
      function (w) {
        S.wb = w;
        say(w === 0
          ? "No residual term: φ is exactly the spline, and the curve hugs its control polygon."
          : "φ(x) = w_b·silu(x) + spline(x). The SiLU term is the KAN paper's residual "
            + "\"base\" function; it gives the edge a sensible shape outside the grid, "
            + "where every B-spline is zero.");
        refresh();
      });

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
    function say(msg) { note.textContent = msg; }

    function spline(x) {
      var B = basis(x, S.t, S.k), s = 0;
      for (var m = 0; m < B.length; m++) s += S.c[m] * B[m];
      return s;
    }
    function phi(x) { return S.wb * silu(x) + spline(x); }

    function rmse() {
      var s = 0;
      for (var i = 0; i < FIT_N; i++) {
        var x = LO + (HI - LO) * (i + 0.5) / FIT_N;
        var d = phi(x) - target(x);
        s += d * d;
      }
      return Math.sqrt(s / FIT_N);
    }

    var refineBtn = button("Refine grid ×2", function () {
      var old = spline, G = Math.min(GMAX, S.G * 2);
      setGrid(G, S.k, fit(old, G, S.k));
      S.active = -1;
      say("Grid refined to G = " + G + ". The old knots are a subset of the new ones, "
        + "so the least-squares refit reproduces the curve exactly on [−1, 1] (outside it, the extended knots differ) — but now there are "
        + (G + S.k) + " coefficients to train, each with a narrower footprint."
        + (S.showTarget ? " Press Fit again to use the extra resolution." : ""));
      refresh();
    }, true);
    var coarsenBtn = button("Coarsen ÷2", function () {
      var old = spline, G = Math.max(GMIN, Math.round(S.G / 2));
      setGrid(G, S.k, fit(old, G, S.k));
      S.active = -1;
      say("Coarsened to G = " + G + ". This direction is lossy: the coarse spline space "
        + "cannot represent every fine curve, so you get its least-squares projection.");
      refresh();
    });
    button("Fit sin(4x)", function () {
      S.showTarget = true;
      var wb = S.wb;
      setGrid(S.G, S.k, fit(function (x) { return target(x) - wb * silu(x); }, S.G, S.k));
      S.active = -1;
      var e = rmse();
      say("Least-squares fit of sin(4x) on the grid domain with G = " + S.G + ", k = "
        + S.k + ": RMS error " + e.toExponential(1) + ". Refine and fit again — for a "
        + "smooth target the error of a cubic spline shrinks roughly like G⁻⁴.");
      refresh();
    });
    button("Reset", function () {
      S.wb = 0; wbSlider.set(0); kSlider.set(K0);
      S.showTarget = false; S.active = -1;
      setGrid(G0, K0, initial(G0, K0));
      say(INTRO);
      refresh();
    });

    var INTRO = "Drag a control point up or down. Its coefficient scales one basis "
      + "function (highlighted below), so only the part of φ over that function's "
      + "k + 1 grid intervals moves: B-splines have local support.";

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0, PAD = 14, MAIN = 0, STRIP = 0, GAP = 18;
    var VMIN = -VIEW, VMAX = VIEW;

    function px(x) { return PAD + (W - 2 * PAD) * (x - VMIN) / (VMAX - VMIN); }
    function py(y) { return PAD + (MAIN - 2 * PAD) * (YMAX - y) / (YMAX - YMIN); }
    function sy(b) { return MAIN + GAP + (STRIP - 8) * (1 - b); }
    function vx(i) { return VMIN + (VMAX - VMIN) * i / (PTS - 1); }

    function palette() {
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        blue: cssVar(mount, "--link", "#2156a5"),
        line: cssVar(mount, "--border", "#e2e2dd"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        shade: cssVar(mount, "--block-bg", "#f3f3f2"),
        bg: cssVar(mount, "--card-bg", "#fff"),
      };
    }

    function path(f, y) {
      ctx.beginPath();
      for (var i = 0; i < PTS; i++) {
        var X = px(vx(i)), Y = y(f(vx(i)));
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
    }

    function draw() {
      var cssW = mount.clientWidth || 560;
      W = cssW;
      MAIN = Math.max(210, Math.min(300, Math.round(cssW * 0.48)));
      STRIP = Math.max(70, Math.round(MAIN * 0.32));
      H = MAIN + GAP + STRIP + 4;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      var p = palette();
      // Widen the window on coarse grids so every control point stays visible.
      VMAX = Math.max(VIEW, S.gr[S.gr.length - 1] + 0.15);
      VMIN = -VMAX;

      // Outside the grid domain every B-spline eventually vanishes: shade it.
      ctx.fillStyle = p.shade;
      ctx.fillRect(0, 0, px(LO), H);
      ctx.fillRect(px(HI), 0, W - px(HI), H);
      ctx.fillStyle = p.muted;
      ctx.font = "10px 'Open Sans', sans-serif";
      ctx.textBaseline = "top";
      ctx.textAlign = "left";
      ctx.fillText("outside grid", 4, 4);
      ctx.textAlign = "right";
      ctx.fillText("outside grid", W - 4, 4);

      // Axes and knots.
      ctx.strokeStyle = p.line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, Math.round(py(0)) + 0.5); ctx.lineTo(W, Math.round(py(0)) + 0.5);
      ctx.moveTo(0, Math.round(sy(0)) + 0.5); ctx.lineTo(W, Math.round(sy(0)) + 0.5);
      ctx.stroke();
      ctx.fillStyle = p.muted;
      for (var j = 0; j < S.t.length; j++) {
        var kx = px(S.t[j]);
        if (kx < 0 || kx > W) continue;
        var interior = S.t[j] >= LO - 1e-9 && S.t[j] <= HI + 1e-9;
        ctx.globalAlpha = interior ? 0.9 : 0.4;
        ctx.fillRect(Math.round(kx) - 0.5, sy(0) - 3, 1, 6);
        ctx.globalAlpha = 1;
      }
      ctx.textAlign = "center";
      ctx.fillText("−1", px(LO), sy(0) + 5);
      ctx.fillText("+1", px(HI), sy(0) + 5);

      // Basis functions in the bottom strip.
      var B = S.c.length, m;
      for (m = 0; m < B; m++) {
        (function (mm) {
          path(function (x) { return basis(x, S.t, S.k)[mm]; }, sy);
        })(m);
        var on = m === S.active;
        ctx.strokeStyle = on ? p.accent : rgba(p.blue, 0.55);
        ctx.lineWidth = on ? 2.2 : 1.2;
        ctx.stroke();
      }

      // Active basis function's footprint in the main plot.
      if (S.active >= 0) {
        var a0 = Math.max(VMIN, S.t[S.active]), a1 = Math.min(VMAX, S.t[S.active + S.k + 1]);
        ctx.fillStyle = rgba(p.accent, 0.08);
        ctx.fillRect(px(a0), 0, px(a1) - px(a0), MAIN);
      }

      // Scaled contributions c_m B_m(x), faint.
      for (m = 0; m < B; m++) {
        (function (mm) {
          path(function (x) { return S.c[mm] * basis(x, S.t, S.k)[mm]; }, py);
        })(m);
        ctx.strokeStyle = rgba(p.blue, m === S.active ? 0.6 : 0.22);
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      if (S.wb !== 0) {
        path(function (x) { return S.wb * silu(x); }, py);
        ctx.strokeStyle = p.muted;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (S.showTarget) {
        path(target, py);
        ctx.strokeStyle = p.fg;
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // Control polygon.
      ctx.strokeStyle = p.muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (m = 0; m < B; m++) {
        var cx = px(S.gr[m]), cy = py(S.c[m]);
        if (m === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // The edge function itself.
      path(phi, py);
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 2.6;
      ctx.lineJoin = "round";
      ctx.stroke();

      // Control points.
      var r = S.G > 20 ? 4 : 6;
      for (m = 0; m < B; m++) {
        var hx = px(S.gr[m]), hy = py(S.c[m]);
        ctx.beginPath();
        ctx.arc(hx, hy, m === S.active ? r + 2 : r, 0, 6.284);
        ctx.fillStyle = m === S.active ? p.accent : p.bg;
        ctx.fill();
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.fillStyle = p.muted;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("basis B_m(x)", px(LO) + 4, MAIN + GAP - 3);
    }

    function refresh() {
      draw();
      meta.textContent = "G = " + S.G + " · k = " + S.k + " · " + S.c.length
        + " coefficients" + (S.wb !== 0 ? " + w_b" : "");
      refineBtn.disabled = S.G * 2 > GMAX;
      coarsenBtn.disabled = S.G <= GMIN;
    }

    // --- interaction -------------------------------------------------------
    var drag = -1;

    function pointer(ev) {
      var r = canvas.getBoundingClientRect();
      return { x: (ev.clientX - r.left) * (W / r.width),
               y: (ev.clientY - r.top) * (H / r.height) };
    }

    function nearest(pt) {
      var best = -1, bestD = 22;
      for (var m = 0; m < S.c.length; m++) {
        var dx = pt.x - px(S.gr[m]), dy = pt.y - py(S.c[m]);
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestD) { bestD = d; best = m; }
      }
      return best;
    }

    function describe(m) {
      var a = S.t[m], b = S.t[m + S.k + 1];
      say("Coefficient c" + m + " = " + (S.c[m] >= 0 ? "+" : "−") + Math.abs(S.c[m]).toFixed(2)
        + ". Its basis function B" + m + " is nonzero only on [" + a.toFixed(2) + ", " + b.toFixed(2)
        + "), so φ changes there and nowhere else."
        + (S.showTarget ? " RMS error vs sin(4x): " + rmse().toExponential(1) + "." : ""));
    }

    canvas.addEventListener("pointerdown", function (ev) {
      var m = nearest(pointer(ev));
      if (m < 0) return;
      ev.preventDefault();
      canvas.setPointerCapture(ev.pointerId);
      drag = m; S.active = m;
      describe(m);
      refresh();
    });

    canvas.addEventListener("pointermove", function (ev) {
      var pt = pointer(ev);
      if (drag < 0) {
        var m = nearest(pt);
        canvas.style.cursor = m >= 0 ? "grab" : "default";
        if (m !== S.active && m >= 0) { S.active = m; refresh(); }
        return;
      }
      var y = YMAX - (pt.y - PAD) * (YMAX - YMIN) / (MAIN - 2 * PAD);
      S.c[drag] = Math.max(YMIN, Math.min(YMAX, y));
      describe(drag);
      refresh();
    });

    function endDrag() { drag = -1; }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    say(INTRO);
    refresh();

    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="kan-spline"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

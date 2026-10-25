// Interactive "decoding by linear programming" (Candès & Tao, 2005).
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-decoding
//   :::
//
// A message f in R^n is encoded as A f with a random Gaussian A in R^{m x n},
// m > n, and a fraction of the m entries are then corrupted arbitrarily:
// y = A f + e with e sparse. The decoder solves
//
//     g* = argmin_g || y - A g ||_1
//
// which is a linear program. Here it is solved with ADMM (Boyd et al. 2011,
// section 6.1, least absolute deviations), run to a 1e-9 residual tolerance:
//
//     g <- (A^T A)^{-1} A^T (y + z - u)
//     z <- S_{1/rho}(A g - y + u)          (soft threshold)
//     u <- u + A g - z - y
//
// Least squares, argmin ||y - A g||_2, is shown alongside for contrast. When
// the L1 decoder misses, the widget also checks the certificate: it found a
// g with ||y - A g||_1 < ||e||_1, so the true f is provably not the LP optimum.
(function () {
  "use strict";

  var N = 32;                       // message length
  var RATIOS = [1.5, 2, 3, 4];      // redundancy m / n
  var RHO = 1;
  var MAXIT = 8000, SWEEP_MAXIT = 3000, TOL = 1e-9;
  var OK_TOL = 1e-3;                // relative error that counts as exact
  var SWEEP_FRACS = 21;             // fractions 0, 2.5%, ..., 50%
  var SWEEP_TRIALS = 20;

  // --- randomness ----------------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussFn(rand) {
    return function () {
      var u = 0;
      while (u === 0) u = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
    };
  }

  // --- linear algebra ------------------------------------------------------
  // P = (A^T A)^{-1} A^T  (n x m), via Cholesky of the Gram matrix.
  function pinv(A, m, n) {
    var G = new Float64Array(n * n), L = new Float64Array(n * n);
    var i, j, k, s;
    for (i = 0; i < n; i++) {
      for (j = 0; j <= i; j++) {
        s = 0;
        for (k = 0; k < m; k++) s += A[k * n + i] * A[k * n + j];
        G[i * n + j] = s; G[j * n + i] = s;
      }
    }
    for (i = 0; i < n; i++) {
      for (j = 0; j <= i; j++) {
        s = G[i * n + j];
        for (k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = i === j ? Math.sqrt(s) : s / L[j * n + j];
      }
    }
    var P = new Float64Array(n * m), col = new Float64Array(n);
    for (var c = 0; c < m; c++) {
      for (i = 0; i < n; i++) {
        s = A[c * n + i];
        for (k = 0; k < i; k++) s -= L[i * n + k] * col[k];
        col[i] = s / L[i * n + i];
      }
      for (i = n - 1; i >= 0; i--) {
        s = col[i];
        for (k = i + 1; k < n; k++) s -= L[k * n + i] * col[k];
        col[i] = s / L[i * n + i];
      }
      for (i = 0; i < n; i++) P[i * m + c] = col[i];
    }
    return P;
  }

  function matvec(M, v, rows, cols, out) {
    for (var i = 0; i < rows; i++) {
      var s = 0, o = i * cols;
      for (var j = 0; j < cols; j++) s += M[o + j] * v[j];
      out[i] = s;
    }
    return out;
  }

  // L1 regression by ADMM. Returns the decoded message and iteration count.
  function l1decode(A, P, y, m, n, maxIt) {
    var g = new Float64Array(n), z = new Float64Array(m), u = new Float64Array(m);
    var v = new Float64Array(m), Ag = new Float64Array(m);
    var t = 1 / RHO, it;
    for (it = 0; it < maxIt; it++) {
      for (var i = 0; i < m; i++) v[i] = y[i] + z[i] - u[i];
      matvec(P, v, n, m, g);
      matvec(A, g, m, n, Ag);
      var rp = 0, rd = 0;
      for (i = 0; i < m; i++) {
        var w = Ag[i] - y[i] + u[i];
        var zn = w > t ? w - t : (w < -t ? w + t : 0);
        rd += (zn - z[i]) * (zn - z[i]);
        z[i] = zn;
        var r = Ag[i] - zn - y[i];
        rp += r * r;
        u[i] += r;
      }
      if (Math.sqrt(rp) < TOL && RHO * Math.sqrt(rd) < TOL) break;
    }
    return { g: g, it: it };
  }

  // One complete instance: code, message, corruption, both decoders.
  function instance(seed, ratio, k, sigE, maxIt) {
    var gauss = gaussFn(mulberry32(seed));
    var pick = mulberry32(seed ^ 0x9e3779b9);
    var m = Math.round(N * ratio), n = N, i;
    var A = new Float64Array(m * n);
    for (i = 0; i < m * n; i++) A[i] = gauss() / Math.sqrt(m);
    var f = new Float64Array(n);
    for (i = 0; i < n; i++) f[i] = gauss();
    // Corruption pattern and values come from their own stream so that moving
    // the slider adds corruptions to the same message instead of redrawing it.
    var order = [];
    for (i = 0; i < m; i++) order.push(i);
    for (i = m - 1; i > 0; i--) {
      var j = Math.floor(pick() * (i + 1));
      var tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    var gE = gaussFn(mulberry32(seed * 3 + 11));
    var vals = new Float64Array(m);
    for (i = 0; i < m; i++) vals[i] = gE();
    var e = new Float64Array(m);
    for (i = 0; i < k; i++) e[order[i]] = sigE * vals[i];
    var y = matvec(A, f, m, n, new Float64Array(m));
    for (i = 0; i < m; i++) y[i] += e[i];

    var P = pinv(A, m, n);
    var l1 = l1decode(A, P, y, m, n, maxIt);
    var ls = matvec(P, y, n, m, new Float64Array(n));

    function relErr(g) {
      var a = 0, b = 0;
      for (var q = 0; q < n; q++) { a += (g[q] - f[q]) * (g[q] - f[q]); b += f[q] * f[q]; }
      return Math.sqrt(a / b);
    }
    var Ag = matvec(A, l1.g, m, n, new Float64Array(m));
    var resid = new Float64Array(m), obj = 0, objTrue = 0;
    for (i = 0; i < m; i++) {
      resid[i] = y[i] - Ag[i];
      obj += Math.abs(resid[i]);
      objTrue += Math.abs(e[i]);
    }
    return {
      m: m, n: n, k: k, f: f, e: e, y: y, l1: l1.g, ls: ls, it: l1.it,
      resid: resid, obj: obj, objTrue: objTrue,
      errL1: relErr(l1.g), errLS: relErr(ls),
    };
  }

  // --- DOM helpers ---------------------------------------------------------
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

  var styled = false;
  function injectStyle() {
    if (styled) return;
    styled = true;
    var s = document.createElement("style");
    s.textContent =
      ".dec-sub{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.6rem;margin:.9rem 0 .3rem}" +
      ".dec-sub .widget-label{margin-bottom:0}" +
      ".dec-key{font-size:.68rem;color:var(--muted);text-align:right}" +
      ".dec-key i{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin:0 .25rem 0 .6rem;vertical-align:-1px;font-style:normal}" +
      ".dec-seg{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.6rem}" +
      ".dec-seg .widget-btn{padding:.3rem .65rem}" +
      ".dec-plot{cursor:default}";
    document.head.appendChild(s);
  }

  function build(mount) {
    injectStyle();
    var st = { ratio: 2, k: 8, sigE: 2, seed: 1 };
    var cur = null;
    var sweeps = {};     // ratio -> { rate: Float64Array, done: Int32Array }
    var sweepJob = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Decoding by linear programming";
    var meta = el("div", "widget-meta", head);
    mount.parentNode.insertBefore(head, mount);

    // --- plots -------------------------------------------------------------
    function sub(title, keys) {
      var d = el("div", "dec-sub", mount);
      el("div", "widget-label", d).textContent = title;
      var k = el("div", "dec-key", d);
      return { label: d.firstChild, key: k, keys: keys };
    }
    var sub1 = sub("Message f");
    var c1 = el("canvas", "widget-plot dec-plot", mount);
    c1.setAttribute("aria-label", "True message, L1 decode and least-squares decode");
    var sub2 = sub("Corruptions e");
    var c2 = el("canvas", "widget-plot dec-plot", mount);
    c2.setAttribute("aria-label", "True corruptions and the residual of the L1 decode");
    sub1.label.style.marginTop = "0";
    sub1.label.parentNode.style.marginTop = "0";

    // --- controls + stats --------------------------------------------------
    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "Redundancy m / n";
    var seg = el("div", "dec-seg", left);
    var ratioBtns = RATIOS.map(function (r) {
      var b = el("button", "widget-btn", seg);
      b.type = "button";
      b.textContent = r + "×";
      b.addEventListener("click", function () {
        var frac = st.k / Math.round(N * st.ratio);
        st.ratio = r;
        var m = Math.round(N * r);
        kIn.max = String(Math.floor(m * 0.6));
        st.k = Math.round(frac * m);
        kIn.value = String(st.k);
        refresh();
      });
      return { b: b, r: r };
    });

    function slider(label, cls, min, max, value, onInput) {
      var row = el("div", "widget-slider-row", left);
      el("span", cls, row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range"; input.min = String(min); input.max = String(max);
      input.step = "1"; input.value = String(value);
      input.setAttribute("aria-label", label);
      var val = el("span", "widget-slider-val", row);
      input.addEventListener("input", function () { onInput(Number(input.value)); });
      return { input: input, val: val };
    }
    var kS = slider("corrupted", "dec-l1", 0, Math.floor(N * st.ratio * 0.6), st.k,
      function (v) { st.k = v; refresh(); });
    var kIn = kS.input;
    var SIG_MIN = 0.05, SIG_MAX = 50;
    function sigPos(v) { return Math.round(1000 * Math.log(v / SIG_MIN) / Math.log(SIG_MAX / SIG_MIN)); }
    function sigVal(p) { return SIG_MIN * Math.pow(SIG_MAX / SIG_MIN, p / 1000); }
    var sS = slider("error size σ", "dec-l2", 0, 1000, sigPos(st.sigE),
      function (p) { st.sigE = sigVal(p); refresh(); });

    el("div", "widget-label", right).textContent = "Result";
    function stat(name) {
      var row = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", row).textContent = name;
      var v = el("span", "widget-row-val", row);
      v.style.width = "auto";
      v.style.flex = "1 1 auto";
      return v;
    }
    var stL1 = stat("L1 rel. error");
    var stLS = stat("L2 rel. error");
    var stObj = stat("‖y−Ag‖₁");
    var stIt = stat("ADMM iters");

    var sub3 = sub("Exact-recovery rate vs. fraction corrupted");
    var c3 = el("canvas", "widget-plot dec-plot", mount);
    c3.setAttribute("aria-label",
      "Fraction of random trials recovered exactly, as a function of the corrupted fraction");
    sub3.label.parentNode.style.marginTop = "1.2rem";

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
    var sweepBtn = button("Run phase transition", startSweep, true);
    button("New draw", function () { st.seed++; refresh(); });

    // --- drawing -----------------------------------------------------------
    function palette() {
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        link: cssVar(mount, "--link", "#2156a5"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        border: cssVar(mount, "--border", "#e2e2dd"),
        block: cssVar(mount, "--block-bg", "#f3f3f2"),
        bg: cssVar(mount, "--card-bg", "#fff"),
      };
    }

    function setup(c, h) {
      var W = Math.max(200, mount.clientWidth - 32);
      var dpr = window.devicePixelRatio || 1;
      c.style.height = h + "px";
      c.width = Math.round(W * dpr);
      c.height = Math.round(h * dpr);
      var ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, h);
      return { ctx: ctx, W: W, H: h };
    }

    function dot(ctx, x, y, r, fill, stroke) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.stroke(); }
    }

    // Bars for `truth`, markers for each overlay series, symmetric y-axis.
    function stems(c, h, truth, series, ymax, flags) {
      var s = setup(c, h), ctx = s.ctx, p = palette();
      var n = truth.length, padX = 6, padY = 8;
      var bw = (s.W - 2 * padX) / n;
      var mid = s.H / 2;
      var sy = (s.H / 2 - padY) / ymax;
      function cy(v) { return mid - Math.max(-ymax, Math.min(ymax, v)) * sy; }
      ctx.strokeStyle = p.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padX, Math.round(mid) + 0.5); ctx.lineTo(s.W - padX, Math.round(mid) + 0.5);
      ctx.stroke();
      for (var i = 0; i < n; i++) {
        var x = padX + i * bw;
        ctx.fillStyle = p.muted;
        ctx.globalAlpha = flags ? 0.85 : 0.35;
        var top = cy(truth[i]);
        ctx.fillRect(x + bw * 0.18, Math.min(top, mid), bw * 0.64, Math.max(1, Math.abs(top - mid)));
        ctx.globalAlpha = 1;
        if (Math.abs(truth[i]) > ymax) {      // clipped: mark with a cap
          ctx.fillStyle = p.fg;
          ctx.fillRect(x + bw * 0.18, top + (truth[i] > 0 ? 0 : -2), bw * 0.64, 2);
        }
      }
      series.forEach(function (sr) {
        for (var i = 0; i < n; i++) {
          var x = padX + (i + 0.5) * bw;
          var r = Math.max(2, Math.min(4, bw * 0.28));
          if (sr.hollow) dot(ctx, x, cy(sr.v[i]), r + 0.6, null, sr.color);
          else dot(ctx, x, cy(sr.v[i]), r, sr.color);
        }
      });
    }

    function drawPhase() {
      var s = setup(c3, 150), ctx = s.ctx, p = palette();
      var padL = 34, padR = 10, top = 8, bot = 22;
      var pw = s.W - padL - padR, ph = s.H - top - bot;
      function px(fr) { return padL + pw * fr / 0.5; }
      function py(r) { return top + ph * (1 - r); }
      ctx.strokeStyle = p.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(padL + 0.5, top + 0.5, pw, ph);
      ctx.fillStyle = p.muted;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText("1", padL - 5, py(1));
      ctx.fillText("0", padL - 5, py(0));
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(function (fr) {
        ctx.fillText(Math.round(fr * 100) + "%", px(fr), top + ph + 5);
      });
      // current fraction marker
      var frac = st.k / Math.round(N * st.ratio);
      if (frac <= 0.5) {
        ctx.strokeStyle = p.fg;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(px(frac) + 0.5, top); ctx.lineTo(px(frac) + 0.5, top + ph);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      var any = false;
      RATIOS.forEach(function (r) {
        var sw = sweeps[r];
        if (!sw) return;
        any = true;
        var active = r === st.ratio;
        ctx.strokeStyle = active ? p.accent : p.muted;
        ctx.globalAlpha = active ? 1 : 0.55;
        ctx.lineWidth = active ? 2 : 1.25;
        ctx.beginPath();
        var started = false, last = null;
        for (var i = 0; i < SWEEP_FRACS; i++) {
          if (sw.done[i] < SWEEP_TRIALS) continue;
          var x = px(i * 0.5 / (SWEEP_FRACS - 1)), y = py(sw.rate[i]);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          last = [x, y];
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (last) {
          ctx.fillStyle = active ? p.accent : p.muted;
          ctx.font = (active ? "600 " : "") + '10px "Open Sans", sans-serif';
          ctx.textAlign = "left"; ctx.textBaseline = "bottom";
          // label each curve where it crosses 50%
          for (i = 0; i < SWEEP_FRACS - 1; i++) {
            if (sw.done[i + 1] < SWEEP_TRIALS) break;
            if (sw.rate[i] >= 0.5 && sw.rate[i + 1] < 0.5) {
              ctx.fillText(r + "×", px((i + 0.5) * 0.5 / (SWEEP_FRACS - 1)) + 4, py(0.5) - 2);
              break;
            }
          }
        }
      });
      if (!any) {
        ctx.fillStyle = p.muted;
        ctx.font = '11px "Open Sans", sans-serif';
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Press “Run phase transition” to solve", padL + pw / 2, top + ph / 2 - 8);
        ctx.fillText(SWEEP_FRACS * SWEEP_TRIALS + " random LPs at m/n = " + st.ratio,
          padL + pw / 2, top + ph / 2 + 8);
      }
    }

    function draw() {
      var p = palette();
      var fmax = 0, i;
      for (i = 0; i < cur.n; i++) fmax = Math.max(fmax, Math.abs(cur.f[i]));
      stems(c1, 130, cur.f, [
        { v: cur.ls, color: p.link, hollow: true },
        { v: cur.l1, color: p.accent },
      ], fmax * 1.35);
      var emax = 0, flags = new Uint8Array(cur.m);
      for (i = 0; i < cur.m; i++) {
        if (cur.e[i] !== 0) flags[i] = 1;
        emax = Math.max(emax, Math.abs(cur.e[i]), Math.abs(cur.resid[i]));
      }
      stems(c2, 130, cur.e, [{ v: cur.resid, color: p.accent }], Math.max(emax * 1.1, 0.5), flags);
      drawPhase();
    }

    function keys() {
      var p = palette();
      sub1.key.innerHTML = "";
      sub2.key.innerHTML = "";
      function k(target, color, text, hollow) {
        var i = el("i", null, target);
        if (hollow) i.style.boxShadow = "inset 0 0 0 1.5px " + color;
        else i.style.background = color;
        target.appendChild(document.createTextNode(text));
      }
      var b = el("i", null, sub1.key);
      b.style.borderRadius = "1px"; b.style.background = p.muted; b.style.opacity = "0.45";
      sub1.key.appendChild(document.createTextNode("true f"));
      k(sub1.key, p.accent, "L1 (LP)");
      k(sub1.key, p.link, "least squares", true);
      var b2 = el("i", null, sub2.key);
      b2.style.borderRadius = "1px"; b2.style.background = p.muted;
      sub2.key.appendChild(document.createTextNode("true e"));
      k(sub2.key, p.accent, "y − A g*");
    }

    function readout() {
      var ok = cur.errL1 < OK_TOL;
      meta.textContent = "n = " + cur.n + " · m = " + cur.m + " · seed " + st.seed;
      kS.val.textContent = Math.round(100 * cur.k / cur.m) + "%";
      sS.val.textContent = st.sigE < 1 ? st.sigE.toFixed(2) : st.sigE.toFixed(1);
      function rel(x) { return x < 1e-6 ? "< 1e-6" : x < 1e-2 ? x.toExponential(1) : (100 * x).toFixed(1) + "%"; }
      stL1.textContent = rel(cur.errL1);
      stLS.textContent = rel(cur.errLS);
      stObj.textContent = cur.obj.toFixed(3) + (ok ? " = ‖e‖₁" : " < ‖e‖₁ " + cur.objTrue.toFixed(3));
      if (!ok && cur.obj >= cur.objTrue - 1e-9) stObj.textContent = cur.obj.toFixed(3) + " (‖e‖₁ " + cur.objTrue.toFixed(3) + ")";
      stIt.textContent = String(cur.it) + (cur.it >= MAXIT ? " (cap)" : "");
      sub2.label.textContent = "Corruptions e (" + cur.k + " of " + cur.m + " entries)";
      ratioBtns.forEach(function (x) {
        if (x.r === st.ratio) x.b.setAttribute("data-primary", ""); else x.b.removeAttribute("data-primary");
        x.b.setAttribute("aria-pressed", x.r === st.ratio ? "true" : "false");
      });

      var msg;
      if (cur.k === 0) {
        msg = "No corruption: both decoders return f exactly, since y = Af has a "
          + "unique solution when A has full column rank.";
      } else if (ok) {
        msg = "Exact recovery. The LP solution reproduces f to " + (cur.errL1 < 1e-6 ? "solver precision" : cur.errL1.toExponential(1))
          + ", and y − Ag* is exactly the sparse error vector. Least squares "
          + "spreads the " + cur.k + " gross errors over every coordinate.";
      } else if (cur.obj < cur.objTrue - 1e-9) {
        msg = "Too many corruptions. The LP found a g with a smaller ℓ₁ residual "
          + "than the true message has, so f is provably not the LP optimum "
          + "and no ℓ₁ solver could recover it here.";
      } else {
        msg = "Not recovered within the iteration cap.";
      }
      note.textContent = msg;
    }

    function refresh() {
      cur = instance(st.seed * 1009 + Math.round(st.ratio * 10), st.ratio, st.k, st.sigE, MAXIT);
      keys();
      draw();
      readout();
    }

    // --- phase-transition sweep -------------------------------------------
    function startSweep() {
      var r = st.ratio;
      if (sweepJob) return;
      var sw = sweeps[r] = { rate: new Float64Array(SWEEP_FRACS), done: new Int32Array(SWEEP_FRACS) };
      var m = Math.round(N * r);
      var fi = 0, t = 0, okCount = 0;
      sweepBtn.disabled = true;
      sweepJob = function step() {
        var t0 = performance.now();
        while (performance.now() - t0 < 24 && fi < SWEEP_FRACS) {
          var k = Math.round(fi * 0.5 / (SWEEP_FRACS - 1) * m);
          var inst = instance(90001 + 7919 * t + 104729 * fi + Math.round(r * 10), r, k, 2, SWEEP_MAXIT);
          if (inst.errL1 < OK_TOL) okCount++;
          t++;
          sw.done[fi] = t;
          sw.rate[fi] = okCount / t;
          if (t === SWEEP_TRIALS) { fi++; t = 0; okCount = 0; }
        }
        drawPhase();
        note.textContent = "Solving random instances at m/n = " + r + "… "
          + Math.min(fi, SWEEP_FRACS) + " of " + SWEEP_FRACS + " corruption levels done.";
        if (fi < SWEEP_FRACS) { requestAnimationFrame(step); return; }
        sweepJob = null;
        sweepBtn.disabled = false;
        var half = null;
        for (var i = 0; i < SWEEP_FRACS - 1; i++) {
          if (sw.rate[i] >= 0.5 && sw.rate[i + 1] < 0.5) { half = (i + 0.5) * 0.5 / (SWEEP_FRACS - 1); break; }
        }
        note.textContent = "At m/n = " + r + ", " + SWEEP_TRIALS + " fresh random "
          + "codes per level: recovery is essentially certain up to a threshold and "
          + "then collapses" + (half !== null ? ", crossing 50% near " + Math.round(half * 100) + "% corrupted" : "")
          + ". Switch the redundancy and run again to compare curves.";
      };
      requestAnimationFrame(sweepJob);
    }

    refresh();
    window.addEventListener("resize", function () { draw(); });
    new MutationObserver(function () { keys(); draw(); }).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="decoding"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

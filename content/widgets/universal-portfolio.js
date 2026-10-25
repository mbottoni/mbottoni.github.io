// Cover's universal portfolio on a two-stock market you can regenerate.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-universal-portfolio
//   :::
//
// Each stock has i.i.d. Gaussian log-returns  log x_{t,i} = mu_i + sigma z_{t,i}
// with a tiny random drift mu_i, so on its own each one mostly wanders. The top
// plot is the final wealth S_T(b) = prod_t (b x_{t,1} + (1-b) x_{t,2}) of every
// constant-rebalanced portfolio b in [0,1]; its endpoints are the two
// buy-and-hold stocks. The universal portfolio's wealth is the uniform average
// of that curve (Cover's identity: playing b_hat_{t+1} = int b S_t / int S_t day
// by day yields exactly int S_T(b) db), computed here by the trapezoid rule on
// 201 points. The guarantee shown is log S_T(b*) - log S_hat_T <= log(T+1).
(function () {
  "use strict";

  var T_MAX = 1000;
  var K = 200;                 // grid b_k = k / K, k = 0..K
  var SIG_MIN = 0.01, SIG_MAX = 0.15;

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

  function rgba(color, a) {
    var h = color.replace("#", "");
    if (!/^[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$/.test(h)) return color;
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + a + ")";
  }

  // Deterministic PRNG so a seed always regenerates the same market.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rnd) {
    var u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Standard-normal shocks and a drift direction per seed; sigma scales them
  // later, so moving the volatility slider reshapes the same market.
  function shocks(seed) {
    var rnd = mulberry32(seed);
    var z1 = new Float64Array(T_MAX), z2 = new Float64Array(T_MAX);
    for (var t = 0; t < T_MAX; t++) { z1[t] = gauss(rnd); z2[t] = gauss(rnd); }
    return { z1: z1, z2: z2, d1: rnd() * 2 - 1, d2: rnd() * 2 - 1 };
  }

  function logMeanExp(a) {
    // trapezoid average of exp(a_k) over the uniform grid on [0,1]
    var m = -Infinity, k;
    for (k = 0; k <= K; k++) if (a[k] > m) m = a[k];
    var s = 0;
    for (k = 0; k <= K; k++) {
      var w = (k === 0 || k === K) ? 0.5 : 1;
      s += w * Math.exp(a[k] - m);
    }
    return m + Math.log(s / K);
  }

  function simulate(z, sigma, T) {
    // drift up to +/- sigma^2 / 4: comparable to the rebalancing bonus, so
    // sometimes one stock clearly wins and sometimes neither does
    var mu1 = z.d1 * sigma * sigma / 4, mu2 = z.d2 * sigma * sigma / 4;
    var x1 = new Float64Array(T), x2 = new Float64Array(T);
    for (var t = 0; t < T; t++) {
      x1[t] = Math.exp(mu1 + sigma * z.z1[t]);
      x2[t] = Math.exp(mu2 + sigma * z.z2[t]);
    }
    var L = new Float64Array(K + 1);            // log S_t(b_k), running
    var up = new Float64Array(T + 1);           // log S_hat_t
    var a1 = new Float64Array(T + 1), a2 = new Float64Array(T + 1);
    for (t = 0; t < T; t++) {
      for (var k = 0; k <= K; k++) {
        var b = k / K;
        L[k] += Math.log(b * x1[t] + (1 - b) * x2[t]);
      }
      up[t + 1] = logMeanExp(L);
      a1[t + 1] = a1[t] + Math.log(x1[t]);
      a2[t + 1] = a2[t] + Math.log(x2[t]);
    }
    var ks = 0;
    for (k = 1; k <= K; k++) if (L[k] > L[ks]) ks = k;
    var bs = ks / K;
    var best = new Float64Array(T + 1);
    for (t = 0; t < T; t++) {
      best[t + 1] = best[t] + Math.log(bs * x1[t] + (1 - bs) * x2[t]);
    }
    return { T: T, L: L, bStar: bs, logStar: L[ks], up: up, a1: a1, a2: a2,
             best: best };
  }

  function fmtWealth(logw) {
    var w = Math.exp(logw);
    if (w >= 1e5 || w < 1e-3) return w.toExponential(1);
    if (w >= 100) return w.toFixed(0);
    return w.toPrecision(3);
  }

  // A few "nice" wealth levels for a log axis spanning [lo, hi] (natural log).
  function logTicks(lo, hi) {
    var cands = [];
    var e0 = Math.floor(lo / Math.LN10) - 1, e1 = Math.ceil(hi / Math.LN10) + 1;
    for (var e = e0; e <= e1; e++) {
      [1, 2, 5].forEach(function (m) { cands.push(m * Math.pow(10, e)); });
    }
    var inRange = cands.filter(function (v) {
      var l = Math.log(v); return l >= lo && l <= hi;
    });
    if (inRange.length > 6) {
      inRange = inRange.filter(function (v) {
        var m = v / Math.pow(10, Math.floor(Math.log10(v) + 1e-9));
        return Math.abs(m - 1) < 1e-6;
      });
    }
    if (inRange.length < 2) {
      // narrow range: fall back to a few evenly spaced levels
      inRange = [];
      for (var i = 0; i <= 3; i++) inRange.push(Math.exp(lo + (hi - lo) * i / 3));
    }
    return inRange;
  }

  function tickLabel(v) {
    if (v >= 1e5 || v < 1e-2) return v.toExponential(0);
    if (v >= 10) return v.toFixed(0);
    return String(Number(v.toPrecision(2)));
  }

  function build(mount) {
    // A market where the universal portfolio ends ahead of both stocks. Many
    // seeds don't: at sigma = 0.06 and T = 500 the stocks' random walks
    // (sigma * sqrt(T) = 1.3 in log-wealth) are larger than the rebalancing
    // bonus (T sigma^2 / 4 = 0.45), so "New market" shows both outcomes.
    var seed = 3691838464;
    var z = shocks(seed);
    var sigma = 0.06, T = 500;
    var res = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent =
      "Universal portfolio vs. all CRPs";
    var meta = el("div", "widget-meta", head);
    mount.parentNode.insertBefore(head, mount);

    var curve = el("canvas", "widget-plot", mount);
    curve.style.cursor = "default";
    curve.setAttribute("aria-label",
      "Final wealth of each constant-rebalanced portfolio b, with the best one and the universal portfolio marked");
    var traj = el("canvas", "widget-plot", mount);
    traj.style.cursor = "default";
    traj.style.marginTop = "0.9rem";
    traj.setAttribute("aria-label",
      "Wealth over time of both stocks, the best constant-rebalanced portfolio and the universal portfolio");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "Market";

    function slider(label, lo, hi, step, initial, fmt, onChange) {
      var row = el("div", "widget-slider-row", left);
      el("span", null, row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range";
      input.min = String(lo); input.max = String(hi); input.step = String(step);
      input.value = String(initial);
      var val = el("span", "widget-slider-val", row);
      val.textContent = fmt(initial);
      input.addEventListener("input", function () {
        var v = Number(input.value);
        val.textContent = fmt(v);
        onChange(v);
      });
    }

    slider("volatility σ", SIG_MIN, SIG_MAX, 0.005, sigma,
      function (v) { return v.toFixed(3); },
      function (v) { sigma = v; refresh(); });
    slider("periods T", 20, T_MAX, 10, T,
      function (v) { return String(v); },
      function (v) { T = v; refresh(); });

    el("div", "widget-label", right).textContent = "After T periods";

    function metric(name) {
      var row = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", row).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", row);
      val.style.width = "4.6em";
      return function (frac, text) {
        val.textContent = text;
        fill.style.width =
          (100 * Math.max(0, Math.min(1, frac))).toFixed(1) + "%";
      };
    }

    var setB = metric("b*");
    var setUp = metric("Ŝ / S*");
    var setAsset = metric("asset / S*");
    var setReg = metric("regret");

    var note = el("div", "widget-note", mount);
    var controls = el("div", "widget-controls", mount);
    var btn = el("button", "widget-btn", controls);
    btn.type = "button";
    btn.textContent = "New market";
    btn.setAttribute("data-primary", "");
    btn.addEventListener("click", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      z = shocks(seed);
      refresh();
    });

    function sizeCanvas(c, ratio, minH, maxH) {
      var W = c.getBoundingClientRect().width || mount.clientWidth || 520;
      var H = Math.max(minH, Math.min(maxH, Math.round(W * ratio)));
      var dpr = window.devicePixelRatio || 1;
      c.style.height = H + "px";
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
      var ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      return { ctx: ctx, W: W, H: H };
    }

    function colors() {
      return {
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        border: cssVar(mount, "--border", "#e2e2dd"),
        accent: cssVar(mount, "--accent", "#ba3925"),
        link: cssVar(mount, "--link", "#2156a5"),
      };
    }

    // Shared frame: log-wealth y axis with wealth-valued tick labels.
    function frame(g, lo, hi, padL, padR, padT, padB, col) {
      var ctx = g.ctx;
      var py = function (v) {
        return g.H - padB - (g.H - padT - padB) * (v - lo) / (hi - lo);
      };
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      logTicks(lo, hi).forEach(function (v) {
        var y = py(Math.log(v));
        ctx.strokeStyle = rgba(col.border, 1);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, Math.round(y) + 0.5);
        ctx.lineTo(g.W - padR, Math.round(y) + 0.5);
        ctx.stroke();
        ctx.fillStyle = col.muted;
        ctx.fillText(tickLabel(v), padL - 5, y);
      });
      return py;
    }

    function drawCurve(col) {
      var g = sizeCanvas(curve, 0.36, 170, 240);
      var ctx = g.ctx;
      var padL = 44, padR = 12, padT = 24, padB = 22;
      var lo = Infinity, hi = -Infinity;
      for (var k = 0; k <= K; k++) {
        lo = Math.min(lo, res.L[k]); hi = Math.max(hi, res.L[k]);
      }
      lo = Math.min(lo, res.up[res.T], 0); hi = Math.max(hi, 0);
      var pad = 0.1 * (hi - lo || 1);
      lo -= pad; hi += pad;
      var py = frame(g, lo, hi, padL, padR, padT, padB, col);
      var px = function (b) { return padL + (g.W - padL - padR) * b; };

      // x axis labels
      ctx.fillStyle = col.muted;
      ctx.textBaseline = "top";
      ctx.font = "10px 'JetBrains Mono', monospace";
      [0, 0.25, 0.5, 0.75, 1].forEach(function (b) {
        ctx.textAlign = b === 0 ? "left" : b === 1 ? "right" : "center";
        ctx.fillText(b === 0 ? "b=0 (all stock 2)" : b === 1 ? "b=1 (all stock 1)"
          : String(b), px(b), g.H - padB + 6);
      });

      // area under the curve, faint: its average height is the universal wealth
      ctx.beginPath();
      ctx.moveTo(px(0), py(lo));
      for (k = 0; k <= K; k++) ctx.lineTo(px(k / K), py(res.L[k]));
      ctx.lineTo(px(1), py(lo));
      ctx.closePath();
      ctx.fillStyle = rgba(col.accent, 0.08);
      ctx.fill();

      ctx.strokeStyle = col.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (k = 0; k <= K; k++) ctx.lineTo(px(k / K), py(res.L[k]));
      ctx.stroke();

      // universal wealth level
      var yu = py(res.up[res.T]);
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = col.link;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, yu); ctx.lineTo(g.W - padR, yu);
      ctx.stroke();

      // b* marker
      var xs = px(res.bStar), ys = py(res.logStar);
      ctx.strokeStyle = col.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xs, ys); ctx.lineTo(xs, g.H - padB);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col.accent;
      ctx.beginPath(); ctx.arc(xs, ys, 4, 0, 6.284); ctx.fill();

      // endpoints = the two buy-and-hold stocks
      ctx.fillStyle = col.fg;
      [0, K].forEach(function (k2) {
        ctx.beginPath(); ctx.arc(px(k2 / K), py(res.L[k2]), 3, 0, 6.284); ctx.fill();
      });

      ctx.font = "11px 'Open Sans', sans-serif";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = col.accent;
      ctx.textAlign = res.bStar > 0.8 ? "right" : "left";
      ctx.fillText("b* = " + res.bStar.toFixed(3) + "  (S* = " +
        fmtWealth(res.logStar) + ")", xs + (res.bStar > 0.8 ? -7 : 7), ys - 3);
      ctx.fillStyle = col.link;
      ctx.textAlign = "left";
      var labelY = Math.abs(yu - ys) < 16 ? yu + 14 : yu - 3;
      ctx.fillText("universal Ŝ = " + fmtWealth(res.up[res.T]),
        padL + 6, labelY);
      ctx.fillStyle = col.muted;
      ctx.textBaseline = "top";
      ctx.fillText("final wealth S_T(b) of each constant-rebalanced portfolio",
        padL, 4);
    }

    function drawTraj(col) {
      var g = sizeCanvas(traj, 0.3, 150, 210);
      var ctx = g.ctx;
      var padL = 44, padR = 12, padT = 24, padB = 10;
      var series = [
        [res.a1, col.muted, [], "stock 1"],
        [res.a2, col.muted, [4, 3], "stock 2"],
        [res.best, col.accent, [], "best CRP b*"],
        [res.up, col.link, [], "universal"],
      ];
      var lo = 0, hi = 0, t, s;
      for (s = 0; s < series.length; s++) {
        for (t = 0; t <= res.T; t++) {
          lo = Math.min(lo, series[s][0][t]); hi = Math.max(hi, series[s][0][t]);
        }
      }
      var pad = 0.08 * (hi - lo || 1);
      lo -= pad; hi += pad;
      var py = frame(g, lo, hi, padL, padR, padT, padB, col);
      var px = function (i) { return padL + (g.W - padL - padR) * i / res.T; };
      for (s = 0; s < series.length; s++) {
        ctx.strokeStyle = series[s][1];
        ctx.setLineDash(series[s][2]);
        ctx.lineWidth = s >= 2 ? 2 : 1.4;
        ctx.beginPath();
        for (t = 0; t <= res.T; t++) ctx.lineTo(px(t), py(series[s][0][t]));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.font = "11px 'Open Sans', sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      var lx = padL;
      for (s = 0; s < series.length; s++) {
        ctx.strokeStyle = series[s][1];
        ctx.setLineDash(series[s][2]);
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lx, 10); ctx.lineTo(lx + 14, 10); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = col.muted;
        ctx.fillText(series[s][3], lx + 19, 10);
        lx += 19 + ctx.measureText(series[s][3]).width + 14;
      }
      ctx.textAlign = "right";
      ctx.fillText("wealth over time, t = 0 … " + res.T, g.W - padR, 10);
    }

    function draw() {
      var col = colors();
      drawCurve(col);
      drawTraj(col);
    }

    function refresh() {
      res = simulate(z, sigma, T);
      draw();
      var bound = Math.log(T + 1);
      var regret = res.logStar - res.up[T];
      var logAsset = Math.max(res.a1[T], res.a2[T]);
      meta.textContent = "seed " + seed + " · two stocks · uniform prior";
      setB(res.bStar, res.bStar.toFixed(3));
      setUp(Math.exp(res.up[T] - res.logStar),
        Math.exp(res.up[T] - res.logStar).toFixed(3));
      setAsset(Math.exp(logAsset - res.logStar),
        Math.exp(logAsset - res.logStar).toFixed(3));
      setReg(regret / bound, regret.toFixed(2));

      var msg = "Regret " + regret.toFixed(2) + " nats against a guaranteed " +
        "ceiling of " + bound.toFixed(2) + "; per period that is " +
        (regret / T).toExponential(1) + ". ";
      if (res.up[T] > logAsset) {
        msg += "The universal portfolio ends ahead of both stocks (" +
          fmtWealth(res.up[T]) + " vs. " + fmtWealth(res.a1[T]) + " and " +
          fmtWealth(res.a2[T]) + "), without ever looking ahead.";
      } else {
        msg += "Here the better stock (" + fmtWealth(logAsset) + ") beats the " +
          "universal portfolio (" + fmtWealth(res.up[T]) + "): the guarantee " +
          "is against the best CRP in growth rate, not a promise to win every " +
          "finite race.";
      }
      if (res.bStar === 0 || res.bStar === 1) {
        msg += " b* sits on the boundary, the worst case for the uniform prior.";
      }
      note.textContent = msg;
    }

    refresh();
    var pending = false;
    window.addEventListener("resize", function () {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    });
    new MutationObserver(draw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="universal-portfolio"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

// Mixture-of-Experts routing: a softmax gate sends a batch of tokens to their
// top-k experts, each expert has a fixed capacity, overflow is dropped, and the
// Switch Transformer load-balancing loss measures how uneven it all is.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-moe-routing
//   :::
//
// The computation is the real one:
//   p_t      = softmax((W x_t + b) / tau)               router probabilities
//   experts  = top-k of p_t                             per token
//   C        = floor(capacity_factor * k * T / E)       slots per expert
//   dispatch = all first choices in token order, then all second choices, ...;
//              an assignment that finds its expert full is dropped
//   L_aux    = E * sum_i f_i * P_i                      (Fedus et al., 2021)
// where f_i is the fraction of the T*k assignments that chose expert i and P_i
// is the mean router probability for i. L_aux = 1 when perfectly balanced.
// "Train router" runs plain gradient descent on L_aux alone with respect to W
// and b (f_i is treated as a constant, as in the paper; the gradient was
// checked against finite differences). Offline, from the default skewed start
// (E = 8, k = 2, tau = 1, cf = 1.25) L_aux falls from 2.31 to 1.00 within 20
// steps and dropped assignments from 62 of 128 to 0. With a sharp, heavily
// skewed router (E = 16, tau = 0.5, skew 2) it stalls near 1.44: experts whose
// probability is already ~0 get almost no gradient, i.e. they are dead.
(function () {
  "use strict";

  var T = 64;                  // tokens in the batch
  var D = 8;                   // token / router dimension
  var CLUSTERS = 4;            // tokens are drawn around a few "topics"
  var LR = 0.5;
  var HIST = 120;
  var STEP_EVERY = 3;          // frames between training steps (to be watchable)

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

  function hexRgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (isNaN(n)) return [255, 255, 255];
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function rgba(hex, a) {
    var c = hexRgb(hex);
    return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rnd) {
    var u = 0;
    while (u === 0) u = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  }

  function makeTokens(seed) {
    var rnd = mulberry32(seed);
    var centers = [];
    for (var c = 0; c < CLUSTERS; c++) {
      var v = [];
      for (var d = 0; d < D; d++) v.push(gauss(rnd));
      centers.push(v);
    }
    var X = [];
    for (var t = 0; t < T; t++) {
      var cc = centers[Math.floor(rnd() * CLUSTERS)];
      var x = [];
      for (var d2 = 0; d2 < D; d2++) x.push(cc[d2] * 0.8 + 0.6 * gauss(rnd));
      X.push(x);
    }
    return X;
  }

  // Router init: small random W, plus a bias that favours low-index experts by
  // `skew` -- a stand-in for a router that has drifted towards a few experts.
  function makeRouter(E, skew, seed) {
    var rnd = mulberry32(seed * 31 + E);
    var W = [], b = [];
    for (var e = 0; e < E; e++) {
      var w = [];
      for (var d = 0; d < D; d++) w.push(gauss(rnd) * 0.4);
      W.push(w);
      b.push(E > 1 ? skew * (1 - 2 * e / (E - 1)) : 0);
    }
    return { W: W, b: b };
  }

  function route(X, R, E, k, tau, cf) {
    var P = [], top = [], t, e;
    for (t = 0; t < T; t++) {
      var z = [], mx = -Infinity;
      for (e = 0; e < E; e++) {
        var s = R.b[e];
        for (var d = 0; d < D; d++) s += R.W[e][d] * X[t][d];
        s /= tau;
        z.push(s);
        if (s > mx) mx = s;
      }
      var Z = 0;
      for (e = 0; e < E; e++) { z[e] = Math.exp(z[e] - mx); Z += z[e]; }
      for (e = 0; e < E; e++) z[e] /= Z;
      P.push(z);
      var idx = [];
      for (e = 0; e < E; e++) idx.push(e);
      idx.sort(function (a, c) { return z[c] - z[a]; });
      top.push(idx.slice(0, k));
    }

    var cap = Math.floor(cf * k * T / E);
    var load = [], kept = [], slots = [];
    for (e = 0; e < E; e++) { load.push(0); kept.push(0); slots.push([]); }
    var dropped = 0, lost = [];
    for (t = 0; t < T; t++) lost.push(0);
    // priority: every token's first choice (in token order), then second, ...
    for (var j = 0; j < k; j++) {
      for (t = 0; t < T; t++) {
        e = top[t][j];
        load[e]++;
        var ok = kept[e] < cap;
        if (ok) kept[e]++; else { dropped++; lost[t]++; }
        slots[e].push({ t: t, rank: j, ok: ok, g: P[t][e] });
      }
    }
    var f = [], Pm = [], aux = 0, top1 = 0;
    for (e = 0; e < E; e++) {
      f.push(load[e] / (T * k));
      var m = 0;
      for (t = 0; t < T; t++) m += P[t][e];
      Pm.push(m / T);
      aux += E * f[e] * Pm[e];
    }
    for (t = 0; t < T; t++) top1 += P[t][top[t][0]] / T;
    var fully = 0;
    for (t = 0; t < T; t++) if (lost[t] === k) fully++;
    var maxLoad = 0;
    for (e = 0; e < E; e++) maxLoad = Math.max(maxLoad, load[e]);
    return {
      P: P, top: top, cap: cap, load: load, slots: slots, dropped: dropped,
      lost: lost, fully: fully, f: f, Pm: Pm, aux: aux, top1: top1,
      maxLoad: maxLoad, E: E, k: k,
    };
  }

  // One gradient step on L_aux, f held fixed:
  //   dL/dz_tj = (E/T) p_tj (f_j - sum_i f_i p_ti),   z = (W x + b) / tau
  function trainStep(X, R, r, tau) {
    var E = r.E;
    var gW = [], gb = [], e, d;
    for (e = 0; e < E; e++) { gb.push(0); gW.push(new Array(D).fill(0)); }
    for (var t = 0; t < T; t++) {
      var p = r.P[t], s = 0;
      for (e = 0; e < E; e++) s += r.f[e] * p[e];
      for (e = 0; e < E; e++) {
        var dz = (E / T) * p[e] * (r.f[e] - s) / tau;
        gb[e] += dz;
        for (d = 0; d < D; d++) gW[e][d] += dz * X[t][d];
      }
    }
    for (e = 0; e < E; e++) {
      R.b[e] -= LR * gb[e];
      for (d = 0; d < D; d++) R.W[e][d] -= LR * gW[e][d];
    }
  }

  function build(mount) {
    var cfg = { E: 8, k: 2, tau: 1, skew: 1.5, cf: 1.25 };
    var seed = 1, batchSeed = 1;
    var X = makeTokens(batchSeed);
    var R = makeRouter(cfg.E, cfg.skew, seed);
    var res = null, hist = [], steps = 0;
    var training = false, raf = null, frameCount = 0, hover = -1;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Top-k routing, capacity and load balancing";
    el("div", "widget-meta", head).textContent = T + " tokens · softmax router";
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Tokens on top are routed to expert columns below; dashed line is each expert's capacity");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "Router";
    function slider(label, min, max, step, key, fmt, resetRouter) {
      var row = el("div", "widget-slider-row", left);
      el("span", null, row).textContent = label;
      var s = el("input", "widget-slider", row);
      s.type = "range";
      s.min = String(min); s.max = String(max); s.step = String(step);
      s.value = String(cfg[key]);
      s.setAttribute("aria-label", label);
      var val = el("span", "widget-slider-val", row);
      val.textContent = fmt(cfg[key]);
      s.addEventListener("input", function () {
        cfg[key] = Number(s.value);
        val.textContent = fmt(cfg[key]);
        if (key === "E") {
          kIn.max = String(Math.min(4, cfg.E));
          if (cfg.k > cfg.E) { cfg.k = cfg.E; kIn.value = String(cfg.k); kIn.dispatchEvent(new Event("sync")); }
        }
        if (resetRouter) newRouter(); else recompute();
      });
      s.addEventListener("sync", function () { val.textContent = fmt(cfg[key]); });
      return s;
    }
    slider("experts E", 2, 16, 1, "E", String, true);
    var kIn = slider("top-k", 1, 4, 1, "k", String, false);
    slider("temperature", 0.25, 3, 0.05, "tau", function (v) { return v.toFixed(2); }, false);
    slider("router skew", 0, 3, 0.1, "skew", function (v) { return v.toFixed(1); }, true);
    slider("capacity ×", 0.5, 2.5, 0.05, "cf", function (v) { return v.toFixed(2); }, false);

    el("div", "widget-label", right).textContent = "Balance";
    function metric(name) {
      var r = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", r).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", r);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      var val = el("span", "widget-row-val", r);
      val.style.width = "4.2em";
      return function (frac, text, bad) {
        fill.style.width = (100 * Math.max(0, Math.min(1, frac))).toFixed(1) + "%";
        if (bad) fill.setAttribute("data-pos", ""); else fill.removeAttribute("data-pos");
        val.textContent = text;
      };
    }
    var mAux = metric("aux loss");
    var mDrop = metric("dropped");
    var mFull = metric("tokens lost");
    var mMax = metric("max load/C");
    var mTop1 = metric("top-1 gate");
    var spark = el("canvas", "widget-spark", right);

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
    var trainBtn = button("Train router (aux loss)", function () {
      if (training) stopTraining(); else startTraining();
    }, true);
    button("Reset router", function () { newRouter(); });
    button("New batch", function () {
      batchSeed = (batchSeed * 1103515245 + 12345) >>> 0;
      X = makeTokens(batchSeed);
      recompute();
    });

    function newRouter() {
      stopTraining();
      R = makeRouter(cfg.E, cfg.skew, seed);
      hist = []; steps = 0;
      recompute();
    }

    function recompute() {
      res = route(X, R, cfg.E, cfg.k, cfg.tau, cfg.cf);
      if (training || !hist.length) {
        hist.push(res.aux);
        if (hist.length > HIST) hist.shift();
      }
      draw();
    }

    function startTraining() {
      training = true;
      trainBtn.textContent = "Stop training";
      raf = requestAnimationFrame(tick);
    }
    function stopTraining() {
      training = false;
      trainBtn.textContent = "Train router (aux loss)";
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      if (res) draw();
    }
    function tick() {
      raf = null;
      if (!training) return;
      if (++frameCount % STEP_EVERY === 0) {
        trainStep(X, R, res, cfg.tau);
        steps++;
        recompute();
      }
      raf = requestAnimationFrame(tick);
    }

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var sctx = spark.getContext("2d");
    var geom = null;

    function palette() {
      var bg = cssVar(mount, "--card-bg", "#fff");
      var c = hexRgb(bg);
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        link: cssVar(mount, "--link", "#2156a5"),
        border: cssVar(mount, "--border", "#e2e2dd"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        block: cssVar(mount, "--block-bg", "#f3f3f2"),
        bg: bg,
        dark: (c[0] + c[1] + c[2]) / 3 < 110,
      };
    }

    // Categorical expert hues, lighter on the dark theme.
    function expertColor(e, E, pal, a) {
      var hue = (210 + e * 360 / E) % 360;
      return "hsla(" + hue.toFixed(0) + "," + (pal.dark ? 65 : 55) + "%,"
        + (pal.dark ? 66 : 46) + "%," + a + ")";
    }

    function draw() {
      var pal = palette();
      var w = mount.clientWidth || 600;
      var narrow = w < 520;
      var perRow = narrow ? 32 : 64;
      var tokRows = Math.ceil(T / perRow);
      var cell = Math.min(14, (w - 8) / perRow);
      var tokH = tokRows * cell;
      var linkH = narrow ? 46 : 64;
      var colH = narrow ? 170 : 190;
      var labelH = 30;
      var H = tokH + linkH + colH + labelH + 8;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, H);

      var E = res.E;
      var tokX0 = (w - perRow * cell) / 2;
      function tokPos(t) {
        var r = Math.floor(t / perRow), c = t % perRow;
        return { x: tokX0 + c * cell, y: r * cell };
      }
      var colGap = Math.max(4, Math.min(14, w / E * 0.18));
      var colW = (w - colGap * (E + 1)) / E;
      var colTop = tokH + linkH;
      var colBase = colTop + colH;
      function colX(e) { return colGap + e * (colW + colGap); }
      var maxUnits = Math.max(res.maxLoad, res.cap * 1.2, 1);
      var unit = Math.min(10, colH / maxUnits);
      geom = { tokPos: tokPos, cell: cell, tokH: tokH };

      // links: token -> each chosen expert
      for (var t = 0; t < T; t++) {
        var tp = tokPos(t);
        for (var j = 0; j < res.k; j++) {
          var e = res.top[t][j];
          var ok = j < res.k && res.slots[e].some(function (s) {
            return s.t === t && s.rank === j && s.ok;
          });
          var hl = hover === t;
          var g = res.P[t][e];
          ctx.strokeStyle = ok
            ? expertColor(e, E, pal, hl ? 0.95 : 0.08 + 0.35 * g)
            : rgba(pal.accent, hl ? 0.95 : 0.35);
          ctx.lineWidth = hl ? 2 : 1;
          ctx.setLineDash(ok ? [] : [3, 3]);
          ctx.beginPath();
          ctx.moveTo(tp.x + cell / 2, tp.y + cell);
          ctx.lineTo(colX(e) + colW / 2, colTop - 2);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // tokens: coloured by first-choice expert, struck through if all k
      // assignments were dropped
      for (var t2 = 0; t2 < T; t2++) {
        var p2 = tokPos(t2);
        var lostAll = res.lost[t2] === res.k;
        ctx.fillStyle = lostAll ? pal.block
          : expertColor(res.top[t2][0], E, pal, 0.9);
        ctx.fillRect(p2.x + 1, p2.y + 1, cell - 2, cell - 2);
        if (lostAll || res.lost[t2] > 0) {
          ctx.strokeStyle = pal.accent;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(p2.x + 1.5, p2.y + 1.5, cell - 3, cell - 3);
        }
        if (lostAll) {
          ctx.beginPath();
          ctx.moveTo(p2.x + 3, p2.y + 3);
          ctx.lineTo(p2.x + cell - 3, p2.y + cell - 3);
          ctx.stroke();
        }
        if (hover === t2) {
          ctx.strokeStyle = pal.fg;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(p2.x + 0.5, p2.y + 0.5, cell - 1, cell - 1);
        }
      }

      // expert columns: one block per assignment, kept ones solid, overflow
      // hollow above the capacity line
      for (var e2 = 0; e2 < E; e2++) {
        var x = colX(e2);
        ctx.fillStyle = pal.block;
        ctx.fillRect(x, colTop, colW, colH);
        res.slots[e2].forEach(function (s, i) {
          var y = colBase - (i + 1) * unit;
          var hl = hover === s.t;
          if (s.ok) {
            ctx.fillStyle = expertColor(e2, E, pal, hl ? 1 : 0.35 + 0.6 * s.g);
            ctx.fillRect(x + 1, y + 0.5, colW - 2, Math.max(1, unit - 1));
          } else {
            ctx.strokeStyle = rgba(pal.accent, hl ? 1 : 0.8);
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 1.5, y + 1, colW - 3, Math.max(1, unit - 2));
          }
        });
        var over = res.load[e2] - Math.min(res.load[e2], res.cap);
        ctx.fillStyle = over ? pal.accent : pal.muted;
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(String(res.load[e2]), x + colW / 2, colBase + 4);
        ctx.fillStyle = pal.muted;
        if (colW > 16) ctx.fillText("E" + (e2 + 1), x + colW / 2, colBase + 16);
      }
      // capacity line
      var capY = colBase - res.cap * unit;
      ctx.strokeStyle = pal.fg;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(4, capY); ctx.lineTo(w - 4, capY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = pal.fg;
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillText("capacity C = " + res.cap, w - 6, capY - 2);

      drawSpark(pal);
      readout();
    }

    function drawSpark(pal) {
      var w = right.clientWidth || 260, h = 44;
      var dpr = window.devicePixelRatio || 1;
      spark.width = Math.round(w * dpr);
      spark.height = Math.round(h * dpr);
      sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      sctx.clearRect(0, 0, w, h);
      var hi = Math.max(1.5, Math.max.apply(null, hist));
      var y = function (v) { return h - 4 - (h - 10) * (v - 1) / (hi - 1); };
      sctx.strokeStyle = pal.border;
      sctx.setLineDash([2, 3]);
      sctx.beginPath();
      sctx.moveTo(0, y(1)); sctx.lineTo(w, y(1));
      sctx.stroke();
      sctx.setLineDash([]);
      if (hist.length < 2) return;
      sctx.strokeStyle = pal.accent;
      sctx.lineWidth = 1.75;
      sctx.beginPath();
      for (var i = 0; i < hist.length; i++) {
        var px = (w - 4) * i / (HIST - 1) + 2;
        if (i === 0) sctx.moveTo(px, y(hist[i])); else sctx.lineTo(px, y(hist[i]));
      }
      sctx.stroke();
    }

    function readout() {
      var E = res.E, A = T * res.k;
      mAux((res.aux - 1) / Math.max(1, E - 1), res.aux.toFixed(3), res.aux > 1.1);
      mDrop(res.dropped / A, res.dropped + "/" + A, res.dropped > 0);
      mFull(res.fully / T, res.fully + "/" + T, res.fully > 0);
      mMax(Math.min(1, res.maxLoad / Math.max(1, res.cap) / 3),
        (res.maxLoad / Math.max(1, res.cap)).toFixed(2), res.maxLoad > res.cap);
      mTop1(res.top1, res.top1.toFixed(2), false);

      if (training) {
        note.textContent = "Gradient step " + steps + " on L_aux alone. Probability "
          + "mass is pushed away from overloaded experts, so their tokens "
          + "spill to idle ones. Watch the dashed (dropped) assignments vanish "
          + "— and the top-1 gate weight fall: with no task loss to oppose it, "
          + "the aux loss is happy with a uniform, uninformative router. Real "
          + "models scale it by α ≈ 0.01.";
      } else if (res.aux > 1.15) {
        note.textContent = "Unbalanced: L_aux = " + res.aux.toFixed(2)
          + " (1.00 is perfect balance, " + E + " is everything on one expert). "
          + res.dropped + " assignments overflowed their expert's capacity and "
          + "were dropped" + (res.fully ? ", and " + res.fully + " tokens lost "
          + "every expert and pass through on the residual alone" : "")
          + ". Press Train router, or raise the capacity factor.";
      } else if (res.dropped > 0) {
        note.textContent = "Close to balanced (L_aux = " + res.aux.toFixed(2)
          + "), but " + res.dropped + " assignments still overflow. Capacity "
          + "is a hard cap: any expert above the dashed line drops tokens.";
      } else {
        note.textContent = "Balanced: L_aux = " + res.aux.toFixed(2)
          + " and every assignment fits under capacity. Hover a token to see "
          + "where it goes.";
      }
    }

    canvas.addEventListener("pointermove", function (ev) {
      if (!geom) return;
      var r = canvas.getBoundingClientRect();
      var mx = ev.clientX - r.left, my = ev.clientY - r.top;
      var h = -1;
      if (my >= 0 && my < geom.tokH) {
        for (var t = 0; t < T; t++) {
          var p = geom.tokPos(t);
          if (mx >= p.x && mx < p.x + geom.cell && my >= p.y && my < p.y + geom.cell) {
            h = t; break;
          }
        }
      }
      if (h !== hover) { hover = h; draw(); }
    });
    canvas.addEventListener("pointerleave", function () {
      if (hover !== -1) { hover = -1; draw(); }
    });

    recompute();
    window.addEventListener("resize", draw);
    new MutationObserver(draw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="moe-routing"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

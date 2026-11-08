// Lottery tickets, found live: iterative magnitude pruning with rewinding on a
// tiny MLP, against the same masks with a fresh random initialisation.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-lottery-ticket
//   :::
//
// Everything is computed in the page. A 2-24-24-1 ReLU MLP is trained
// full-batch with Adam on a two-arm spiral. Then, each round:
//   1. prune p% of the surviving weights in each layer by magnitude (output
//      layer at p/2, as in Frankle & Carbin), using the trained ticket;
//   2. rewind the survivors to their ORIGINAL initial values theta_0 and retrain
//      (the "winning ticket");
//   3. train the same mask from a FRESH random init (the control).
// Training runs in time-boxed requestAnimationFrame chunks so the page never
// blocks.
//
// Checked offline by running this file's own Net/prune/data code (node, 12
// seeds, p = 30%, 8 rounds). Mean test accuracy, ticket / reinit, and how often
// the ticket won:
//   25.0% weights  0.995 / 0.985   8/12
//   17.7%          0.973 / 0.986   7/12   (no real difference yet)
//   12.7%          0.929 / 0.846  10/12
//    9.1%          0.821 / 0.721   9/12
//    6.6%          0.743 / 0.647   8/12
// So the lottery-ticket gap is real but only shows up below ~15% of weights,
// and any single run can go the other way. The default seed (12) is one where
// the gap is clean; "New seed" shows the spread.
(function () {
  "use strict";

  var H = 24;                  // hidden width (two hidden layers)
  var STEPS = 1500;            // full-batch Adam steps per training run
  var LR = 0.01;
  var NTRAIN = 200, NTEST = 1000;
  var NOISE = 0.05, TURNS = 1;
  var MIN_DENSITY = 0.05;      // stop pruning before the nets are all dead
  var FRAME_MS = 10;           // compute budget per animation frame
  var MAX_STEPS_PER_FRAME = 60;
  var GRID = 56;               // decision-boundary resolution
  var EXT = 1.35;              // plot extent in data units

  var LAYERS = [               // name, length, fan-in (0 = bias)
    ["W1", H * 2, 2], ["b1", H, 0],
    ["W2", H * H, H], ["b2", H, 0],
    ["W3", H, H], ["b3", 1, 0],
  ];
  var WEIGHTS = ["W1", "W2", "W3"];
  var TOTAL_W = H * 2 + H * H + H;

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
    if (isNaN(n)) return [128, 128, 128];
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

  // Two interleaved spiral arms.
  function makeData(n, rnd) {
    var X = new Float64Array(n * 2), Y = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var c = i % 2, t = rnd();
      var r = 0.15 + 0.85 * t, a = t * TURNS * 2 * Math.PI + c * Math.PI;
      X[2 * i] = r * Math.cos(a) + NOISE * gauss(rnd);
      X[2 * i + 1] = r * Math.sin(a) + NOISE * gauss(rnd);
      Y[i] = c;
    }
    return { X: X, Y: Y, n: n };
  }

  // He-normal init for weights, zero biases.
  function initParams(rnd) {
    var p = {};
    LAYERS.forEach(function (L) {
      var a = new Float64Array(L[1]);
      if (L[2]) {
        var s = Math.sqrt(2 / L[2]);
        for (var i = 0; i < a.length; i++) a[i] = gauss(rnd) * s;
      }
      p[L[0]] = a;
    });
    return p;
  }

  function cloneParams(p) {
    var q = {};
    for (var k in p) q[k] = Float64Array.from(p[k]);
    return q;
  }

  function fullMask() {
    var m = {};
    WEIGHTS.forEach(function (k) {
      m[k] = new Float64Array(LAYERS.filter(function (L) {
        return L[0] === k;
      })[0][1]).fill(1);
    });
    return m;
  }

  function density(mask) {
    var a = 0;
    WEIGHTS.forEach(function (k) {
      for (var i = 0; i < mask[k].length; i++) a += mask[k][i];
    });
    return a / TOTAL_W;
  }

  // Layer-wise magnitude pruning of the surviving weights.
  function prune(p, mask, rate) {
    var next = {};
    WEIGHTS.forEach(function (k) {
      var r = k === "W3" ? rate / 2 : rate;
      var alive = [];
      for (var i = 0; i < p[k].length; i++) {
        if (mask[k][i]) alive.push([Math.abs(p[k][i]), i]);
      }
      alive.sort(function (a, b) { return a[0] - b[0]; });
      var n = Math.round(alive.length * r);
      next[k] = Float64Array.from(mask[k]);
      for (var j = 0; j < n; j++) next[k][alive[j][1]] = 0;
    });
    return next;
  }

  // A network being trained: params, mask, Adam state.
  function Net(params, mask) {
    this.p = params;
    this.mask = mask;
    this.m1 = {}; this.m2 = {}; this.g = {};
    for (var k in params) {
      this.m1[k] = new Float64Array(params[k].length);
      this.m2[k] = new Float64Array(params[k].length);
      this.g[k] = new Float64Array(params[k].length);
      if (mask[k]) {
        for (var i = 0; i < params[k].length; i++) params[k][i] *= mask[k][i];
      }
    }
    this.t = 0;
    this.a1 = new Float64Array(H);
    this.a2 = new Float64Array(H);
    this.d1 = new Float64Array(H);
    this.d2 = new Float64Array(H);
  }

  Net.prototype.forward = function (x0, x1) {
    var p = this.p, a1 = this.a1, a2 = this.a2, j, k;
    for (j = 0; j < H; j++) {
      var z1 = p.W1[2 * j] * x0 + p.W1[2 * j + 1] * x1 + p.b1[j];
      a1[j] = z1 > 0 ? z1 : 0;
    }
    for (j = 0; j < H; j++) {
      var z2 = p.b2[j], o = j * H;
      for (k = 0; k < H; k++) z2 += p.W2[o + k] * a1[k];
      a2[j] = z2 > 0 ? z2 : 0;
    }
    var z = p.b3[0];
    for (k = 0; k < H; k++) z += p.W3[k] * a2[k];
    return z;
  };

  Net.prototype.step = function (data) {
    var p = this.p, g = this.g, a1 = this.a1, a2 = this.a2;
    var d1 = this.d1, d2 = this.d2, k, j, i;
    for (k in g) g[k].fill(0);
    for (i = 0; i < data.n; i++) {
      var x0 = data.X[2 * i], x1 = data.X[2 * i + 1];
      var z = this.forward(x0, x1);
      var dz = (1 / (1 + Math.exp(-z)) - data.Y[i]) / data.n;  // BCE grad
      g.b3[0] += dz;
      for (k = 0; k < H; k++) {
        g.W3[k] += dz * a2[k];
        d2[k] = a2[k] > 0 ? dz * p.W3[k] : 0;
      }
      d1.fill(0);
      for (j = 0; j < H; j++) {
        var dj = d2[j];
        if (dj === 0) continue;
        g.b2[j] += dj;
        var o = j * H;
        for (k = 0; k < H; k++) {
          g.W2[o + k] += dj * a1[k];
          d1[k] += dj * p.W2[o + k];
        }
      }
      for (j = 0; j < H; j++) {
        if (a1[j] <= 0) continue;
        g.b1[j] += d1[j];
        g.W1[2 * j] += d1[j] * x0;
        g.W1[2 * j + 1] += d1[j] * x1;
      }
    }
    this.t++;
    var c1 = 1 - Math.pow(0.9, this.t), c2 = 1 - Math.pow(0.999, this.t);
    for (k in p) {
      var pk = p[k], gk = g[k], m1 = this.m1[k], m2 = this.m2[k];
      var mk = this.mask[k];
      for (i = 0; i < pk.length; i++) {
        if (mk && mk[i] === 0) continue;       // pruned weights stay at zero
        m1[i] = 0.9 * m1[i] + 0.1 * gk[i];
        m2[i] = 0.999 * m2[i] + 0.001 * gk[i] * gk[i];
        pk[i] -= LR * (m1[i] / c1) / (Math.sqrt(m2[i] / c2) + 1e-8);
      }
    }
  };

  Net.prototype.accuracy = function (data) {
    var c = 0;
    for (var i = 0; i < data.n; i++) {
      var z = this.forward(data.X[2 * i], data.X[2 * i + 1]);
      if ((z > 0 ? 1 : 0) === data.Y[i]) c++;
    }
    return c / data.n;
  };

  function build(mount) {
    var seed = 12, rate = 0.3;
    var exp = null;            // the running experiment
    var raf = null, running = false, started = false;
    var view = null;           // { round, kind } shown in the boundary plot

    // --- chrome ------------------------------------------------------------
    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent =
      "Iterative magnitude pruning with rewinding";
    el("div", "widget-meta", head).textContent =
      "2-" + H + "-" + H + "-1 MLP · " + TOTAL_W + " weights";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    stage.style.flexBasis = "40%";
    var bLabel = el("div", "widget-label", stage);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Decision boundary of the selected sparse network over the spiral data");

    var side = el("div", "widget-side", body);
    side.style.flexBasis = "60%";
    el("div", "widget-label", side).textContent =
      "Test accuracy vs weights remaining";
    var chart = el("canvas", "widget-plot", side);
    chart.style.cursor = "pointer";
    chart.setAttribute("aria-label",
      "Test accuracy against fraction of weights remaining; click a round to view it");

    var legend = el("div", null, side);
    legend.style.cssText = "margin-top:0.5rem;font-size:0.74rem;line-height:1.6;";
    var legTicket = el("span", null, legend);
    var legReinit = el("span", null, legend);

    var rateRow = el("div", "widget-slider-row", side);
    rateRow.style.marginTop = "0.6rem";
    el("span", null, rateRow).textContent = "prune rate";
    var rateIn = el("input", "widget-slider", rateRow);
    rateIn.type = "range";
    rateIn.min = "0.2"; rateIn.max = "0.5"; rateIn.step = "0.05";
    rateIn.value = String(rate);
    rateIn.setAttribute("aria-label", "fraction pruned per round");
    var rateVal = el("span", "widget-slider-val", rateRow);
    rateVal.textContent = Math.round(rate * 100) + "%";
    rateIn.addEventListener("input", function () {
      rate = Number(rateIn.value);
      rateVal.textContent = Math.round(rate * 100) + "%";
    });
    rateIn.addEventListener("change", function () { reset(); start(); });

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

    var runBtn = button("Pause", function () {
      if (running) pause(); else if (exp && exp.done) { reset(); start(); }
      else start();
    }, true);
    var showTicket = button("Show ticket", function () { setKind("ticket"); });
    var showReinit = button("Show reinit", function () { setKind("reinit"); });
    button("New seed", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      reset(); start();
    });

    // --- experiment ---------------------------------------------------------
    function reset() {
      pause();
      var rnd = mulberry32(seed);
      var rounds = Math.floor(Math.log(MIN_DENSITY) / Math.log(1 - rate));
      exp = {
        train: makeData(NTRAIN, rnd),
        test: makeData(NTEST, rnd),
        theta0: initParams(rnd),
        rounds: rounds,
        round: 0,
        phase: "ticket",          // which net of this round is training
        mask: fullMask(),
        net: null,
        results: [],              // { density, ticket, reinit, tNet, rNet }
        done: false,
      };
      exp.net = new Net(cloneParams(exp.theta0), exp.mask);
      view = null;
      refresh();
    }

    function finishRun() {
      var acc = exp.net.accuracy(exp.test);
      var r = exp.results[exp.round];
      if (!r) {
        r = { density: density(exp.mask), ticket: null, reinit: null };
        exp.results[exp.round] = r;
      }
      if (exp.phase === "ticket") {
        r.ticket = acc; r.tNet = exp.net;
        if (exp.round === 0) {        // dense net: both curves start here
          r.reinit = acc; r.rNet = exp.net;
          nextRound();
        } else {
          exp.phase = "reinit";
          var rr = mulberry32((seed * 7919 + exp.round * 104729) >>> 0);
          exp.net = new Net(initParams(rr), exp.mask);
        }
      } else {
        r.reinit = acc; r.rNet = exp.net;
        nextRound();
      }
    }

    function nextRound() {
      var ticket = exp.results[exp.round].tNet;
      if (exp.round >= exp.rounds) { exp.done = true; return; }
      exp.mask = prune(ticket.p, exp.mask, rate);
      exp.round++;
      exp.phase = "ticket";
      exp.net = new Net(cloneParams(exp.theta0), exp.mask);   // rewind
    }

    function frame() {
      raf = null;
      if (!running) return;
      var t0 = performance.now(), n = 0;
      // time-boxed, with a step cap in case the clock is frozen (e.g. headless)
      while (!exp.done && n++ < MAX_STEPS_PER_FRAME &&
        performance.now() - t0 < FRAME_MS) {
        exp.net.step(exp.train);
        if (exp.net.t >= STEPS) finishRun();
      }
      refresh();
      if (exp.done) {
        running = false;
        runBtn.textContent = "Run again";
        if (!view) view = { round: exp.results.length - 1, kind: "ticket" };
        refresh();
        return;
      }
      raf = requestAnimationFrame(frame);
    }

    function start() {
      if (running || !exp || exp.done) return;
      running = true; started = true;
      runBtn.textContent = "Pause";
      raf = requestAnimationFrame(frame);
    }

    function pause() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      runBtn.textContent = exp && exp.done ? "Run again" : "Resume";
    }

    function setKind(kind) {
      var last = exp.results.length - 1;
      if (last < 0) return;
      var round = view ? view.round : last;
      view = { round: round, kind: kind };
      refresh();
    }

    // --- drawing -------------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var cctx = chart.getContext("2d");
    var off = document.createElement("canvas");
    off.width = GRID; off.height = GRID;
    var octx = off.getContext("2d");
    var chartGeom = null;

    function palette() {
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        link: cssVar(mount, "--link", "#2156a5"),
        border: cssVar(mount, "--border", "#e2e2dd"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        bg: cssVar(mount, "--card-bg", "#fff"),
      };
    }

    // Which network the boundary plot shows: a selected round, or whatever
    // is training right now.
    function shownNet() {
      if (view && exp.results[view.round]) {
        var r = exp.results[view.round];
        var net = view.kind === "reinit" ? r.rNet : r.tNet;
        if (net) {
          return {
            net: net, round: view.round, kind: view.kind,
            density: r.density, live: false,
          };
        }
      }
      return {
        net: exp.net, round: exp.round, kind: exp.phase,
        density: density(exp.mask), live: !exp.done,
      };
    }

    function drawBoundary(pal) {
      var w = stage.clientWidth || 280;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = w + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);

      var s = shownNet();
      var A = hexRgb(pal.accent), B = hexRgb(pal.link);
      var img = octx.createImageData(GRID, GRID);
      for (var gy = 0; gy < GRID; gy++) {
        for (var gx = 0; gx < GRID; gx++) {
          var x = -EXT + 2 * EXT * (gx + 0.5) / GRID;
          var y = EXT - 2 * EXT * (gy + 0.5) / GRID;
          var pr = 1 / (1 + Math.exp(-s.net.forward(x, y)));
          var c = pr > 0.5 ? A : B;
          var o = 4 * (gy * GRID + gx);
          img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2];
          img.data[o + 3] = Math.round(255 * (0.06 + 0.2 * Math.abs(pr - 0.5) * 2));
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, w);

      var d = exp.train, sc = w / (2 * EXT);
      for (var i = 0; i < d.n; i++) {
        ctx.beginPath();
        ctx.arc((d.X[2 * i] + EXT) * sc, (EXT - d.X[2 * i + 1]) * sc,
          2.6, 0, 6.2832);
        ctx.fillStyle = d.Y[i] ? pal.accent : pal.link;
        ctx.fill();
        ctx.lineWidth = 0.8;
        ctx.strokeStyle = pal.bg;
        ctx.stroke();
      }

      var what = s.round === 0 ? "dense network"
        : (s.kind === "reinit" ? "random reinit" : "rewound ticket")
          + " · round " + s.round;
      bLabel.textContent = (s.live ? "Training: " : "Showing: ") + what + " · "
        + (100 * s.density).toFixed(1) + "% weights";
      showTicket.disabled = s.kind === "ticket" && !s.live;
      showReinit.disabled = (s.kind === "reinit" && !s.live) || s.round === 0;
    }

    function drawChart(pal) {
      var w = side.clientWidth || 320;
      var h = Math.max(170, Math.min(230, Math.round(w * 0.55)));
      var dpr = window.devicePixelRatio || 1;
      chart.style.height = h + "px";
      chart.width = Math.round(w * dpr);
      chart.height = Math.round(h * dpr);
      cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cctx.clearRect(0, 0, w, h);

      var L = 34, R = 10, T = 8, Bm = 24;
      var dMin = Math.pow(1 - rate, exp.rounds);
      var lx = function (dens) {
        return L + (w - L - R) * Math.log(dens) / Math.log(dMin);
      };
      var ly = function (a) { return T + (h - T - Bm) * (1 - a) / 0.5; };
      chartGeom = { lx: lx, ly: ly };

      cctx.font = "10px 'JetBrains Mono', monospace";
      cctx.lineWidth = 1;
      cctx.strokeStyle = pal.border;
      cctx.fillStyle = pal.muted;
      cctx.textAlign = "right";
      cctx.textBaseline = "middle";
      [0.5, 0.75, 1].forEach(function (a) {
        cctx.beginPath();
        cctx.moveTo(L, ly(a)); cctx.lineTo(w - R, ly(a));
        cctx.stroke();
        cctx.textBaseline = a === 0.5 ? "bottom" : "middle";
        cctx.fillText(Math.round(a * 100) + "%", L - 4, ly(a));
      });
      cctx.textAlign = "center";
      cctx.textBaseline = "top";
      [1, 0.5, 0.25, 0.125, 0.0625].forEach(function (dn) {
        if (dn < dMin * 0.95) return;
        cctx.fillText(+(dn * 100).toFixed(1) + "%", lx(dn), h - Bm + 6);
      });

      function series(key, col) {
        var pts = exp.results.filter(function (r) { return r[key] !== null; });
        if (!pts.length) return;
        cctx.strokeStyle = col;
        cctx.lineWidth = 2;
        cctx.lineJoin = "round";
        cctx.beginPath();
        pts.forEach(function (r, i) {
          var x = lx(r.density), y = ly(Math.max(0.5, r[key]));
          if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
        });
        cctx.stroke();
        pts.forEach(function (r) {
          cctx.beginPath();
          cctx.arc(lx(r.density), ly(Math.max(0.5, r[key])), 3, 0, 6.2832);
          cctx.fillStyle = col;
          cctx.fill();
        });
      }
      series("reinit", pal.link);
      series("ticket", pal.accent);

      // selected / training round marker
      var s = shownNet();
      cctx.strokeStyle = pal.muted;
      cctx.setLineDash([3, 3]);
      cctx.beginPath();
      cctx.moveTo(lx(s.density), T);
      cctx.lineTo(lx(s.density), h - Bm);
      cctx.stroke();
      cctx.setLineDash([]);

      legTicket.innerHTML = "";
      legReinit.innerHTML = "";
      function key(span, col, text) {
        var sw = el("span", null, span);
        sw.style.cssText = "display:inline-block;width:14px;height:3px;"
          + "border-radius:2px;vertical-align:middle;margin-right:6px;background:"
          + col;
        el("span", null, span).textContent = text;
        span.style.marginRight = "1rem";
        span.style.whiteSpace = "nowrap";
        span.style.color = pal.muted;
      }
      key(legTicket, pal.accent, "rewound to θ₀ (ticket)");
      key(legReinit, pal.link, "same mask, random reinit");
    }

    function describe() {
      var res = exp.results;
      if (!res.length) {
        return "Training the dense network from θ₀. Each round then prunes "
          + Math.round(rate * 100) + "% of the remaining weights, rewinds the "
          + "survivors to θ₀ and retrains.";
      }
      var last = res[res.length - 1];
      if (!exp.done) {
        return "Round " + exp.round + " of " + exp.rounds + ": training the "
          + (exp.phase === "ticket" ? "rewound ticket" : "random-reinit control")
          + " on " + (100 * density(exp.mask)).toFixed(1) + "% of the weights. "
          + "Click a point on the chart to inspect that round.";
      }
      var wins = 0, n = 0;
      res.forEach(function (r) {
        if (r.density < 0.15) { n++; if (r.ticket > r.reinit) wins++; }
      });
      return "Done. Below 15% of weights the rewound ticket beat the reinitialised "
        + "mask in " + wins + " of " + n + " rounds; at "
        + (100 * last.density).toFixed(1) + "% it scores "
        + (100 * last.ticket).toFixed(1) + "% vs "
        + (100 * last.reinit).toFixed(1) + "%. Single runs are noisy: try New "
        + "seed. Click any point to see that network's boundary.";
    }

    function refresh() {
      if (!exp) return;
      var pal = palette();
      drawBoundary(pal);
      drawChart(pal);
      note.textContent = describe();
    }

    chart.addEventListener("pointerdown", function (ev) {
      if (!chartGeom || !exp.results.length) return;
      var r = chart.getBoundingClientRect();
      var mx = ev.clientX - r.left, my = ev.clientY - r.top;
      var best = -1, bestD = 1e9, kind = "ticket";
      exp.results.forEach(function (res, i) {
        ["ticket", "reinit"].forEach(function (k) {
          if (res[k] === null || !(k === "ticket" ? res.tNet : res.rNet)) return;
          var dx = chartGeom.lx(res.density) - mx;
          var dy = chartGeom.ly(Math.max(0.5, res[k])) - my;
          var dd = dx * dx + dy * dy * 0.25;
          if (dd < bestD) { bestD = dd; best = i; kind = k; }
        });
      });
      if (best < 0) return;
      view = { round: best, kind: kind };
      refresh();
    });

    // --- lifecycle ------------------------------------------------------------
    reset();
    runBtn.textContent = "Run";
    // Start once the figure is on screen, so readers who never scroll here
    // don't pay for ~25 training runs.
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting && !started) { start(); io.disconnect(); }
      }, { threshold: 0.2 });
      io.observe(mount);
    } else {
      start();
    }

    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="lottery-ticket"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

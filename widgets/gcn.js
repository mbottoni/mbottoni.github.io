// GCN propagation on a two-community graph, and the over-smoothing that follows.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-gcn
//   :::
//
// Each layer applies the symmetric normalised adjacency with self-loops,
//   H <- D^-1/2 (A + I) D^-1/2 H,
// which is exactly the propagation rule in Kipf & Welling's GCN with the weight
// matrix and nonlinearity stripped out. Stack enough of them and every node
// converges to the same signal (up to a sqrt(degree) factor), because that
// matrix's dominant eigenvector is proportional to sqrt(deg) -- so the community
// structure the features started with is destroyed. That is over-smoothing.
(function () {
  "use strict";

  var N = 22;                    // nodes, split evenly into two communities
  var HALF = N / 2;
  // P_OUT and MAX_LAYERS are set from a sweep: the communities have to mix
  // slowly enough to look like communities, but fast enough that the collapse
  // is visible inside the slider's range. At these values the community
  // separation runs 1.00 (depth 0) -> 0.65 -> 0.40 (depth 12) -> 0.13 (depth 24).
  var MAX_LAYERS = 24;
  var P_IN = 0.34, P_OUT = 0.12;

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function cssVar(el_, name, fallback) {
    var v = getComputedStyle(el_).getPropertyValue(name).trim();
    return v || fallback;
  }

  function rgba(hex, a) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
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

  // Stochastic block model, with a ring inside each community so nothing is
  // ever isolated and the graph stays connected.
  function makeGraph(seed) {
    var rnd = mulberry32(seed);
    var A = [];
    for (var i = 0; i < N; i++) A.push(new Float64Array(N));
    function link(i, j) { A[i][j] = 1; A[j][i] = 1; }

    for (var c = 0; c < 2; c++) {
      var base = c * HALF;
      for (var k = 0; k < HALF; k++) link(base + k, base + (k + 1) % HALF);
    }
    for (var a = 0; a < N; a++) {
      for (var b = a + 1; b < N; b++) {
        var same = (a < HALF) === (b < HALF);
        if (rnd() < (same ? P_IN : P_OUT)) link(a, b);
      }
    }
    // guarantee at least two bridges between the communities
    var bridges = 0;
    for (var x = 0; x < HALF; x++) {
      for (var y = HALF; y < N; y++) if (A[x][y]) bridges++;
    }
    while (bridges < 2) {
      link((rnd() * HALF) | 0, HALF + ((rnd() * HALF) | 0));
      bridges++;
    }

    // positions: two rings, jittered
    var pos = [];
    for (var p = 0; p < N; p++) {
      var comm = p < HALF ? 0 : 1;
      var idx = p % HALF;
      var ang = 2 * Math.PI * idx / HALF + (comm ? 0.3 : 0);
      var cx = comm ? 0.70 : 0.30;
      var r = 0.175 + 0.03 * (rnd() - 0.5);
      pos.push({
        x: cx + r * Math.cos(ang) * 0.95,
        y: 0.5 + r * Math.sin(ang) * 2.45,
      });
    }

    // normalised adjacency with self-loops
    var deg = new Float64Array(N);
    for (var i2 = 0; i2 < N; i2++) {
      var d = 1;
      for (var j2 = 0; j2 < N; j2++) d += A[i2][j2];
      deg[i2] = d;
    }
    var Ahat = [];
    for (var i3 = 0; i3 < N; i3++) {
      Ahat.push(new Float64Array(N));
      for (var j3 = 0; j3 < N; j3++) {
        var v = (i3 === j3 ? 1 : 0) + A[i3][j3];
        Ahat[i3][j3] = v / Math.sqrt(deg[i3] * deg[j3]);
      }
    }
    return { A: A, Ahat: Ahat, deg: deg, pos: pos };
  }

  function propagate(Ahat, h0, layers) {
    var h = h0.slice();
    for (var l = 0; l < layers; l++) {
      var nh = new Float64Array(N);
      for (var i = 0; i < N; i++) {
        var s = 0;
        for (var j = 0; j < N; j++) s += Ahat[i][j] * h[j];
        nh[i] = s;
      }
      h = nh;
    }
    return h;
  }

  // Dirichlet energy of the normalised signal -- the standard measure of how
  // smooth a signal is over a graph. It goes to zero under over-smoothing.
  function dirichlet(A, deg, h) {
    var e = 0;
    for (var i = 0; i < N; i++) {
      for (var j = i + 1; j < N; j++) {
        if (!A[i][j]) continue;
        var d = h[i] / Math.sqrt(deg[i]) - h[j] / Math.sqrt(deg[j]);
        e += d * d;
      }
    }
    return e;
  }

  function build(mount) {
    var seed = 7;
    var g = makeGraph(seed);
    var layers = 0;
    var mode = "communities";
    var h0 = new Float64Array(N);

    function setInit(kind, node) {
      mode = kind;
      h0 = new Float64Array(N);
      if (kind === "communities") {
        for (var i = 0; i < N; i++) h0[i] = i < HALF ? 1 : -1;
      } else if (kind === "spike") {
        h0[node === undefined ? 0 : node] = 1;
      } else {
        var rnd = mulberry32(seed * 31 + layers);
        for (var j = 0; j < N; j++) h0[j] = rnd() * 2 - 1;
      }
    }
    setInit("communities");

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "GCN propagation and over-smoothing";
    el("div", "widget-meta", head).textContent = N + " nodes · 2 communities";
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.style.cursor = "pointer";
    canvas.setAttribute("aria-label",
      "Two-community graph; node colour is its feature after the chosen number of GCN layers");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "Depth";
    var row = el("div", "widget-slider-row", left);
    el("span", null, row).textContent = "layers";
    var slider = el("input", "widget-slider", row);
    slider.type = "range";
    slider.min = "0";
    slider.max = String(MAX_LAYERS);
    slider.step = "1";
    slider.value = "0";
    var layerVal = el("span", "widget-slider-val", row);
    slider.addEventListener("input", function () {
      layers = Number(slider.value);
      refresh();
    });

    el("div", "widget-label", right).textContent = "Signal";

    function metric(name, logBar) {
      var r = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", r).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", r);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", r);
      return function (v) {
        // Dirichlet energy spans four decades over this slider, so a linear bar
        // would sit pinned at zero for most of the range.
        var bar = logBar
          ? 1 + Math.log(Math.max(v, 1e-5)) / (5 * Math.LN10)
          : v;
        fill.style.width = (100 * Math.max(0, Math.min(1, bar))).toFixed(1) + "%";
        val.textContent = v >= 0.001 || v === 0 ? v.toFixed(3) : v.toExponential(1);
      };
    }
    var setSep = metric("separation", false);
    var setEnergy = metric("smoothness", true);

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

    button("Two communities", function () {
      setInit("communities"); layers = 0; slider.value = "0"; refresh();
    }, true);
    button("Single spike", function () {
      setInit("spike", 0); layers = 0; slider.value = "0"; refresh();
    });
    button("New graph", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      g = makeGraph(seed);
      setInit(mode === "spike" ? "spike" : "communities", 0);
      layers = 0; slider.value = "0";
      refresh();
    });

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0;
    var lastH = h0;

    function draw(h) {
      var cssW = mount.clientWidth || 520;
      W = cssW;
      H = Math.max(180, Math.min(240, Math.round(cssW * 0.40)));
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      var accent = cssVar(mount, "--accent", "#ba3925");
      var link = cssVar(mount, "--link", "#2156a5");
      var border = cssVar(mount, "--border", "#e2e2dd");
      var base = cssVar(mount, "--card-bg", "#fff");

      var maxAbs = 1e-9;
      for (var i = 0; i < N; i++) maxAbs = Math.max(maxAbs, Math.abs(h[i]));

      var X = function (p) { return 20 + (W - 40) * p.x; };
      var Y = function (p) { return 14 + (H - 28) * p.y; };

      ctx.strokeStyle = rgba(border, 1);
      ctx.lineWidth = 1;
      for (var a = 0; a < N; a++) {
        for (var b = a + 1; b < N; b++) {
          if (!g.A[a][b]) continue;
          ctx.beginPath();
          ctx.moveTo(X(g.pos[a]), Y(g.pos[a]));
          ctx.lineTo(X(g.pos[b]), Y(g.pos[b]));
          ctx.stroke();
        }
      }

      for (var n = 0; n < N; n++) {
        var v = h[n] / maxAbs;
        var col = v >= 0 ? accent : link;
        ctx.beginPath();
        ctx.arc(X(g.pos[n]), Y(g.pos[n]), 9, 0, 6.284);
        ctx.fillStyle = rgba(col, Math.max(0.06, Math.abs(v)));
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = rgba(col, 0.85);
        ctx.stroke();
      }
      void base;
    }

    function refresh() {
      var h = propagate(g.Ahat, h0, layers);
      lastH = h;
      draw(h);
      layerVal.textContent = String(layers);

      var maxAbs = 1e-9;
      for (var i = 0; i < N; i++) maxAbs = Math.max(maxAbs, Math.abs(h[i]));
      var mA = 0, mB = 0;
      for (var a = 0; a < HALF; a++) mA += h[a];
      for (var b = HALF; b < N; b++) mB += h[b];
      mA /= HALF; mB /= HALF;
      var sep = Math.abs(mA - mB) / (2 * maxAbs);

      var e0 = dirichlet(g.A, g.deg, h0);
      var e = dirichlet(g.A, g.deg, h);
      var energy = e0 > 1e-12 ? e / e0 : 0;

      setSep(sep);
      setEnergy(energy);

      if (mode === "spike") {
        note.textContent = layers === 0
          ? "One node carries all the signal. Each layer mixes a node with its "
            + "neighbours, so the signal spreads one hop per layer — click any "
            + "node to move the spike."
          : "The signal has spread " + layers + " hop" + (layers > 1 ? "s" : "")
            + " from the source. A GCN's receptive field is exactly its depth.";
      } else if (layers === 0) {
        note.textContent = "Two communities, opposite features, perfectly "
          + "separated. Drag the layer slider and watch what stacking "
          + "propagation does to that separation.";
      } else if (sep > 0.45) {
        note.textContent = "After " + layers + " layers the communities are "
          + "still distinguishable, but the contrast is fading.";
      } else if (sep > 0.2) {
        note.textContent = "Separation is down to " + sep.toFixed(2)
          + ". A classifier reading these features has much less to work with "
          + "than it did at layer 1.";
      } else {
        note.textContent = "Over-smoothed. Every node has converged to nearly "
          + "the same value — the dominant eigenvector of the propagation "
          + "matrix, which is proportional to sqrt(degree) and carries no "
          + "community information at all. This is why deep GCNs get worse, "
          + "not better.";
      }
    }

    canvas.addEventListener("pointerdown", function (ev) {
      var r = canvas.getBoundingClientRect();
      var mx = (ev.clientX - r.left) * (W / r.width);
      var my = (ev.clientY - r.top) * (H / r.height);
      var best = -1, bestD = 18;
      for (var i = 0; i < N; i++) {
        var dx = mx - (20 + (W - 40) * g.pos[i].x);
        var dy = my - (14 + (H - 28) * g.pos[i].y);
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best < 0) return;
      ev.preventDefault();
      setInit("spike", best);
      refresh();
    });

    refresh();
    void lastH;
    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="gcn"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

// Interactive comparison of divergences between two distributions.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-divergences
//   :::
//
// Two Gaussians are discretised onto a shared grid and every quantity is
// computed from those two pmfs: KL in both directions, Jensen-Shannon,
// Hellinger and total variation. The point of dragging them around is to watch
// KL(P||Q) and KL(Q||P) disagree while the three symmetric measures stay put.
//
// Verified against the closed forms for Gaussians (see the commit that added
// this file): KL, Hellinger, and -- for equal variances -- total variation.
(function () {
  "use strict";

  // The integration grid is much wider than the plotted window on purpose. A
  // wide Q still has real mass outside the visible range, and out there
  // log(q/p) is large, so a grid that stops at the edge of the picture
  // truncates KL by a visible amount.
  var M = 1400;                // integration grid points
  var XMIN = -22, XMAX = 22;   // integration range
  var VMIN = -7, VMAX = 7;     // plotted window
  var DX = (XMAX - XMIN) / (M - 1);
  var SMIN = 0.35, SMAX = 2.5;
  var YMAX = 1.2;
  var LN2 = Math.log(2);
  var PLOTPTS = 320;           // samples used to draw a curve

  var GRID = new Float64Array(M);
  for (var gi = 0; gi < M; gi++) GRID[gi] = XMIN + gi * DX;

  // Everything is kept in log space. The post's Python clips the pmfs at 1e-10
  // before taking the ratio, which is fine when the two distributions overlap
  // but silently truncates KL once they separate -- exactly the regime where KL
  // is most interesting. Working with log-densities keeps it exact.
  function logpmf(mu, sigma) {
    var lp = new Float64Array(M), mx = -Infinity, i;
    for (i = 0; i < M; i++) {
      var z = (GRID[i] - mu) / sigma;
      lp[i] = -0.5 * z * z - Math.log(sigma);
      if (lp[i] > mx) mx = lp[i];
    }
    var s = 0;
    for (i = 0; i < M; i++) s += Math.exp(lp[i] - mx);
    var lz = mx + Math.log(s);
    for (i = 0; i < M; i++) lp[i] -= lz;
    return lp;
  }

  function kl(lp, lq) {
    var s = 0;
    for (var i = 0; i < M; i++) s += Math.exp(lp[i]) * (lp[i] - lq[i]);
    return s;
  }

  function logMix(lp, lq) {
    var lm = new Float64Array(M);
    for (var i = 0; i < M; i++) {
      var a = lp[i], b = lq[i];
      var mx = a > b ? a : b;
      lm[i] = mx + Math.log(Math.exp(a - mx) + Math.exp(b - mx)) - LN2;
    }
    return lm;
  }

  function js(lp, lq) {
    var lm = logMix(lp, lq);
    return 0.5 * kl(lp, lm) + 0.5 * kl(lq, lm);
  }

  function hellinger(lp, lq) {
    var s = 0;
    for (var i = 0; i < M; i++) {
      var d = Math.exp(0.5 * lp[i]) - Math.exp(0.5 * lq[i]);
      s += d * d;
    }
    return Math.sqrt(s) / Math.SQRT2;
  }

  function tv(lp, lq) {
    var s = 0;
    for (var i = 0; i < M; i++) s += Math.abs(Math.exp(lp[i]) - Math.exp(lq[i]));
    return s / 2;
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
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + a + ")";
  }

  var PRESETS = [
    { label: "Reset", P: [-1.2, 0.9], Q: [1.2, 0.9] },
    { label: "Same mean", P: [0, 0.6], Q: [0, 1.6] },
    { label: "Far apart", P: [-2.6, 0.8], Q: [2.6, 0.8] },
    { label: "Nested", P: [0, 0.45], Q: [0.3, 2.1] },
  ];

  function build(mount) {
    var P = { mu: -1.2, sigma: 0.9 };
    var Q = { mu: 1.2, sigma: 0.9 };

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Divergences between two distributions";
    el("div", "widget-meta", head).textContent = "drag P and Q";
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.setAttribute("aria-label",
      "Two distributions P and Q; drag each handle to move its mean or change its width");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "Asymmetric (nats)";
    el("div", "widget-label", right).textContent = "Symmetric";

    function metric(parent, name, max) {
      var row = el("div", "widget-row widget-row--wide", parent);
      el("span", "widget-row-name", row).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", row);
      return function (v) {
        val.textContent = v.toFixed(3);
        fill.style.width = (100 * Math.min(v / max, 1)).toFixed(1) + "%";
      };
    }

    var setKLpq = metric(left, "KL(P‖Q)", 6);
    var setKLqp = metric(left, "KL(Q‖P)", 6);
    var gap = el("div", "widget-note", left);
    gap.style.minHeight = "0";
    gap.style.marginTop = "0.35rem";

    var setJS = metric(right, "JS", LN2);
    var setHel = metric(right, "Hellinger", 1);
    var setTV = metric(right, "Total var.", 1);

    var note = el("div", "widget-note", mount);
    var controls = el("div", "widget-controls", mount);

    PRESETS.forEach(function (preset, idx) {
      var b = el("button", "widget-btn", controls);
      b.type = "button";
      b.textContent = preset.label;
      if (idx === 0) b.setAttribute("data-primary", "");
      b.addEventListener("click", function () {
        P.mu = preset.P[0]; P.sigma = preset.P[1];
        Q.mu = preset.Q[0]; Q.sigma = preset.Q[1];
        refresh();
      });
    });

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0, PAD = 14;

    function px(x) { return PAD + (W - 2 * PAD) * (x - VMIN) / (VMAX - VMIN); }
    var yscale = YMAX;
    function py(v) { return H - 18 - (H - 34) * Math.min(v / yscale, 1); }
    function vx(k) { return VMIN + (VMAX - VMIN) * k / (PLOTPTS - 1); }

    function density(d, x) {
      var z = (x - d.mu) / d.sigma;
      return Math.exp(-0.5 * z * z) / (d.sigma * Math.sqrt(2 * Math.PI));
    }

    function curve(d, colour) {
      ctx.beginPath();
      ctx.moveTo(px(VMIN), py(0));
      for (var i = 0; i < PLOTPTS; i++) ctx.lineTo(px(vx(i)), py(density(d, vx(i))));
      ctx.lineTo(px(VMAX), py(0));
      ctx.closePath();
      ctx.fillStyle = rgba(colour, 0.16);
      ctx.fill();
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function draw() {
      var cssW = mount.clientWidth || 520;
      H = Math.max(190, Math.min(250, Math.round(cssW * 0.42)));
      W = cssW;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // Scale to the taller of the two peaks: a fixed scale wastes most of the
      // plot for typical widths. Relative height between P and Q is preserved.
      yscale = Math.max(0.35, Math.min(YMAX,
        1.14 * Math.max(density(P, P.mu), density(Q, Q.mu))));

      var blue = cssVar(mount, "--link", "#2156a5");
      var red = cssVar(mount, "--accent", "#ba3925");
      var line = cssVar(mount, "--border", "#e2e2dd");
      var muted = cssVar(mount, "--muted", "#6a6a6a");

      // Shared mass: total variation is 1 minus the area of this region.
      ctx.beginPath();
      ctx.moveTo(px(VMIN), py(0));
      for (var i = 0; i < PLOTPTS; i++) {
        ctx.lineTo(px(vx(i)),
          py(Math.min(density(P, vx(i)), density(Q, vx(i)))));
      }
      ctx.lineTo(px(VMAX), py(0));
      ctx.closePath();
      ctx.fillStyle = rgba(muted, 0.28);
      ctx.fill();

      curve(P, blue);
      curve(Q, red);

      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, py(0) + 0.5);
      ctx.lineTo(W, py(0) + 0.5);
      ctx.stroke();

      handle(P, blue, "P");
      handle(Q, red, "Q");
    }

    function handle(d, colour, label) {
      var x = px(d.mu), y = py(density(d, d.mu));
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 6.284);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "600 10px 'Open Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x, y + 0.5);
    }

    function refresh() {
      draw();
      var p = logpmf(P.mu, P.sigma), q = logpmf(Q.mu, Q.sigma);
      var a = kl(p, q), b = kl(q, p);
      setKLpq(a); setKLqp(b);
      setJS(js(p, q)); setHel(hellinger(p, q)); setTV(tv(p, q));
      var d = Math.abs(a - b);
      gap.textContent = d < 0.005
        ? "The two directions agree here — but that is a coincidence of this "
          + "arrangement, not a property of KL."
        : "The two directions differ by " + d.toFixed(3) + " nats. KL is not a "
          + "distance.";
    }

    // --- interaction -------------------------------------------------------
    var drag = null;

    function pointer(ev) {
      var r = canvas.getBoundingClientRect();
      return { x: (ev.clientX - r.left) * (W / r.width),
               y: (ev.clientY - r.top) * (H / r.height) };
    }

    canvas.addEventListener("pointerdown", function (ev) {
      var pt = pointer(ev);
      var best = null, bestD = 26;
      [[P, "P"], [Q, "Q"]].forEach(function (pair) {
        var d = pair[0];
        var dx = pt.x - px(d.mu), dy = pt.y - py(density(d, d.mu));
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestD) { bestD = dist; best = d; }
      });
      if (!best) return;
      ev.preventDefault();
      canvas.setPointerCapture(ev.pointerId);
      drag = { d: best, y0: pt.y, sigma0: best.sigma };
      note.textContent = "Dragging. Sideways moves the mean; up and down "
        + "changes the width.";
    });

    canvas.addEventListener("pointermove", function (ev) {
      if (!drag) return;
      var pt = pointer(ev);
      var x = VMIN + (pt.x - PAD) * (VMAX - VMIN) / (W - 2 * PAD);
      drag.d.mu = Math.max(-5, Math.min(5, x));
      var s = drag.sigma0 * Math.exp((pt.y - drag.y0) / 90);
      drag.d.sigma = Math.max(SMIN, Math.min(SMAX, s));
      refresh();
    });

    function endDrag() { drag = null; }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    note.textContent = "Drag the P and Q handles: sideways moves the mean, up "
      + "and down changes the width. The shaded overlap is the mass the two "
      + "distributions share — total variation is one minus its area.";
    refresh();

    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="divergences"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

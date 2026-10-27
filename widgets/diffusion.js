// DDPM vs DDIM sampling on a 2-D toy distribution.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-diffusion
//   :::
//
// There is no trained network here, and that is the point. The target is a
// known mixture of Gaussians, so the noised marginal
//   p_t(x) = sum_k w_k N(x; sqrt(abar_t) mu_k, (abar_t s^2 + 1 - abar_t) I)
// has a closed-form score, and eps_hat = -sqrt(1 - abar_t) * score is therefore
// exact (checked against a finite-difference gradient of log p_t to 1.5e-8).
// Whatever you see is the sampler's behaviour, not a network's quality.
//
// Both samplers are the one DDIM update with different eta, exactly as in the
// post: eta = 0 is deterministic DDIM, eta = 1 recovers the stochastic DDPM
// ancestral step on the same subsequence of timesteps.
//
// A caveat this widget is deliberately honest about. Because the score is
// exact, DDPM is *not* worse than DDIM here -- measured over 2000 samples it is
// slightly better at moderate step counts (mean distance to a mode centre 0.126
// against DDIM's 0.143 at 30 steps). DDIM's real-world advantage at low step
// counts comes from learned networks, whose errors compound differently under a
// stochastic sampler; it is not a property of the update rule on its own. What
// this toy can show honestly is the part that *is* intrinsic to eta: at eta = 0
// the sampler is a deterministic map from the initial noise to the sample, and
// at eta > 0 it is not. That is what makes DDIM latents interpolable.
(function () {
  "use strict";

  var T = 1000;                        // full schedule length
  var BETA0 = 1e-4, BETA1 = 0.02;      // linear beta schedule
  var P = 700;                         // particles
  var NMODES = 8;
  var RING = 1.55, MODE_SD = 0.13;
  var VIEW = 2.6;                      // plot half-width in data units
  var IDEAL = MODE_SD * Math.sqrt(Math.PI / 2);   // mean |x - mu| for a hit

  var ABAR = new Float64Array(T);
  (function () {
    var prod = 1;
    for (var t = 0; t < T; t++) {
      prod *= (1 - (BETA0 + (BETA1 - BETA0) * t / (T - 1)));
      ABAR[t] = prod;
    }
  })();

  var MODES = [];
  for (var m = 0; m < NMODES; m++) {
    var a = 2 * Math.PI * m / NMODES;
    MODES.push([RING * Math.cos(a), RING * Math.sin(a)]);
  }

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

  function rgba(hex, alpha) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + alpha + ")";
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussFrom(rnd) {
    var u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Exact eps-prediction for the mixture, at noise level abar.
  function epsHat(x, y, abar, out) {
    var sa = Math.sqrt(abar);
    var vt = abar * MODE_SD * MODE_SD + (1 - abar);
    var best = -Infinity, i;
    var w = new Float64Array(NMODES);
    for (i = 0; i < NMODES; i++) {
      var dx = x - sa * MODES[i][0];
      var dy = y - sa * MODES[i][1];
      w[i] = -(dx * dx + dy * dy) / (2 * vt);
      if (w[i] > best) best = w[i];
    }
    var sum = 0;
    for (i = 0; i < NMODES; i++) { w[i] = Math.exp(w[i] - best); sum += w[i]; }
    var sx = 0, sy = 0;
    for (i = 0; i < NMODES; i++) {
      var r = w[i] / sum;
      sx += r * (sa * MODES[i][0] - x);
      sy += r * (sa * MODES[i][1] - y);
    }
    var k = -Math.sqrt(1 - abar) / vt;     // eps = -sqrt(1-abar) * score
    out[0] = k * sx;
    out[1] = k * sy;
  }

  function timesteps(steps) {
    var seq = [];
    for (var i = 0; i < steps; i++) seq.push(Math.floor(i * T / steps));
    return seq;
  }

  // One reverse step, applied in place to a particle array.
  var EPS = [0, 0];
  function reverseStep(xs, seq, cursor, eta, rnd) {
    var t = seq[cursor];
    var tPrev = cursor > 0 ? seq[cursor - 1] : -1;
    var abar = ABAR[t];
    var abarNext = tPrev >= 0 ? ABAR[tPrev] : 1;
    var sig = eta * Math.sqrt(
      (1 - abar / abarNext) * (1 - abarNext) / (1 - abar));
    // guard the final step, where 1 - abarNext - sig^2 can go slightly negative
    var dirCoef = Math.sqrt(Math.max(0, 1 - abarNext - sig * sig));
    var sa = Math.sqrt(abar), sNext = Math.sqrt(abarNext);
    var s1ma = Math.sqrt(1 - abar);
    var n = xs.length / 2;
    for (var i = 0; i < n; i++) {
      var x = xs[2 * i], y = xs[2 * i + 1];
      epsHat(x, y, abar, EPS);
      var x0x = (x - s1ma * EPS[0]) / sa;
      var x0y = (y - s1ma * EPS[1]) / sa;
      var nx = sNext * x0x + dirCoef * EPS[0];
      var ny = sNext * x0y + dirCoef * EPS[1];
      if (sig > 0 && cursor > 0) {
        nx += sig * gaussFrom(rnd);
        ny += sig * gaussFrom(rnd);
      }
      xs[2 * i] = nx;
      xs[2 * i + 1] = ny;
    }
  }

  function build(mount) {
    var steps = 40, eta = 0;
    var seedNoise = 20260802;          // seeds the *initial* latent
    var seedStream = 1;                // seeds the injected noise
    var xs = new Float64Array(P * 2);
    var x0 = new Float64Array(P * 2);  // the initial latent, kept for replay
    var seq = timesteps(steps);
    var cursor = -1, raf = null, running = false;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "DDPM vs DDIM sampling";
    el("div", "widget-meta", head).textContent = P + " samples · exact score";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Samples moving from Gaussian noise to an eight-mode ring distribution");
    var side = el("div", "widget-side", body);

    el("div", "widget-label", side).textContent = "Sampler";

    function slider(label, lo, hi, initial, fmt, onChange) {
      var row = el("div", "widget-slider-row", side);
      el("span", null, row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range";
      input.min = String(lo); input.max = String(hi); input.step = "1";
      input.value = String(initial);
      var val = el("span", "widget-slider-val", row);
      val.textContent = fmt(initial);
      input.addEventListener("input", function () {
        var v = Number(input.value);
        val.textContent = fmt(v);
        onChange(v);
      });
      return {
        set: function (v) { input.value = String(v); val.textContent = fmt(v); },
      };
    }

    var stepSlider = slider("steps", 3, 120, steps,
      function (v) { return String(v); },
      function (v) { steps = v; restart(true); });
    var etaSlider = slider("eta", 0, 100, 0,
      function (v) { return (v / 100).toFixed(2); },
      function (v) { eta = v / 100; restart(true); });

    el("div", "widget-label", side).textContent = "Result";

    function metric(name, fmt) {
      var r = el("div", "widget-row widget-row--wide", side);
      var n = el("span", "widget-row-name", r);
      n.style.width = "7.4em";
      n.textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", r);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", r);
      return function (v, bar) {
        fill.style.width = (100 * Math.max(0, Math.min(1, bar))).toFixed(1) + "%";
        val.textContent = fmt(v);
      };
    }
    var setDist = metric("mode dist", function (v) { return v.toFixed(3); });
    var setRepro = metric("re-run drift", function (v) { return v.toFixed(3); });
    var setNfe = metric("network calls", function (v) { return String(v); });

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

    button("DDIM (η = 0)", function () {
      eta = 0; etaSlider.set(0); restart(true);
    }, true);
    button("DDPM (η = 1)", function () {
      eta = 1; etaSlider.set(100); restart(true);
    });
    button("Few steps", function () {
      steps = 6; stepSlider.set(6); restart(true);
    });
    button("New latent", function () {
      seedNoise = (seedNoise * 1103515245 + 12345) >>> 0;
      restart(false);
    });

    // --- running ------------------------------------------------------------
    function restart(keepLatent) {
      if (raf) cancelAnimationFrame(raf);
      if (!keepLatent || !x0[0]) {
        var rnd = mulberry32(seedNoise);
        for (var i = 0; i < P * 2; i++) x0[i] = gaussFrom(rnd);
      }
      xs.set(x0);
      seq = timesteps(steps);
      cursor = seq.length - 1;
      running = true;
      raf = requestAnimationFrame(tick);
    }

    function tick() {
      var rnd = mulberry32(seedStream * 7919 + cursor);
      var perFrame = Math.max(1, Math.round(seq.length / 40));
      for (var k = 0; k < perFrame && cursor >= 0; k++) {
        reverseStep(xs, seq, cursor, eta, rnd);
        cursor--;
      }
      refresh();
      if (cursor >= 0) {
        raf = requestAnimationFrame(tick);
      } else {
        running = false;
        finish();
      }
    }

    // Re-run from the *same* initial latent with a different injected-noise
    // stream. At eta = 0 nothing is injected, so this must land in exactly the
    // same place; at eta > 0 it cannot.
    function rerunDrift() {
      var ys = new Float64Array(P * 2);
      ys.set(x0);
      for (var c = seq.length - 1; c >= 0; c--) {
        reverseStep(ys, seq, c, eta, mulberry32(987654 + c * 31));
      }
      var s = 0;
      for (var i = 0; i < P; i++) {
        var dx = ys[2 * i] - xs[2 * i], dy = ys[2 * i + 1] - xs[2 * i + 1];
        s += Math.sqrt(dx * dx + dy * dy);
      }
      return s / P;
    }

    function meanModeDist() {
      var s = 0;
      for (var i = 0; i < P; i++) {
        var x = xs[2 * i], y = xs[2 * i + 1], bd = Infinity;
        for (var m = 0; m < NMODES; m++) {
          var dx = x - MODES[m][0], dy = y - MODES[m][1];
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < bd) bd = d;
        }
        s += bd;
      }
      return s / P;
    }

    function finish() {
      var dist = meanModeDist();
      var drift = rerunDrift();
      setDist(dist, Math.max(0, 1 - (dist - IDEAL) / 1.0));
      setRepro(drift, Math.min(1, drift / 0.6));
      setNfe(steps, steps / 120);

      var quality = dist < IDEAL * 1.25
        ? "The samples have landed on the modes (mean distance "
          + dist.toFixed(3) + " against an ideal " + IDEAL.toFixed(3) + "). "
        : "Only " + steps + " steps, so the samples are still smeared between "
          + "the modes (mean distance " + dist.toFixed(3) + " against an ideal "
          + IDEAL.toFixed(3) + "). ";
      var determinism = eta < 0.005
        ? "Re-running from the same starting noise gives drift "
          + drift.toFixed(3) + " — bit-identical. At η = 0 the sampler is a "
          + "deterministic map, which is what lets you interpolate DDIM latents."
        : "Re-running from the same starting noise drifts by " + drift.toFixed(3)
          + ". With η > 0 fresh noise enters at every step, so the latent no "
          + "longer determines the sample.";
      note.textContent = quality + determinism;
    }

    // --- drawing ------------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function refresh() {
      var w = stage.clientWidth || 260;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = w + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var bg = cssVar(mount, "--card-bg", "#fff");
      var accent = cssVar(mount, "--accent", "#ba3925");
      var border = cssVar(mount, "--border", "#e2e2dd");

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, w);

      var X = function (v) { return w / 2 + (w / 2) * v / VIEW; };

      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      for (var m = 0; m < NMODES; m++) {
        ctx.beginPath();
        ctx.arc(X(MODES[m][0]), X(MODES[m][1]),
          (w / 2) * (2 * MODE_SD) / VIEW, 0, 6.284);
        ctx.stroke();
      }

      ctx.fillStyle = rgba(accent, 0.45);
      for (var i = 0; i < P; i++) {
        ctx.beginPath();
        ctx.arc(X(xs[2 * i]), X(xs[2 * i + 1]), 1.5, 0, 6.284);
        ctx.fill();
      }

      if (running) {
        note.textContent = "Sampling… step " + (seq.length - cursor - 1)
          + " of " + seq.length + ".";
        setNfe(steps, steps / 120);
      }
    }

    restart(false);
    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="diffusion"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

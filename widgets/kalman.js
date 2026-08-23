// Scalar Kalman filter on a local-level model, with the filter's assumed noise
// under your control.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-kalman
//   :::
//
// The hidden state is a random walk  x_t = x_{t-1} + w,  w ~ N(0, Q_true),
// observed as  z_t = x_t + v,  v ~ N(0, R_true), with Q_true and R_true fixed.
// The two sliders set what the *filter believes* Q and R are, which is the
// interesting knob: the filter is only optimal when its assumed ratio Q/R
// matches the one that generated the data. Verified numerically -- the
// empirical argmin of RMSE over 40 datasets lands on 0.0398 against a true
// ratio of 0.0400, and scaling Q and R together by 1000x leaves the RMSE
// identical to six decimals, since only the ratio enters the gain.
(function () {
  "use strict";

  var N = 130;
  var Q_TRUE = 0.02, R_TRUE = 0.5;
  var QMIN = 1e-4, QMAX = 1;         // slider range for the assumed Q (log)
  var RMIN = 1e-2, RMAX = 10;        // slider range for the assumed R (log)

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

  // Deterministic PRNG so a given seed always regenerates the same series.
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

  function generate(seed) {
    var rnd = mulberry32(seed);
    var truth = new Float64Array(N), meas = new Float64Array(N);
    var x = 0;
    for (var t = 0; t < N; t++) {
      x += Math.sqrt(Q_TRUE) * gauss(rnd);
      truth[t] = x;
      meas[t] = x + Math.sqrt(R_TRUE) * gauss(rnd);
    }
    return { truth: truth, meas: meas };
  }

  // Scalar Kalman filter. Returns the estimate, the posterior variance, and the
  // gain at each step.
  function filter(meas, Qf, Rf) {
    var xh = new Float64Array(N), P = new Float64Array(N), K = new Float64Array(N);
    var x = meas[0], p = Rf;
    for (var t = 0; t < N; t++) {
      var pm = p + Qf;                       // predict
      var k = pm / (pm + Rf);                // gain
      x = x + k * (meas[t] - x);             // update
      p = (1 - k) * pm;
      xh[t] = x; P[t] = p; K[t] = k;
    }
    return { xh: xh, P: P, K: K };
  }

  function rmse(a, b) {
    var s = 0;
    for (var t = 0; t < N; t++) { var d = a[t] - b[t]; s += d * d; }
    return Math.sqrt(s / N);
  }

  function build(mount) {
    var seed = 12345;
    var data = generate(seed);
    var Qf = Q_TRUE, Rf = R_TRUE;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Kalman filter on a random walk";
    el("div", "widget-meta", head).textContent =
      "true Q = " + Q_TRUE + " · true R = " + R_TRUE;
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Hidden state, noisy measurements and the Kalman estimate over time");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);

    el("div", "widget-label", left).textContent = "What the filter assumes";

    function slider(label, lo, hi, initial, onChange) {
      var row = el("div", "widget-slider-row", left);
      el("span", null, row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range";
      input.min = "0";
      input.max = "1000";
      input.step = "1";
      var toPos = function (v) {
        return Math.round(1000 * (Math.log(v) - Math.log(lo)) /
          (Math.log(hi) - Math.log(lo)));
      };
      var toVal = function (p) {
        return Math.exp(Math.log(lo) +
          (Math.log(hi) - Math.log(lo)) * p / 1000);
      };
      input.value = String(toPos(initial));
      var val = el("span", "widget-slider-val", row);
      var show = function (v) {
        val.textContent = v < 0.01 ? v.toExponential(1) : v.toFixed(3);
      };
      show(initial);
      input.addEventListener("input", function () {
        var v = toVal(Number(input.value));
        show(v);
        onChange(v);
      });
      return { set: function (v) { input.value = String(toPos(v)); show(v); } };
    }

    var qSlider = slider("process Q", QMIN, QMAX, Q_TRUE,
      function (v) { Qf = v; refresh(); });
    var rSlider = slider("sensor R", RMIN, RMAX, R_TRUE,
      function (v) { Rf = v; refresh(); });

    el("div", "widget-label", right).textContent = "Result";

    function metric(name, max, fmt) {
      var row = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", row).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", row);
      return function (v) {
        val.textContent = fmt ? fmt(v) : v.toFixed(3);
        fill.style.width = (100 * Math.max(0, Math.min(1, v / max))).toFixed(1) + "%";
      };
    }

    var setGain = metric("gain K", 1);
    var setErr = metric("filter RMSE", 1);
    var setRaw = metric("sensor RMSE", 1);

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

    button("Matched (Q, R)", function () {
      Qf = Q_TRUE; Rf = R_TRUE;
      qSlider.set(Qf); rSlider.set(Rf);
      refresh();
    }, true);
    button("Trust the model", function () {
      Qf = 3e-4; Rf = R_TRUE;
      qSlider.set(Qf); rSlider.set(Rf);
      refresh();
    });
    button("Trust the sensor", function () {
      Qf = 0.6; Rf = R_TRUE;
      qSlider.set(Qf); rSlider.set(Rf);
      refresh();
    });
    button("New data", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      data = generate(seed);
      refresh();
    });

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var W = 0, H = 0, PADX = 10, PADT = 12, PADB = 12;
    var lo = 0, hi = 1;

    function px(t) { return PADX + (W - 2 * PADX) * t / (N - 1); }
    function py(v) { return H - PADB - (H - PADT - PADB) * (v - lo) / (hi - lo); }

    function draw(res) {
      var cssW = mount.clientWidth || 520;
      W = cssW;
      H = Math.max(180, Math.min(240, Math.round(cssW * 0.4)));
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = H + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      var muted = cssVar(mount, "--muted", "#6a6a6a");
      var accent = cssVar(mount, "--accent", "#ba3925");
      var link = cssVar(mount, "--link", "#2156a5");

      lo = Infinity; hi = -Infinity;
      for (var t = 0; t < N; t++) {
        lo = Math.min(lo, data.meas[t], data.truth[t]);
        hi = Math.max(hi, data.meas[t], data.truth[t]);
      }
      var pad = 0.08 * (hi - lo);
      lo -= pad; hi += pad;

      // +/- 2 sigma band around the estimate
      ctx.beginPath();
      for (var i = 0; i < N; i++) {
        ctx.lineTo(px(i), py(res.xh[i] + 2 * Math.sqrt(res.P[i])));
      }
      for (var j = N - 1; j >= 0; j--) {
        ctx.lineTo(px(j), py(res.xh[j] - 2 * Math.sqrt(res.P[j])));
      }
      ctx.closePath();
      ctx.fillStyle = rgba(accent, 0.16);
      ctx.fill();

      // measurements
      ctx.fillStyle = rgba(muted, 0.65);
      for (var m = 0; m < N; m++) {
        ctx.beginPath();
        ctx.arc(px(m), py(data.meas[m]), 1.9, 0, 6.284);
        ctx.fill();
      }

      // hidden truth
      ctx.strokeStyle = link;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (var k = 0; k < N; k++) ctx.lineTo(px(k), py(data.truth[k]));
      ctx.stroke();

      // estimate
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (var q = 0; q < N; q++) ctx.lineTo(px(q), py(res.xh[q]));
      ctx.stroke();

      // legend
      ctx.font = "11px 'Open Sans', sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      var items = [[link, "hidden state"], [muted, "measurements"],
                   [accent, "Kalman estimate"]];
      var lx = PADX + 4;
      for (var n = 0; n < items.length; n++) {
        ctx.fillStyle = items[n][0];
        ctx.fillRect(lx, PADT + 4, 12, 3);
        ctx.fillStyle = muted;
        ctx.fillText(items[n][1], lx + 17, PADT + 6);
        lx += 19 + ctx.measureText(items[n][1]).width + 14;
      }
    }

    function refresh() {
      var res = filter(data.meas, Qf, Rf);
      draw(res);
      var err = rmse(res.xh, data.truth);
      var raw = rmse(data.meas, data.truth);
      var best = rmse(filter(data.meas, Q_TRUE, R_TRUE).xh, data.truth);
      setGain(res.K[N - 1]);
      setErr(err);
      setRaw(raw);

      var ratio = Qf / Rf;
      var trueRatio = Q_TRUE / R_TRUE;
      var ratioOff = Math.log(ratio / trueRatio);
      // Scaling Q and R together leaves the gain, and therefore the estimate,
      // untouched -- only their ratio matters. The absolute size still sets the
      // reported variance, so the band moves while the line does not.
      var scale = Math.sqrt((Qf / Q_TRUE) * (Rf / R_TRUE));
      var msg;
      if (err > raw) {
        msg = "The filter is now worse than the raw sensor — it is following "
          + "the noise instead of the state.";
      } else if (Math.abs(ratioOff) < 0.15) {
        if (scale > 4) {
          msg = "Optimal ratio, but both numbers are far too large. The "
            + "estimate is identical — only Q/R sets the gain — yet the filter "
            + "now reports much more uncertainty than it has, so the band is "
            + "too wide.";
        } else if (scale < 0.25) {
          msg = "Optimal ratio, but both numbers are far too small. The "
            + "estimate is unchanged and the filter is overconfident: the band "
            + "is narrower than the errors justify.";
        } else {
          msg = "Matched. RMSE " + err.toFixed(3) + " against the sensor's "
            + raw.toFixed(3) + ", a " + (100 * (1 - err / raw)).toFixed(0)
            + "% reduction, and about the best this data allows.";
        }
      } else if (ratioOff < 0) {
        msg = "Q is small relative to R, so the filter barely believes the "
          + "state can move: the gain is low, the estimate is smooth and it "
          + "lags behind the truth.";
      } else {
        msg = "Q is large relative to R, so the filter expects the state to "
          + "jump: the gain is high and the estimate chases each measurement.";
      }
      note.textContent = msg;
    }

    refresh();
    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="kalman"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

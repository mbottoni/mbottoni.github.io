// Flow matching in 2-D: the same Gaussian-to-two-moons transport under three
// probability paths, with the marginal velocity field computed exactly.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-flow-matching
//   :::
//
// Nothing here is trained: every velocity field is evaluated in closed form.
// The target is a finite sample {b_j} (K points), the source is N(0, I).
//
// (a) Independent coupling, straight path. Conditioning on x_1 = b_j,
//       p_t(x | b_j) = N(x; t b_j, (1 - t)^2 I),  u_t(x | b_j) = (b_j - x) / (1 - t),
//     so the marginal velocity is the posterior-weighted average
//       u_t(x) = sum_j softmax_j( -|x - t b_j|^2 / 2(1 - t)^2 ) (b_j - x) / (1 - t).
//     This is exact: it is the field a perfectly trained flow-matching network
//     would learn for this data set.
// (c) VP / trigonometric path x_t = sin(pi t/2) x_1 + cos(pi t/2) x_0, again
//     with independent coupling: same construction with mean alpha_t b_j and
//     std beta_t, u_t(x | b_j) = alpha' b_j + (beta'/beta)(x - alpha b_j).
// (b) Minibatch OT. The source batch a_i ~ N(0, I) and a shuffled data batch
//     are paired by an exact assignment (Hungarian algorithm, squared-distance
//     cost) inside each minibatch of size B; B = K is exact OT on the whole
//     sample. The marginal velocity of a minibatch-OT coupling has no closed
//     form for a continuous source, so here the coupling is represented by its
//     K pairs, each widened into a Gaussian tube
//       p_t(x | i) = N(x; (1 - t) a_i + t b_i, (H (1 - t))^2 I)
//     (a kernel estimate of the coupling with bandwidth H at t = 0, collapsing
//     onto the data at t = 1). Its marginal velocity is then exact:
//       u_t(x) = sum_i softmax_i(...) [ (b_i - a_i) - (x - mu_i(t)) / (1 - t) ].
//
// Particles start from exact samples of each construction's p_0; the
// reference trajectories are RK4 with 120 steps, and Euler with n steps is
// compared against them.
(function () {
  "use strict";

  var K = 256;                 // pairs (and particles)
  var H = 0.15;                // kernel bandwidth of the OT pair tubes at t = 0
  var T_EPS = 1e-4;            // keep 1 - t and beta_t away from zero
  var REF_STEPS = 120;         // RK4 steps for the reference ODE solve
  var VIEW = 2.7;
  var TRAILS = 90;             // trajectories drawn
  var BATCHES = [2, 4, 8, 16, 32, 64, 128, 256];

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

  function shuffle(arr, rnd) {
    for (var k = arr.length - 1; k > 0; k--) {
      var j = Math.floor(rnd() * (k + 1));
      var t = arr[k]; arr[k] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // --- data -------------------------------------------------------------------
  function moons(n, rnd) {
    var out = new Float64Array(n * 2);
    for (var i = 0; i < n; i++) {
      var th = Math.PI * rnd(), x, y;
      if (i % 2 === 0) { x = Math.cos(th); y = Math.sin(th); }
      else { x = 1 - Math.cos(th); y = 0.5 - Math.sin(th); }
      x += 0.07 * gaussFrom(rnd); y += 0.07 * gaussFrom(rnd);
      out[2 * i] = 1.55 * (x - 0.5);
      out[2 * i + 1] = 1.55 * (y - 0.25);
    }
    return out;
  }

  function checker(n, rnd) {
    var out = new Float64Array(n * 2);
    for (var i = 0; i < n; i++) {
      var x, y;
      do { x = rnd() * 4 - 2; y = rnd() * 4 - 2; }
      while ((Math.floor(x + 2) + Math.floor(y + 2)) % 2 !== 0);
      out[2 * i] = x * 1.1; out[2 * i + 1] = y * 1.1;
    }
    return out;
  }

  // --- exact assignment (Hungarian / Kuhn-Munkres, O(n^3)) ---------------------
  // cost is n x n row-major; returns match[row] = column.
  function hungarian(cost, n) {
    var INF = Infinity;
    var u = new Float64Array(n + 1), v = new Float64Array(n + 1);
    var p = new Int32Array(n + 1), way = new Int32Array(n + 1);
    var minv = new Float64Array(n + 1), used = new Uint8Array(n + 1);
    for (var i = 1; i <= n; i++) {
      p[0] = i;
      var j0 = 0;
      minv.fill(INF); used.fill(0);
      do {
        used[j0] = 1;
        var i0 = p[j0], delta = INF, j1 = 0;
        for (var j = 1; j <= n; j++) {
          if (used[j]) continue;
          var cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (j = 0; j <= n; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
          else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { var jj = way[j0]; p[j0] = p[jj]; j0 = jj; } while (j0);
    }
    var match = new Int32Array(n);
    for (j = 1; j <= n; j++) match[p[j] - 1] = j - 1;
    return match;
  }

  // Build the coupling: returns B-ordered target points so that pair i is
  // (src[i], tgt[i]).
  function couple(src, data, mode, batch, rnd) {
    var tgt = new Float64Array(K * 2);
    var perm = shuffle(Array.from({ length: K }, function (_, i) { return i; }), rnd);
    if (mode !== "ot") {
      for (var i = 0; i < K; i++) {
        tgt[2 * i] = data[2 * perm[i]]; tgt[2 * i + 1] = data[2 * perm[i] + 1];
      }
      return tgt;
    }
    // Minibatch OT: source index i and data index perm[i] are both random,
    // so slicing [s, s + batch) gives independent minibatches of each.
    var cost = new Float64Array(batch * batch);
    for (var s = 0; s < K; s += batch) {
      for (var r = 0; r < batch; r++) {
        var ax = src[2 * (s + r)], ay = src[2 * (s + r) + 1];
        for (var c = 0; c < batch; c++) {
          var bi = perm[s + c];
          var dx = ax - data[2 * bi], dy = ay - data[2 * bi + 1];
          cost[r * batch + c] = dx * dx + dy * dy;
        }
      }
      var m = hungarian(cost, batch);
      for (r = 0; r < batch; r++) {
        var b = perm[s + m[r]];
        tgt[2 * (s + r)] = data[2 * b]; tgt[2 * (s + r) + 1] = data[2 * b + 1];
      }
    }
    return tgt;
  }

  // --- the marginal velocity fields ------------------------------------------
  // Each returns field(x, y, t, out). Weights use log-sum-exp; terms more than
  // e^-40 below the best are dropped.
  function makeField(path, data, src, tgt) {
    var logw = new Float64Array(K);
    var mu = new Float64Array(K * 2), du = new Float64Array(K * 2);
    var lastT = -1, m, sd, a, ad, b, bd;

    // per-time constants: component means mu_j and std sd, plus the pair
    // velocities du_i for the OT tubes
    function prep(t) {
      if (t === lastT) return;
      lastT = t;
      var i;
      if (path === "linear") {
        sd = Math.max(1 - t, T_EPS);
        for (i = 0; i < 2 * K; i++) mu[i] = t * data[i];
      } else if (path === "vp") {
        var h = Math.PI / 2;
        a = Math.sin(h * t); ad = h * Math.cos(h * t);
        b = Math.max(Math.cos(h * t), T_EPS); bd = -h * Math.sin(h * t);
        sd = b;
        for (i = 0; i < 2 * K; i++) mu[i] = a * data[i];
      } else {
        m = Math.max(1 - t, T_EPS);
        sd = H * m;
        for (i = 0; i < 2 * K; i++) {
          mu[i] = (1 - t) * src[i] + t * tgt[i];
          du[i] = tgt[i] - src[i];
        }
      }
    }

    return function (x, y, t, out) {
      prep(t);
      var inv = 1 / (2 * sd * sd), best = -Infinity, i;
      for (i = 0; i < K; i++) {
        var dx = x - mu[2 * i], dy = y - mu[2 * i + 1];
        var l = -(dx * dx + dy * dy) * inv;
        logw[i] = l;
        if (l > best) best = l;
      }
      var sum = 0, vx = 0, vy = 0, cx, cy;
      for (i = 0; i < K; i++) {
        var d = logw[i] - best;
        if (d < -40) continue;
        var w = Math.exp(d);
        if (path === "linear") {
          cx = (data[2 * i] - x) / sd; cy = (data[2 * i + 1] - y) / sd;
        } else if (path === "vp") {
          cx = ad * data[2 * i] + (bd / b) * (x - mu[2 * i]);
          cy = ad * data[2 * i + 1] + (bd / b) * (y - mu[2 * i + 1]);
        } else {
          cx = du[2 * i] - (x - mu[2 * i]) / m;
          cy = du[2 * i + 1] - (y - mu[2 * i + 1]) / m;
        }
        sum += w; vx += w * cx; vy += w * cy;
      }
      out[0] = vx / sum; out[1] = vy / sum;
    };
  }

  // RK4 over all particles; returns (steps + 1) * K * 2 positions.
  function solveRK4(field, x0, steps) {
    var tr = new Float64Array((steps + 1) * K * 2);
    tr.set(x0);
    var h = 1 / steps, k1 = [0, 0], k2 = [0, 0], k3 = [0, 0], k4 = [0, 0];
    // evaluate stage-by-stage so every particle shares one prep(t)
    var x = new Float64Array(x0), s1 = new Float64Array(K * 2),
      s2 = new Float64Array(K * 2), s3 = new Float64Array(K * 2),
      s4 = new Float64Array(K * 2);
    for (var n = 0; n < steps; n++) {
      var t = n * h, i;
      for (i = 0; i < K; i++) {
        field(x[2 * i], x[2 * i + 1], t, k1);
        s1[2 * i] = k1[0]; s1[2 * i + 1] = k1[1];
      }
      for (i = 0; i < K; i++) {
        field(x[2 * i] + 0.5 * h * s1[2 * i], x[2 * i + 1] + 0.5 * h * s1[2 * i + 1],
          t + 0.5 * h, k2);
        s2[2 * i] = k2[0]; s2[2 * i + 1] = k2[1];
      }
      for (i = 0; i < K; i++) {
        field(x[2 * i] + 0.5 * h * s2[2 * i], x[2 * i + 1] + 0.5 * h * s2[2 * i + 1],
          t + 0.5 * h, k3);
        s3[2 * i] = k3[0]; s3[2 * i + 1] = k3[1];
      }
      for (i = 0; i < K; i++) {
        field(x[2 * i] + h * s3[2 * i], x[2 * i + 1] + h * s3[2 * i + 1], t + h, k4);
        s4[2 * i] = k4[0]; s4[2 * i + 1] = k4[1];
      }
      for (i = 0; i < 2 * K; i++) {
        x[i] += h / 6 * (s1[i] + 2 * s2[i] + 2 * s3[i] + s4[i]);
      }
      tr.set(x, (n + 1) * K * 2);
    }
    return tr;
  }

  function solveEuler(field, x0, steps) {
    var tr = new Float64Array((steps + 1) * K * 2);
    tr.set(x0);
    var x = new Float64Array(x0), v = [0, 0], h = 1 / steps;
    for (var n = 0; n < steps; n++) {
      var t = n * h;
      // all velocities at the old positions first (a proper Euler step)
      var nx = new Float64Array(2 * K);
      for (var i = 0; i < K; i++) {
        field(x[2 * i], x[2 * i + 1], t, v);
        nx[2 * i] = x[2 * i] + h * v[0];
        nx[2 * i + 1] = x[2 * i + 1] + h * v[1];
      }
      x = nx;
      tr.set(x, (n + 1) * K * 2);
    }
    return tr;
  }

  // Mean chord / arc length over trajectories: 1 means perfectly straight.
  function straightness(tr, steps) {
    var s = 0;
    for (var i = 0; i < K; i++) {
      var arc = 0;
      for (var n = 0; n < steps; n++) {
        var a = (n * K + i) * 2, b = ((n + 1) * K + i) * 2;
        arc += Math.hypot(tr[b] - tr[a], tr[b + 1] - tr[a + 1]);
      }
      var e = (steps * K + i) * 2;
      var chord = Math.hypot(tr[e] - tr[2 * i], tr[e + 1] - tr[2 * i + 1]);
      s += arc > 1e-12 ? chord / arc : 1;
    }
    return s / K;
  }

  function endpointErr(a, as, b, bs) {
    var s = 0, oa = as * K * 2, ob = bs * K * 2;
    for (var i = 0; i < K; i++) {
      s += Math.hypot(a[oa + 2 * i] - b[ob + 2 * i], a[oa + 2 * i + 1] - b[ob + 2 * i + 1]);
    }
    return s / K;
  }

  // Mean distance from each endpoint to the nearest target point.
  function distToData(tr, steps, data) {
    var s = 0, o = steps * K * 2;
    for (var i = 0; i < K; i++) {
      var x = tr[o + 2 * i], y = tr[o + 2 * i + 1], bd = Infinity;
      for (var j = 0; j < K; j++) {
        var dx = x - data[2 * j], dy = y - data[2 * j + 1];
        var d = dx * dx + dy * dy;
        if (d < bd) bd = d;
      }
      s += Math.sqrt(bd);
    }
    return s / K;
  }

  // --- DOM helpers ---------------------------------------------------------------
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
    if (isNaN(n)) return hex;
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," +
      (n & 255) + "," + alpha + ")";
  }

  var PATHS = [
    { id: "ind", label: "Independent", path: "linear" },
    { id: "ot", label: "Minibatch OT", path: "pairs" },
    { id: "vp", label: "VP diffusion", path: "vp" },
  ];

  function build(mount) {
    var mode = "ind", batchIdx = BATCHES.length - 1, eulerSteps = 3;
    var target = "moons", seed = 3;
    var t = 1, playing = false, raf = null, showPairs = false;
    var src, noise, x0, data, ref, eul, field;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Flow matching: paths and couplings";
    el("div", "widget-meta", head).textContent =
      K + " pairs · exact marginal velocity";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    stage.style.flexBasis = "54%";
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Particles flowing from a Gaussian to the target along the marginal ODE");
    var side = el("div", "widget-side", body);
    side.style.flexBasis = "46%";

    el("div", "widget-label", side).textContent = "Probability path";
    var chipRow = el("div", "widget-chips", side);
    chipRow.style.flexWrap = "wrap";
    var chips = PATHS.map(function (p) {
      var chip = el("button", "widget-chip", chipRow);
      chip.type = "button";
      chip.textContent = p.label;
      chip.style.padding = "0.45rem 0.6rem";
      chip.addEventListener("click", function () { mode = p.id; recompute(); });
      return chip;
    });

    function slider(label, lo, hi, step, initial, onInput) {
      var row = el("div", "widget-slider-row", side);
      el("span", null, row).textContent = label;
      var input = el("input", "widget-slider", row);
      input.type = "range";
      input.min = String(lo); input.max = String(hi); input.step = String(step);
      input.value = String(initial);
      input.setAttribute("aria-label", label);
      var val = el("span", "widget-slider-val", row);
      input.addEventListener("input", function () { onInput(Number(input.value)); });
      return { input: input, val: val, row: row };
    }

    var tS = slider("time", 0, 1, 0.005, t, function (v) {
      stopPlay(); t = v; draw();
    });
    var bS = slider("OT batch", 0, BATCHES.length - 1, 1, batchIdx, function (v) {
      batchIdx = v; recompute();
    });
    var eS = slider("Euler n", 1, 20, 1, eulerSteps, function (v) {
      eulerSteps = v; runEuler(); readout(); draw();
    });

    el("div", "widget-label", side).textContent = "Measured";
    function metric(name) {
      var r = el("div", "widget-row widget-row--wide", side);
      el("span", "widget-row-name", r).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", r);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", r);
      val.style.width = "4.2em";
      return function (text, bar) {
        fill.style.width = (100 * Math.max(0, Math.min(1, bar))).toFixed(1) + "%";
        val.textContent = text;
      };
    }
    var setStraight = metric("straightness");
    var setErr = metric("Euler error");
    var setEulerData = metric("Euler→data");
    var setRefData = metric("ODE→data");

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
    var playBtn = button("Play", function () {
      if (playing) { stopPlay(); return; }
      play();
    }, true);
    var pairsBtn = button("Show pairs", function () {
      showPairs = !showPairs;
      pairsBtn.textContent = showPairs ? "Hide pairs" : "Show pairs";
      draw();
    });
    var targetBtn = button("Target: checkerboard", function () {
      target = target === "moons" ? "checker" : "moons";
      targetBtn.textContent = target === "moons"
        ? "Target: checkerboard" : "Target: two moons";
      newData();
      recompute();
    });
    button("Resample", function () {
      seed = (seed * 1103515245 + 12345) >>> 0;
      newData();
      recompute();
    });

    // --- computation ------------------------------------------------------------
    var pairsTgt;

    function newData() {
      var rnd = mulberry32(seed);
      src = new Float64Array(K * 2);
      for (var i = 0; i < 2 * K; i++) src[i] = gaussFrom(rnd);
      noise = new Float64Array(K * 2);
      for (i = 0; i < 2 * K; i++) noise[i] = gaussFrom(rnd);
      data = target === "moons" ? moons(K, rnd) : checker(K, rnd);
    }

    function recompute() {
      chips.forEach(function (c, k) {
        if (PATHS[k].id === mode) c.setAttribute("data-active", "");
        else c.removeAttribute("data-active");
      });
      bS.val.textContent = String(BATCHES[batchIdx]);
      bS.input.disabled = mode !== "ot";
      bS.row.style.opacity = mode === "ot" ? "1" : "0.45";
      var p = PATHS.filter(function (q) { return q.id === mode; })[0];
      // the coupling RNG is fixed per seed, so switching paths re-uses the same
      // random pairing and only the OT step changes it
      pairsTgt = couple(src, data, mode, BATCHES[batchIdx], mulberry32(seed + 17));
      field = makeField(p.path, data, src, pairsTgt);
      // start points: exact samples of each construction's p_0 -- N(0, I) for
      // the exact paths, a_i + H z for the OT pair tubes
      x0 = new Float64Array(K * 2);
      for (var i = 0; i < 2 * K; i++) {
        x0[i] = mode === "ot" ? src[i] + H * noise[i] : src[i];
      }
      ref = solveRK4(field, x0, REF_STEPS);
      runEuler();
      readout();
      draw();
    }

    function runEuler() {
      eS.val.textContent = String(eulerSteps);
      eul = solveEuler(field, x0, eulerSteps);
    }

    function readout() {
      var st = straightness(ref, REF_STEPS);
      var err = endpointErr(eul, eulerSteps, ref, REF_STEPS);
      var de = distToData(eul, eulerSteps, data);
      var dr = distToData(ref, REF_STEPS, data);
      setStraight(st.toFixed(3), st);
      setErr(err.toFixed(3), err / 1.0);
      setEulerData(de.toFixed(3), de / 1.0);
      setRefData(dr.toFixed(3), dr / 1.0);

      var what;
      if (mode === "ind") {
        what = "Each conditional path is a straight line, but with random pairs "
          + "the lines cross. Where they cross, the marginal velocity averages "
          + "over the pairs that could be there, so the particles' actual paths "
          + "bend.";
      } else if (mode === "ot") {
        what = BATCHES[batchIdx] === K
          ? "Exact OT on the whole sample: optimal straight-line paths never "
            + "cross, the posterior over pairs stays sharp, and the marginal flow "
            + "is (almost) the conditional one."
          : "OT inside minibatches of " + BATCHES[batchIdx] + ": pairs are only "
            + "locally optimal, so some crossing (and curvature) remains. "
            + "Raise the batch size to straighten the flow.";
      } else {
        what = "The VP path moves along arcs, sin(πt/2)·x₁ + cos(πt/2)·x₀, so "
          + "even a single pair's path is curved, and on top of that the random "
          + "pairing crosses.";
      }
      note.textContent = what + " With " + eulerSteps + " Euler step"
        + (eulerSteps > 1 ? "s" : "") + " the endpoints miss the exact ODE "
        + "solution by " + err.toFixed(3) + " on average.";
    }

    // --- playback ---------------------------------------------------------------
    function play() {
      playing = true;
      playBtn.textContent = "Pause";
      if (t >= 1) t = 0;
      var last = null;
      var tick = function (now) {
        t = Math.min(1, t + (last === null ? 0 : Math.max(0, now - last)) / 2600);
        last = now;
        tS.input.value = String(t);
        draw();
        if (t < 1 && playing) raf = requestAnimationFrame(tick);
        else stopPlay();
      };
      raf = requestAnimationFrame(tick);
    }

    function stopPlay() {
      playing = false;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      playBtn.textContent = "Play";
    }

    // --- drawing ----------------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function draw() {
      tS.val.textContent = t.toFixed(2);
      var cw = stage.clientWidth || 260;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = cw + "px";
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(cw * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var bg = cssVar(mount, "--card-bg", "#fff");
      var border = cssVar(mount, "--border", "#e2e2dd");
      var muted = cssVar(mount, "--muted", "#6a6a6a");
      var accent = cssVar(mount, "--accent", "#ba3925");
      var link = cssVar(mount, "--link", "#2156a5");

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, cw, cw);
      var X = function (v) { return cw / 2 + (cw / 2) * v / VIEW; };
      var Y = function (v) { return cw / 2 - (cw / 2) * v / VIEW; };

      // target sample
      ctx.fillStyle = rgba(link, 0.35);
      for (var j = 0; j < K; j++) {
        ctx.beginPath();
        ctx.arc(X(data[2 * j]), Y(data[2 * j + 1]), 1.8, 0, 6.284);
        ctx.fill();
      }

      // the coupling itself: straight segments a_i -> b_i
      if (showPairs) {
        ctx.strokeStyle = rgba(link, 0.22);
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        for (var i = 0; i < K; i++) {
          ctx.moveTo(X(src[2 * i]), Y(src[2 * i + 1]));
          ctx.lineTo(X(pairsTgt[2 * i]), Y(pairsTgt[2 * i + 1]));
        }
        ctx.stroke();
      }

      // marginal ODE trajectories (the exact flow), up to the current t
      var upto = Math.round(t * REF_STEPS);
      ctx.strokeStyle = rgba(muted, 0.5);
      ctx.lineWidth = 0.8;
      ctx.lineJoin = "round";
      for (i = 0; i < TRAILS; i++) {
        ctx.beginPath();
        for (var n = 0; n <= REF_STEPS; n++) {
          var o = (n * K + i) * 2;
          if (n === 0) ctx.moveTo(X(ref[o]), Y(ref[o + 1]));
          else ctx.lineTo(X(ref[o]), Y(ref[o + 1]));
        }
        ctx.globalAlpha = 0.35;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = rgba(accent, 0.55);
      ctx.lineWidth = 1;
      for (i = 0; i < TRAILS; i++) {
        ctx.beginPath();
        for (n = 0; n <= upto; n++) {
          o = (n * K + i) * 2;
          if (n === 0) ctx.moveTo(X(ref[o]), Y(ref[o + 1]));
          else ctx.lineTo(X(ref[o]), Y(ref[o + 1]));
        }
        ctx.stroke();
      }

      // few-step Euler: polyline for a handful, endpoints as rings at t = 1
      ctx.strokeStyle = rgba(muted, 0.9);
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      for (i = 0; i < 10; i++) {
        ctx.beginPath();
        for (n = 0; n <= eulerSteps; n++) {
          o = (n * K + i) * 2;
          if (n === 0) ctx.moveTo(X(eul[o]), Y(eul[o + 1]));
          else ctx.lineTo(X(eul[o]), Y(eul[o + 1]));
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (t >= 0.999) {
        ctx.strokeStyle = rgba(muted, 0.8);
        var oe = eulerSteps * K * 2;
        for (i = 0; i < K; i++) {
          ctx.beginPath();
          ctx.arc(X(eul[oe + 2 * i]), Y(eul[oe + 2 * i + 1]), 2.4, 0, 6.284);
          ctx.stroke();
        }
      }

      // particles at time t (linear interpolation between solver steps)
      var f = t * REF_STEPS, n0 = Math.min(REF_STEPS - 1, Math.floor(f)), a = f - n0;
      ctx.fillStyle = accent;
      for (i = 0; i < K; i++) {
        var p0 = (n0 * K + i) * 2, p1 = ((n0 + 1) * K + i) * 2;
        var px = ref[p0] + a * (ref[p1] - ref[p0]);
        var py = ref[p0 + 1] + a * (ref[p1 + 1] - ref[p0 + 1]);
        ctx.beginPath();
        ctx.arc(X(px), Y(py), 1.9, 0, 6.284);
        ctx.fill();
      }

      var label = "t = " + t.toFixed(2);
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textBaseline = "top";
      ctx.fillStyle = rgba(bg, 0.85);
      ctx.fillRect(4, 4, ctx.measureText(label).width + 8, 18);
      ctx.fillStyle = muted;
      ctx.fillText(label, 8, 7);
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, cw - 1, cw - 1);
    }

    newData();
    recompute();
    window.addEventListener("resize", draw);
    new MutationObserver(draw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="flow-matching"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

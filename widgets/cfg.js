// Classifier-free guidance on a 2-D toy: drag the guidance scale and watch the
// samples leave the modes the class shares with its neighbours, then leave the
// data altogether.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-cfg
//   :::
//
// There is no trained network. The data is a known mixture of isotropic
// Gaussians -- three classes, three components each -- so under the VP forward
// process x_t = sqrt(abar) x_0 + sqrt(1 - abar) eps every noised density is
// again a mixture,
//   p_t(x | c) = sum_{k in c} w_k N(x; sqrt(abar) mu_k, (abar s^2 + 1 - abar) I),
//   p_t(x)     = sum_c p(c) p_t(x | c),
// and both the conditional score s(x | c) and the unconditional score s(x) are
// exact. Guidance uses exactly the rule from the post,
//   eps~ = (1 + w) eps(x, c) - w eps(x),   eps = -sqrt(1 - abar) * score,
// i.e. s~ = (1 + w) s(x | c) - w s(x) = s(x | c) + w grad log p_t(c | x).
// The sampler is DDIM (eta = 0, the probability-flow ODE discretised) or the
// ancestral DDPM step (eta = 1) on 50 of the 1000 training timesteps.
(function () {
  "use strict";

  var T = 1000;
  var BETA0 = 1e-4, BETA1 = 0.02;
  var STEPS = 50;
  var P = 400;                 // particles
  var TRAILS = 44;             // of which this many draw their trajectory
  var SD = 0.22;               // every component's std
  var VIEW = 3.1;              // plot half-width in data units
  var W_MAX = 10;

  var ABAR = new Float64Array(T);
  (function () {
    var prod = 1;
    for (var t = 0; t < T; t++) {
      prod *= 1 - (BETA0 + (BETA1 - BETA0) * t / (T - 1));
      ABAR[t] = prod;
    }
  })();

  // Three classes at 120 degrees. Each has one heavy component near the centre,
  // where it overlaps the other classes' inner components, and two lighter
  // ones further out where the class is unambiguous.
  var CLASS_NAMES = ["A", "B", "C"];
  var COMPS = [];              // {mu:[x,y], w: weight within class, c: class}
  (function () {
    for (var c = 0; c < 3; c++) {
      var th = Math.PI / 2 + c * 2 * Math.PI / 3;
      var parts = [
        { r: 0.55, d: 0, w: 0.44 },
        { r: 1.75, d: -0.42, w: 0.28 },
        { r: 1.75, d: 0.42, w: 0.28 },
      ];
      for (var k = 0; k < parts.length; k++) {
        var a = th + parts[k].d;
        COMPS.push({
          mu: [parts[k].r * Math.cos(a), parts[k].r * Math.sin(a)],
          w: parts[k].w, c: c, inner: k === 0,
        });
      }
    }
  })();
  var K = COMPS.length;

  // Total std of p(x | c), for the spread readout: sqrt(E|x - E x|^2).
  function trueSpread(c) {
    var mx = 0, my = 0, m2 = 0;
    for (var k = 0; k < K; k++) {
      if (COMPS[k].c !== c) continue;
      var w = COMPS[k].w, mu = COMPS[k].mu;
      mx += w * mu[0]; my += w * mu[1];
      m2 += w * (mu[0] * mu[0] + mu[1] * mu[1] + 2 * SD * SD);
    }
    return Math.sqrt(m2 - mx * mx - my * my);
  }

  // Score of the noised mixture, restricted to class `c` (or all classes if
  // c < 0, with p(c) = 1/3). All components share one variance, so the
  // normaliser cancels in the responsibilities.
  var LOGW = new Float64Array(K);
  function score(x, y, abar, c, out) {
    var sa = Math.sqrt(abar);
    var v = abar * SD * SD + (1 - abar);
    var best = -Infinity, k;
    for (k = 0; k < K; k++) {
      if (c >= 0 && COMPS[k].c !== c) { LOGW[k] = -Infinity; continue; }
      var dx = x - sa * COMPS[k].mu[0], dy = y - sa * COMPS[k].mu[1];
      LOGW[k] = Math.log(COMPS[k].w) - (dx * dx + dy * dy) / (2 * v);
      if (LOGW[k] > best) best = LOGW[k];
    }
    var sum = 0, sx = 0, sy = 0;
    for (k = 0; k < K; k++) {
      if (LOGW[k] === -Infinity) continue;
      var r = Math.exp(LOGW[k] - best);
      sum += r;
      sx += r * (sa * COMPS[k].mu[0] - x);
      sy += r * (sa * COMPS[k].mu[1] - y);
    }
    out[0] = sx / (sum * v);
    out[1] = sy / (sum * v);
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

  // Run the whole reverse process; returns trajectories laid out as
  // traj[(step * P + i) * 2 + {0,1}], step 0 = pure noise, STEPS = final.
  var SC = [0, 0], SU = [0, 0];
  function sample(x0, c, w, eta, seed) {
    var seq = [];
    for (var s = 0; s < STEPS; s++) seq.push(Math.floor(s * T / STEPS));
    var traj = new Float64Array((STEPS + 1) * P * 2);
    traj.set(x0);
    var rnd = mulberry32(seed);
    for (var j = 0; j < STEPS; j++) {
      var cur = STEPS - 1 - j;                 // walk the subsequence backwards
      var t = seq[cur];
      var abar = ABAR[t];
      var abarNext = cur > 0 ? ABAR[seq[cur - 1]] : 1;
      var sig = eta * Math.sqrt(
        (1 - abar / abarNext) * (1 - abarNext) / (1 - abar));
      var dirCoef = Math.sqrt(Math.max(0, 1 - abarNext - sig * sig));
      var sa = Math.sqrt(abar), sNext = Math.sqrt(abarNext);
      var s1ma = Math.sqrt(1 - abar);
      var src = j * P * 2, dst = (j + 1) * P * 2;
      for (var i = 0; i < P; i++) {
        var x = traj[src + 2 * i], y = traj[src + 2 * i + 1];
        score(x, y, abar, c, SC);
        score(x, y, abar, -1, SU);
        // guided score, then the equivalent guided eps
        var gx = (1 + w) * SC[0] - w * SU[0];
        var gy = (1 + w) * SC[1] - w * SU[1];
        var ex = -s1ma * gx, ey = -s1ma * gy;
        var x0x = (x - s1ma * ex) / sa, x0y = (y - s1ma * ey) / sa;
        var nx = sNext * x0x + dirCoef * ex;
        var ny = sNext * x0y + dirCoef * ey;
        if (sig > 0 && cur > 0) {
          nx += sig * gaussFrom(rnd);
          ny += sig * gaussFrom(rnd);
        }
        traj[dst + 2 * i] = nx;
        traj[dst + 2 * i + 1] = ny;
      }
    }
    return traj;
  }

  // Statistics of the final samples against the clean data distribution.
  function stats(traj, c) {
    var base = STEPS * P * 2;
    var counts = [0, 0, 0], off = 0, wrong = 0, mx = 0, my = 0, m2 = 0;
    var cn = [0, 0, 0], cx = [0, 0, 0], cy = [0, 0, 0], c2 = [0, 0, 0];
    var own = [];
    for (var k = 0; k < K; k++) if (COMPS[k].c === c) own.push(k);
    var post = [0, 0, 0];
    for (var i = 0; i < P; i++) {
      var x = traj[base + 2 * i], y = traj[base + 2 * i + 1];
      mx += x; my += y; m2 += x * x + y * y;
      // nearest own component, in units of the component std
      var bd = Infinity, bj = 0;
      for (var j = 0; j < own.length; j++) {
        var mu = COMPS[own[j]].mu;
        var d = Math.hypot(x - mu[0], y - mu[1]) / SD;
        if (d < bd) { bd = d; bj = j; }
      }
      cn[bj]++; cx[bj] += x; cy[bj] += y; c2[bj] += x * x + y * y;
      if (bd > 3) off++; else counts[bj]++;
      // Bayes classifier p(c | x) on clean data
      post[0] = post[1] = post[2] = 0;
      var best = -Infinity;
      for (k = 0; k < K; k++) {
        var dx = x - COMPS[k].mu[0], dy = y - COMPS[k].mu[1];
        var lw = Math.log(COMPS[k].w) - (dx * dx + dy * dy) / (2 * SD * SD);
        if (lw > best) best = lw;
        LOGW[k] = lw;
      }
      for (k = 0; k < K; k++) post[COMPS[k].c] += Math.exp(LOGW[k] - best);
      var arg = 0;
      for (var q = 1; q < 3; q++) if (post[q] > post[arg]) arg = q;
      if (arg !== c) wrong++;
    }
    mx /= P; my /= P;
    var spread = Math.sqrt(Math.max(0, m2 / P - mx * mx - my * my));
    // Within-cluster spread: group samples by nearest own component, take the
    // pooled RMS distance to each group's own centroid, in units of SD (for
    // samples from p(x | c) this is ~1, since E|z|^2 = 2 in 2-D). It measures
    // how tight the clusters are independently of where they sit.
    var ss = 0, dof = 0;
    for (j = 0; j < 3; j++) {
      if (cn[j] < 2) continue;
      ss += c2[j] - (cx[j] * cx[j] + cy[j] * cy[j]) / cn[j];
      dof += 2 * (cn[j] - 1);
    }
    var width = dof ? Math.sqrt(ss / dof) / SD : 0;
    var hit = 0;
    for (j = 0; j < 3; j++) if (counts[j] >= 0.05 * P) hit++;
    return {
      width: width,
      spread: spread, hit: hit, off: off / P, wrong: wrong / P,
      shares: counts.map(function (n) { return n / P; }),
      weights: own.map(function (k) { return COMPS[k].w; }),
    };
  }

  // --- DOM helpers ----------------------------------------------------------
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

  function isDark() {
    return document.documentElement.getAttribute("data-theme") === "dark";
  }

  function build(mount) {
    var cls = 0, w = 3, eta = 0;
    var latentSeed = 7, streamSeed = 11;
    var x0 = new Float64Array(P * 2);
    var traj = null, shown = STEPS, raf = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Classifier-free guidance";
    el("div", "widget-meta", head).textContent =
      P + " samples · exact mixture score";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    stage.style.flexBasis = "52%";
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.style.cursor = "default";
    canvas.setAttribute("aria-label",
      "Guided samples and their trajectories over a three-class Gaussian mixture");
    var side = el("div", "widget-side", body);
    side.style.flexBasis = "48%";

    el("div", "widget-label", side).textContent = "Condition on class";
    var chipRow = el("div", "widget-chips", side);
    var chips = CLASS_NAMES.map(function (name, c) {
      var chip = el("button", "widget-chip", chipRow);
      chip.type = "button";
      chip.title = "Sample class " + name;
      chip.style.flexDirection = "row";
      chip.style.padding = "0.45rem 0.7rem";
      var sw = el("span", null, chip);
      sw.style.cssText = "display:inline-block;width:0.7rem;height:0.7rem;"
        + "border-radius:50%";
      el("span", null, chip).textContent = "class " + name;
      chip.addEventListener("click", function () {
        cls = c; run(true);
      });
      return { chip: chip, sw: sw };
    });

    el("div", "widget-label", side).textContent = "Guidance";
    var wRow = el("div", "widget-slider-row", side);
    el("span", null, wRow).textContent = "scale w";
    var wInput = el("input", "widget-slider", wRow);
    wInput.type = "range"; wInput.min = "0"; wInput.max = String(W_MAX);
    wInput.step = "0.1"; wInput.value = String(w);
    wInput.setAttribute("aria-label", "Guidance scale w");
    var wVal = el("span", "widget-slider-val", wRow);
    wInput.addEventListener("input", function () {
      w = Number(wInput.value);
      run(false);
    });

    el("div", "widget-label", side).textContent = "Final samples";
    function metric(name) {
      var r = el("div", "widget-row widget-row--wide", side);
      el("span", "widget-row-name", r).textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", r);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", r);
      val.style.width = "5.6em";
      val.style.whiteSpace = "nowrap";
      return function (text, bar) {
        fill.style.width = (100 * Math.max(0, Math.min(1, bar))).toFixed(1) + "%";
        val.textContent = text;
      };
    }
    var setSpread = metric("spread");
    var setWidth = metric("mode width");
    var setHit = metric("modes hit");
    var setOff = metric("off support");
    var setWrong = metric("wrong class");

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
    button("Resample", function () {
      latentSeed = (latentSeed * 1103515245 + 12345) >>> 0;
      streamSeed++;
      run(true);
    }, true);
    [0, 1, 3, 10].forEach(function (v) {
      button("w = " + v, function () {
        w = v; wInput.value = String(v); run(true);
      });
    });
    var etaBtn = button("", function () {
      eta = eta ? 0 : 1; run(true);
    });

    // --- running --------------------------------------------------------------
    function newLatent() {
      var rnd = mulberry32(latentSeed);
      for (var i = 0; i < P * 2; i++) x0[i] = gaussFrom(rnd);
    }

    function run(animate) {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      wVal.textContent = w.toFixed(1);
      etaBtn.textContent = eta ? "Sampler: DDPM (η = 1)" : "Sampler: DDIM (η = 0)";
      chips.forEach(function (ch, c) {
        if (c === cls) ch.chip.setAttribute("data-active", "");
        else ch.chip.removeAttribute("data-active");
      });
      traj = sample(x0, cls, w, eta, streamSeed);
      readout();
      if (!animate) { shown = STEPS; draw(); return; }
      shown = 0;
      var start = performance.now();
      var tick = function (now) {
        shown = Math.min(STEPS, Math.floor((now - start) / 22));
        draw();
        if (shown < STEPS) raf = requestAnimationFrame(tick);
        else raf = null;
      };
      raf = requestAnimationFrame(tick);
    }

    function pct(v) { return (100 * v).toFixed(0) + "%"; }

    function readout() {
      var s = stats(traj, cls);
      var ref = trueSpread(cls);
      setSpread(s.spread.toFixed(2) + "/" + ref.toFixed(2),
        s.spread / (2 * ref));
      setWidth(s.width.toFixed(2) + "σ", s.width / 2);
      setHit(s.hit + " / 3", s.hit / 3);
      setOff(pct(s.off), s.off);
      setWrong(pct(s.wrong), s.wrong);

      var shareText = "Shares (inner, outer, outer): "
        + s.shares.map(pct).join(", ") + " against the true "
        + s.weights.map(pct).join(", ") + ". ";
      var story;
      if (w < 0.05) {
        story = "With w = 0 this is plain conditional sampling: the samples "
          + "follow p(x | c), heavy inner mode and all.";
      } else if (s.off > 0.25) {
        story = "At this scale most samples overshoot: the guided field is no "
          + "longer the score of the noised data distribution, and "
          + pct(s.off) + " land more than 3σ from every component of the class.";
      } else if (s.shares[0] < 0.5 * s.weights[0]) {
        story = "Guidance adds w·∇log p(c | x), which pushes samples to where the "
          + "class is unambiguous. The inner mode, shared with the other "
          + "classes, empties out even though it carries the most mass.";
      } else {
        story = "Mild guidance already sharpens the samples: spread shrinks "
          + "and fewer samples sit where the classes overlap.";
      }
      note.textContent = shareText + story;
    }

    // --- drawing --------------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function classColors() {
      return [
        cssVar(mount, "--accent", "#ba3925"),
        cssVar(mount, "--link", "#2156a5"),
        isDark() ? "#6cc28f" : "#2e7d4f",
      ];
    }

    function draw() {
      var cw = stage.clientWidth || 260;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = cw + "px";
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(cw * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      var bg = cssVar(mount, "--card-bg", "#fff");
      var border = cssVar(mount, "--border", "#e2e2dd");
      var muted = cssVar(mount, "--muted", "#6a6a6a");
      var cols = classColors();
      chips.forEach(function (ch, c) { ch.sw.style.background = cols[c]; });

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, cw, cw);
      var X = function (v) { return cw / 2 + (cw / 2) * v / VIEW; };
      var Y = function (v) { return cw / 2 - (cw / 2) * v / VIEW; };
      var px = (cw / 2) / VIEW;

      // axes
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, Y(0)); ctx.lineTo(cw, Y(0));
      ctx.moveTo(X(0), 0); ctx.lineTo(X(0), cw);
      ctx.stroke();

      // data components: 1 and 2 sigma contours, selected class emphasised
      for (var k = 0; k < K; k++) {
        var cp = COMPS[k], sel = cp.c === cls;
        ctx.fillStyle = rgba(cols[cp.c], sel ? 0.13 : 0.06);
        ctx.beginPath();
        ctx.arc(X(cp.mu[0]), Y(cp.mu[1]), 2 * SD * px, 0, 6.284);
        ctx.fill();
        ctx.strokeStyle = rgba(cols[cp.c], sel ? 0.75 : 0.3);
        ctx.lineWidth = sel ? 1.3 : 1;
        ctx.setLineDash(sel ? [] : [3, 3]);
        ctx.beginPath();
        ctx.arc(X(cp.mu[0]), Y(cp.mu[1]), SD * px, 0, 6.284);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      if (!traj) return;
      var col = cols[cls];
      var lim = VIEW * 0.985;
      var clamp = function (v) { return Math.max(-lim, Math.min(lim, v)); };

      // trajectories
      ctx.strokeStyle = rgba(muted, 0.42);
      ctx.lineWidth = 0.9;
      ctx.lineJoin = "round";
      // stochastic (DDPM) paths are jagged, so draw fewer of them
      var trails = eta ? 12 : TRAILS;
      for (var i = 0; i < trails; i++) {
        ctx.beginPath();
        for (var s = 0; s <= shown; s++) {
          var o = (s * P + i) * 2;
          var tx = X(clamp(traj[o])), ty = Y(clamp(traj[o + 1]));
          if (s === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
        }
        ctx.stroke();
      }

      // particles at the current step; ones outside the view are pinned to the
      // edge as hollow rings so they still count visually
      var base = shown * P * 2;
      for (i = 0; i < P; i++) {
        var x = traj[base + 2 * i], y = traj[base + 2 * i + 1];
        var out = Math.abs(x) > lim || Math.abs(y) > lim;
        if (out) {
          ctx.strokeStyle = rgba(col, 0.7);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(X(clamp(x)), Y(clamp(y)), 2.2, 0, 6.284);
          ctx.stroke();
        } else {
          ctx.fillStyle = rgba(col, i < TRAILS ? 0.95 : 0.55);
          ctx.beginPath();
          ctx.arc(X(x), Y(y), i < TRAILS ? 2.1 : 1.6, 0, 6.284);
          ctx.fill();
        }
      }

      var label = "step " + shown + "/" + STEPS + "   w = " + w.toFixed(1);
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

    newLatent();
    run(true);
    window.addEventListener("resize", draw);
    new MutationObserver(draw).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="cfg"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

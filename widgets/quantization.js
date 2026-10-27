// Interactive weight quantization: pick a format, a mapping and a granularity,
// and see what happens to a real (seeded, synthetic) weight matrix.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-quantization
//   :::
//
// The tensor is 64 output channels x 128 inputs, drawn from a Laplace
// distribution (LLM weights are heavier-tailed than a Gaussian) with a mild
// per-channel scale. The "outlier channels" toggle multiplies four rows by 12,
// the kind of systematic outlier that LLM.int8() (Dettmers et al., 2022) found
// breaks a single per-tensor absmax scale.
//
// Every number is computed from the actual round trip w -> q -> w_hat:
//   symmetric (absmax)  s = max|w| / (2^(b-1) - 1),  q = clamp(round(w / s))
//   asymmetric          s = (max - min) / (2^b - 1), z = round(-min / s),
//                       q = clamp(round(w / s) + z, 0, 2^b - 1), w_hat = (q - z) s
//   NF4                 w_hat = c_k * max|w|, c_k the nearest of the 16 NormalFloat
//                       code values from QLoRA (Dettmers et al., 2023)
// grouped per tensor, per output channel, or in contiguous groups along the
// input dimension.
(function () {
  "use strict";

  var ROWS = 64, COLS = 128;
  var OUTLIER_ROWS = [5, 22, 41, 58];
  var OUTLIER_GAIN = 12;
  var LAPLACE_B = 0.015;
  var GROUP_SIZES = [8, 16, 32, 64, 128];

  // bitsandbytes' NF4 code book (QLoRA, appendix E).
  var NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141100311, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
  ];

  // --- seeded tensor -------------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rand = mulberry32(20240929);
  function gauss() {
    var u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }
  function laplace(b) {
    var u = rand() - 0.5;
    return -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u) + 1e-300);
  }

  var BASE = new Float64Array(ROWS * COLS);
  (function () {
    for (var i = 0; i < ROWS; i++) {
      var rs = Math.exp(0.25 * gauss());
      for (var j = 0; j < COLS; j++) BASE[i * COLS + j] = rs * laplace(LAPLACE_B);
    }
  })();
  var IS_OUTLIER = new Uint8Array(ROWS);
  OUTLIER_ROWS.forEach(function (r) { IS_OUTLIER[r] = 1; });

  function tensor(outliers) {
    var w = new Float64Array(BASE);
    if (outliers) {
      OUTLIER_ROWS.forEach(function (r) {
        for (var j = 0; j < COLS; j++) w[r * COLS + j] *= OUTLIER_GAIN;
      });
    }
    return w;
  }

  // Standard deviation of the non-outlier rows: a fixed yardstick for colour.
  var BULK_STD = (function () {
    var s = 0, n = 0;
    for (var i = 0; i < ROWS; i++) {
      if (IS_OUTLIER[i]) continue;
      for (var j = 0; j < COLS; j++) { s += BASE[i * COLS + j] * BASE[i * COLS + j]; n++; }
    }
    return Math.sqrt(s / n);
  })();

  // --- quantizers ----------------------------------------------------------
  function bitsOf(fmt) { return fmt === "int8" ? 8 : 4; }

  // Fit quantization parameters to one group's values.
  function fit(vals, cfg) {
    var lo = Infinity, hi = -Infinity, am = 0;
    for (var k = 0; k < vals.length; k++) {
      var v = vals[k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      if (Math.abs(v) > am) am = Math.abs(v);
    }
    if (cfg.fmt === "nf4") return { kind: "nf4", absmax: am || 1e-12 };
    var b = bitsOf(cfg.fmt);
    if (cfg.map === "sym") {
      var qmax = Math.pow(2, b - 1) - 1;
      return { kind: "sym", scale: (am || 1e-12) / qmax, qmax: qmax };
    }
    // Asymmetric: the range must contain 0 so that 0 is exactly representable.
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    var levels = Math.pow(2, b) - 1;
    var scale = (hi - lo) / levels || 1e-12;
    var zp = Math.round(-lo / scale);
    return { kind: "asym", scale: scale, zp: zp, qmax: levels };
  }

  function nearestNF4(x) {
    // NF4 is sorted; binary search for the nearest code.
    var lo = 0, hi = NF4.length - 1;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (NF4[mid] <= x) lo = mid; else hi = mid;
    }
    return (x - NF4[lo] <= NF4[hi] - x) ? lo : hi;
  }

  // Returns the integer code for v; dequant() maps it back.
  function encode(v, p) {
    if (p.kind === "nf4") return nearestNF4(v / p.absmax);
    if (p.kind === "sym") {
      var q = Math.round(v / p.scale);
      return Math.max(-p.qmax, Math.min(p.qmax, q));
    }
    var qa = Math.round(v / p.scale) + p.zp;
    return Math.max(0, Math.min(p.qmax, qa));
  }
  function dequant(q, p) {
    if (p.kind === "nf4") return NF4[q] * p.absmax;
    if (p.kind === "sym") return q * p.scale;
    return (q - p.zp) * p.scale;
  }
  function levelsOf(p) {
    var out = [], q;
    if (p.kind === "nf4") for (q = 0; q < 16; q++) out.push({ q: q, v: NF4[q] * p.absmax });
    else if (p.kind === "sym") for (q = -p.qmax; q <= p.qmax; q++) out.push({ q: q, v: q * p.scale });
    else for (q = 0; q <= p.qmax; q++) out.push({ q: q, v: (q - p.zp) * p.scale });
    return out;
  }

  // Group id of weight (i, j) under the chosen granularity.
  function groupLayout(cfg) {
    if (cfg.gran === "tensor") return { perRow: 1, size: COLS, n: 1, id: function () { return 0; } };
    if (cfg.gran === "channel") return { perRow: 1, size: COLS, n: ROWS, id: function (i) { return i; } };
    var g = cfg.group, per = COLS / g;
    return { perRow: per, size: g, n: ROWS * per, id: function (i, j) { return i * per + Math.floor(j / g); } };
  }

  function run(cfg) {
    var w = tensor(cfg.outliers);
    var lay = groupLayout(cfg);
    var members = [];
    for (var gi = 0; gi < lay.n; gi++) members.push([]);
    for (var i = 0; i < ROWS; i++) {
      for (var j = 0; j < COLS; j++) members[lay.id(i, j)].push(i * COLS + j);
    }
    var params = new Array(lay.n);
    var code = new Int16Array(ROWS * COLS);
    var hat = new Float64Array(ROWS * COLS);
    for (gi = 0; gi < lay.n; gi++) {
      var idx = members[gi];
      var vals = idx.map(function (k) { return w[k]; });
      var p = fit(vals, cfg);
      params[gi] = p;
      for (var m = 0; m < idx.length; m++) {
        var q = encode(w[idx[m]], p);
        code[idx[m]] = q;
        hat[idx[m]] = dequant(q, p);
      }
    }
    var se = 0, sig = 0, maxErr = 0, seBulk = 0, sigBulk = 0;
    for (var k = 0; k < w.length; k++) {
      var e = w[k] - hat[k];
      se += e * e; sig += w[k] * w[k];
      if (Math.abs(e) > maxErr) maxErr = Math.abs(e);
      if (!IS_OUTLIER[Math.floor(k / COLS)]) { seBulk += e * e; sigBulk += w[k] * w[k]; }
    }
    // Storage: b bits per weight plus an fp16 scale (and fp16 zero point for
    // asymmetric) per group.
    var b = bitsOf(cfg.fmt);
    var overhead = (cfg.map === "asym" && cfg.fmt !== "nf4" ? 32 : 16) * lay.n / w.length;
    var zeroBulk = 0, nBulk = 0;
    for (k = 0; k < w.length; k++) {
      if (IS_OUTLIER[Math.floor(k / COLS)]) continue;
      nBulk++;
      if (hat[k] === 0) zeroBulk++;
    }
    return {
      w: w, hat: hat, code: code, lay: lay, params: params, members: members,
      mse: se / w.length,
      sqnr: 10 * Math.log10(sig / (se || 1e-300)),
      sqnrBulk: 10 * Math.log10(sigBulk / (seBulk || 1e-300)),
      maxErr: maxErr,
      bpw: b + overhead,
      zeroFrac: zeroBulk / nBulk,
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
      ".quant-opts{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;margin-bottom:.8rem}" +
      ".quant-group{display:flex;flex-direction:column;gap:.35rem}" +
      ".quant-group .widget-label{margin-bottom:0}" +
      ".quant-seg{display:flex;flex-wrap:wrap;gap:.3rem}" +
      ".quant-seg .widget-btn{padding:.3rem .65rem}" +
      ".quant-slider{max-width:28rem}" +
      ".quant-slider[data-off]{opacity:.4}" +
      ".quant-sub{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;margin:1rem 0 .35rem}" +
      ".quant-sub .widget-label{margin-bottom:0}" +
      ".quant-info{font-size:.68rem;color:var(--muted);text-align:right}" +
      ".quant-heat{cursor:pointer}";
    document.head.appendChild(s);
  }

  function build(mount) {
    injectStyle();
    var cfg = { fmt: "int4", map: "sym", gran: "tensor", group: 64, outliers: true };
    var sel = { row: 0, col: 0 };      // which weight's group to inspect
    var R = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Quantizing a weight matrix";
    el("div", "widget-meta", head).textContent =
      ROWS + " × " + COLS + " weights · Laplace, seeded";
    mount.parentNode.insertBefore(head, mount);

    // --- controls ----------------------------------------------------------
    var optsRow = el("div", "quant-opts", mount);
    var syncs = [];
    function segmented(label, choices, get, set, disabled) {
      var g = el("div", "quant-group", optsRow);
      el("div", "widget-label", g).textContent = label;
      var seg = el("div", "quant-seg", g);
      var btns = choices.map(function (c) {
        var b = el("button", "widget-btn", seg);
        b.type = "button";
        b.textContent = c[1];
        b.addEventListener("click", function () { set(c[0]); refresh(); });
        return { b: b, v: c[0] };
      });
      syncs.push(function () {
        btns.forEach(function (x) {
          var on = x.v === get();
          if (on) x.b.setAttribute("data-primary", ""); else x.b.removeAttribute("data-primary");
          x.b.setAttribute("aria-pressed", on ? "true" : "false");
          x.b.disabled = !!(disabled && disabled(x.v));
        });
      });
    }
    segmented("Format", [["int8", "int8"], ["int4", "int4"], ["nf4", "NF4"]],
      function () { return cfg.fmt; }, function (v) { cfg.fmt = v; });
    segmented("Mapping", [["sym", "absmax"], ["asym", "zero-point"]],
      function () { return cfg.fmt === "nf4" ? "sym" : cfg.map; },
      function (v) { cfg.map = v; },
      function (v) { return cfg.fmt === "nf4" && v === "asym"; });
    segmented("Granularity", [["tensor", "per-tensor"], ["channel", "per-channel"], ["group", "group-wise"]],
      function () { return cfg.gran; }, function (v) { cfg.gran = v; });
    segmented("Outlier channels", [[false, "off"], [true, "on"]],
      function () { return cfg.outliers; }, function (v) { cfg.outliers = v; });

    var sl = el("div", "widget-slider-row quant-slider", mount);
    el("span", "quant-slabel", sl).textContent = "group size";
    var gIn = el("input", "widget-slider", sl);
    gIn.type = "range"; gIn.min = "0"; gIn.max = String(GROUP_SIZES.length - 1); gIn.step = "1";
    gIn.value = String(GROUP_SIZES.indexOf(cfg.group));
    gIn.setAttribute("aria-label", "Group size along the input dimension");
    var gVal = el("span", "widget-slider-val", sl);
    gIn.addEventListener("input", function () {
      cfg.group = GROUP_SIZES[Number(gIn.value)];
      if (cfg.gran !== "group") cfg.gran = "group";
      refresh();
    });

    var sub1 = el("div", "quant-sub", mount);
    el("div", "widget-label", sub1).textContent = "Weight histogram and quantization levels";
    var histInfo = el("span", "quant-info", sub1);
    var hist = el("canvas", "widget-plot", mount);
    hist.style.cursor = "default";
    hist.setAttribute("aria-label",
      "Histogram of the original weights with the quantization levels of the selected group");

    var sub2 = el("div", "quant-sub", mount);
    el("div", "widget-label", sub2).textContent = "|error| per weight";
    var heatInfo = el("span", "quant-info", sub2);
    var heat = el("canvas", "widget-plot quant-heat", mount);
    heat.setAttribute("aria-label",
      "Absolute quantization error for every weight; rows are output channels. " +
      "Click to choose which group's levels are drawn above.");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);
    el("div", "widget-label", left).textContent = "Whole tensor";
    el("div", "widget-label", right).textContent = "Selected group";
    function stat(parent, name) {
      var row = el("div", "widget-row widget-row--wide", parent);
      el("span", "widget-row-name", row).textContent = name;
      var v = el("span", "widget-row-val", row);
      v.style.width = "auto";
      v.style.flex = "1 1 auto";
      return v;
    }
    var stMse = stat(left, "MSE");
    var stSq = stat(left, "SQNR");
    var stSqB = stat(left, "SQNR (bulk)");
    var stMax = stat(left, "max |err|");
    var stBpw = stat(left, "bits / weight");
    var stWhich = stat(right, "group");
    var stScale = stat(right, "scale");
    var stRange = stat(right, "range");
    var stUsed = stat(right, "levels used");
    var stGSq = stat(right, "SQNR");
    var note = el("div", "widget-note", mount);

    // --- drawing -----------------------------------------------------------
    var hctx = hist.getContext("2d");
    var ectx = heat.getContext("2d");

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

    function sizeCanvas(c, ctx, h) {
      var W = Math.max(200, mount.clientWidth - 32);
      var dpr = window.devicePixelRatio || 1;
      c.style.height = h + "px";
      c.width = Math.round(W * dpr);
      c.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, h);
      return W;
    }

    function selGroup() { return R.lay.id(sel.row, sel.col); }

    function drawHist() {
      var H = 190;
      var W = sizeCanvas(hist, hctx, H);
      var p = palette();
      var gid = selGroup();
      var gp = R.params[gid];
      var idx = R.members[gid];

      // x-range: the selected group's own range, padded, so the levels that
      // matter are readable. A per-tensor fit zooms to the non-outlier bulk
      // instead, which is exactly where its coarse levels hurt. Anything
      // beyond the window is counted at the edges.
      var am = 0, bulkMax = 0, k;
      for (k = 0; k < idx.length; k++) am = Math.max(am, Math.abs(R.w[idx[k]]));
      for (k = 0; k < R.w.length; k++) {
        if (!IS_OUTLIER[Math.floor(k / COLS)]) bulkMax = Math.max(bulkMax, Math.abs(R.w[k]));
      }
      var X = 1.08 * (cfg.gran === "tensor" ? bulkMax : am);
      var padL = 8, padR = 8, top = 10, base = H - 22;
      function px(v) { return padL + (W - padL - padR) * (v + X) / (2 * X); }

      // Histogram of every weight in the tensor (light) and of the selected
      // group (accent), on a square-root count axis.
      var NB = Math.max(40, Math.min(120, Math.floor(W / 6)));
      var all = new Float64Array(NB), grp = new Float64Array(NB);
      var outL = 0, outR = 0;
      for (k = 0; k < R.w.length; k++) {
        var v = R.w[k];
        if (v < -X) { outL++; continue; }
        if (v > X) { outR++; continue; }
        all[Math.min(NB - 1, Math.floor((v + X) / (2 * X) * NB))]++;
      }
      for (k = 0; k < idx.length; k++) {
        v = R.w[idx[k]];
        if (v >= -X && v <= X) grp[Math.min(NB - 1, Math.floor((v + X) / (2 * X) * NB))]++;
      }
      var mx = 0;
      for (k = 0; k < NB; k++) mx = Math.max(mx, all[k]);
      var bw = (W - padL - padR) / NB;
      function hy(c) { return (base - top) * Math.sqrt(c / mx); }
      hctx.fillStyle = p.block;
      for (k = 0; k < NB; k++) {
        var h = hy(all[k]);
        hctx.fillRect(padL + k * bw + 0.5, base - h, Math.max(1, bw - 1), h);
      }
      hctx.fillStyle = p.link;
      hctx.globalAlpha = 0.55;
      for (k = 0; k < NB; k++) {
        if (!grp[k]) continue;
        h = hy(grp[k]);
        hctx.fillRect(padL + k * bw + 0.5, base - h, Math.max(1, bw - 1), h);
      }
      hctx.globalAlpha = 1;

      // Quantization levels of the selected group, height = how many of its
      // weights landed on that level.
      var levels = levelsOf(gp);
      var use = {};
      for (k = 0; k < idx.length; k++) use[R.code[idx[k]]] = (use[R.code[idx[k]]] || 0) + 1;
      var umax = 0;
      for (var key in use) umax = Math.max(umax, use[key]);
      var inView = 0, used = 0;
      hctx.strokeStyle = p.accent;
      levels.forEach(function (L) {
        if (use[L.q]) used++;
        if (L.v < -X || L.v > X) return;
        inView++;
        var x = Math.round(px(L.v)) + 0.5;
        hctx.globalAlpha = 0.35;
        hctx.lineWidth = 1;
        hctx.beginPath();
        hctx.moveTo(x, top); hctx.lineTo(x, base);
        hctx.stroke();
        hctx.globalAlpha = 1;
        if (use[L.q]) {
          var lh = 6 + (base - top - 6) * 0.45 * Math.sqrt(use[L.q] / umax);
          hctx.lineWidth = levels.length > 64 ? 1.5 : 2.5;
          hctx.beginPath();
          hctx.moveTo(x, base); hctx.lineTo(x, base - lh);
          hctx.stroke();
        }
      });
      // axis
      hctx.strokeStyle = p.border;
      hctx.lineWidth = 1;
      hctx.beginPath();
      hctx.moveTo(padL, base + 0.5); hctx.lineTo(W - padR, base + 0.5);
      hctx.stroke();
      hctx.fillStyle = p.muted;
      hctx.font = '10px "JetBrains Mono", monospace';
      hctx.textBaseline = "top";
      hctx.textAlign = "left";
      hctx.fillText((-X).toFixed(3), padL, base + 5);
      hctx.textAlign = "center";
      hctx.fillText("0", px(0), base + 5);
      hctx.textAlign = "right";
      hctx.fillText(X.toFixed(3), W - padR, base + 5);
      // out-of-view weights
      hctx.font = '10px "Open Sans", sans-serif';
      hctx.textBaseline = "top";
      if (outL) { hctx.textAlign = "left"; hctx.fillText("◀ " + outL + " outside", padL, top - 4); }
      if (outR) { hctx.textAlign = "right"; hctx.fillText(outR + " outside ▶", W - padR, top - 4); }

      histInfo.textContent = levels.length + " levels, " + inView + " in view, "
        + used + " used by this group";
      return { used: used, levels: levels.length };
    }

    var heatGeom = null;
    function drawHeat() {
      var Wfull = Math.max(200, mount.clientWidth - 32);
      var cw = Math.max(2, Math.min(5, Math.floor(Wfull / COLS)));
      var ch = cw;
      var H = ch * ROWS;
      var W = sizeCanvas(heat, ectx, H);
      var gw = cw * COLS;
      var x0 = Math.floor((W - gw) / 2);
      heatGeom = { x0: x0, cw: cw, ch: ch };
      var p = palette();
      ectx.fillStyle = p.block;
      ectx.fillRect(x0, 0, gw, H);
      ectx.fillStyle = p.accent;
      var ref = 0.5 * BULK_STD;   // fixed yardstick: |err| of half a std is full colour
      for (var i = 0; i < ROWS; i++) {
        for (var j = 0; j < COLS; j++) {
          var k = i * COLS + j;
          var a = Math.min(1, Math.abs(R.w[k] - R.hat[k]) / ref);
          if (a < 0.02) continue;
          ectx.globalAlpha = a;
          ectx.fillRect(x0 + j * cw, i * ch, cw, ch);
        }
      }
      ectx.globalAlpha = 1;
      // group boundaries along the input dim
      if (cfg.gran === "group" && cw * cfg.group >= 8) {
        ectx.strokeStyle = p.bg;
        ectx.lineWidth = 1;
        ectx.beginPath();
        for (var g = cfg.group; g < COLS; g += cfg.group) {
          ectx.moveTo(x0 + g * cw + 0.5, 0); ectx.lineTo(x0 + g * cw + 0.5, H);
        }
        ectx.stroke();
      }
      // outlier-channel markers
      if (cfg.outliers) {
        ectx.fillStyle = p.link;
        OUTLIER_ROWS.forEach(function (r) {
          ectx.fillRect(Math.max(0, x0 - 6), r * ch, 3, ch);
        });
      }
      // selected group outline
      var gx0, gx1, gy0, gy1;
      if (cfg.gran === "tensor") { gx0 = 0; gx1 = COLS; gy0 = 0; gy1 = ROWS; }
      else if (cfg.gran === "channel") { gx0 = 0; gx1 = COLS; gy0 = sel.row; gy1 = sel.row + 1; }
      else {
        gx0 = Math.floor(sel.col / cfg.group) * cfg.group; gx1 = gx0 + cfg.group;
        gy0 = sel.row; gy1 = sel.row + 1;
      }
      ectx.strokeStyle = p.fg;
      ectx.lineWidth = 1.5;
      ectx.strokeRect(x0 + gx0 * cw - 1, gy0 * ch - 1, (gx1 - gx0) * cw + 2, (gy1 - gy0) * ch + 2);
      heatInfo.textContent = "full colour = " + ref.toFixed(4) + " (½σ of the bulk)"
        + (cfg.outliers ? " · blue ticks mark outlier rows" : "");
    }

    function fmtSci(x) {
      if (x === 0) return "0";
      return x < 1e-3 ? x.toExponential(2) : x.toFixed(4);
    }

    function readout(h) {
      stMse.textContent = R.mse.toExponential(2);
      stSq.textContent = R.sqnr.toFixed(1) + " dB";
      stSqB.textContent = R.sqnrBulk.toFixed(1) + " dB";
      stMax.textContent = fmtSci(R.maxErr);
      stBpw.textContent = R.bpw.toFixed(2);
      var gid = selGroup(), gp = R.params[gid], idx = R.members[gid];
      var where = cfg.gran === "tensor" ? "whole tensor"
        : cfg.gran === "channel" ? "row " + sel.row
        : "row " + sel.row + ", cols " + Math.floor(sel.col / cfg.group) * cfg.group + "–"
          + (Math.floor(sel.col / cfg.group) * cfg.group + cfg.group - 1);
      stWhich.textContent = where + (IS_OUTLIER[sel.row] && cfg.outliers && cfg.gran !== "tensor" ? " (outlier)" : "");
      stScale.textContent = gp.kind === "nf4" ? "absmax " + fmtSci(gp.absmax)
        : fmtSci(gp.scale) + (gp.kind === "asym" ? ", z = " + gp.zp : "");
      var lo = Infinity, hi = -Infinity, s = 0, e = 0;
      for (var k = 0; k < idx.length; k++) {
        var v = R.w[idx[k]];
        lo = Math.min(lo, v); hi = Math.max(hi, v);
        s += v * v; e += (v - R.hat[idx[k]]) * (v - R.hat[idx[k]]);
      }
      stRange.textContent = "[" + lo.toFixed(3) + ", " + hi.toFixed(3) + "]";
      stUsed.textContent = h.used + " / " + h.levels;
      stGSq.textContent = (10 * Math.log10(s / (e || 1e-300))).toFixed(1) + " dB";
    }

    function explain(h) {
      var msg;
      if (cfg.gran === "tensor" && cfg.outliers) {
        msg = "One absmax scale for the whole tensor is set by the outlier rows, "
          + "so the ordinary weights only reach " + h.used + " of the "
          + h.levels + " levels and " + Math.round(100 * R.zeroFrac)
          + "% of them round to exactly zero. This is the "
          + "failure LLM.int8() was built around. Try per-channel or group-wise.";
      } else if (cfg.fmt === "nf4") {
        msg = "NF4 levels are quantiles of a normal distribution scaled by each "
          + "group's absmax, so they bunch up near zero where most weights live.";
      } else if (cfg.map === "asym") {
        msg = "Zero-point mapping spends the full code range on [min, max] "
          + "instead of [−absmax, absmax]. It helps most when a group is lopsided, "
          + "and costs an extra stored offset per group.";
      } else if (cfg.gran === "group") {
        msg = "Smaller groups track the local range more tightly and raise SQNR, "
          + "but every group stores its own fp16 scale, which shows up in bits/weight.";
      } else {
        msg = "Each " + (cfg.gran === "channel" ? "output channel" : "tensor")
          + " gets its own absmax scale. Click the error map to inspect another "
          + "group's levels.";
      }
      note.textContent = msg;
    }

    function draw() {
      var h = drawHist();
      drawHeat();
      readout(h);
      return h;
    }

    function refresh() {
      if (cfg.fmt === "nf4") cfg.map = "sym";
      syncs.forEach(function (f) { f(); });
      gVal.textContent = String(cfg.group);
      if (cfg.gran === "group") sl.removeAttribute("data-off"); else sl.setAttribute("data-off", "");
      R = run(cfg);
      explain(draw());
    }

    heat.addEventListener("click", function (ev) {
      if (!heatGeom) return;
      var r = heat.getBoundingClientRect();
      var j = Math.floor((ev.clientX - r.left - heatGeom.x0) / heatGeom.cw);
      var i = Math.floor((ev.clientY - r.top) / heatGeom.ch);
      if (i < 0 || i >= ROWS || j < 0 || j >= COLS) return;
      sel = { row: i, col: j };
      draw();
    });

    refresh();
    window.addEventListener("resize", function () { draw(); });
    new MutationObserver(function () { draw(); }).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="quantization"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

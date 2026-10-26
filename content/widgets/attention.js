// Interactive scaled dot-product attention on an eight-token sentence.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-attention
//   :::
//
// Nothing here is canned. Each token has a fixed (seeded) random embedding in
// R^16, positional information is injected one of three ways, and every head
// computes softmax(Q K^T / sqrt(d_k)) from its own slice of seeded W_Q / W_K.
//
//   none        x_t = e_t. Attention is permutation-equivariant, so the two
//               "the" tokens get identical rows and identical columns.
//   sinusoidal  x_t = e_t + PE(t), PE(t, 2i) = sin(t / 10000^(2i/d)),
//               PE(t, 2i+1) = cos(t / 10000^(2i/d))   (Vaswani et al.)
//   RoPE        x_t = e_t, then q and k are rotated pairwise by t * theta_i
//               with theta_i = 10000^(-2i/d_k), so q_m . k_n depends on the
//               offset m - n only (Su et al., RoFormer).
//
// Heads split the model dimension exactly as in multi-head attention: with h
// heads each one uses d_k = 16 / h columns of the same projection matrices.
(function () {
  "use strict";

  var TOKENS = ["the", "cat", "saw", "the", "dog", "and", "it", "ran"];
  var T = TOKENS.length;
  var D = 16;                 // model dimension
  var GAIN = 1.25;            // projection gain, sets the typical logit spread
  var HEAD_CHOICES = [1, 2, 4];

  // --- seeded randomness ---------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rand = mulberry32(20240108);
  function gauss() {
    var u = 0, v = 0;
    while (u === 0) u = rand();
    v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function matrix(rows, cols, std) {
    var m = [];
    for (var i = 0; i < rows; i++) {
      var r = new Float64Array(cols);
      for (var j = 0; j < cols; j++) r[j] = gauss() * std;
      m.push(r);
    }
    return m;
  }

  // One embedding per *word type*, so repeated words share a vector.
  var VOCAB = {};
  TOKENS.forEach(function (w) {
    if (!VOCAB[w]) VOCAB[w] = matrix(1, D, 1)[0];
  });
  var WQ = matrix(D, D, GAIN / Math.sqrt(D));
  var WK = matrix(D, D, GAIN / Math.sqrt(D));

  function sinusoid(t, d) {
    var pe = new Float64Array(d);
    for (var i = 0; i < d / 2; i++) {
      var w = Math.pow(10000, -2 * i / d);
      pe[2 * i] = Math.sin(t * w);
      pe[2 * i + 1] = Math.cos(t * w);
    }
    return pe;
  }

  function project(x, W) {
    var out = new Float64Array(D);
    for (var j = 0; j < D; j++) {
      var s = 0;
      for (var i = 0; i < D; i++) s += x[i] * W[i][j];
      out[j] = s;
    }
    return out;
  }

  // Rotate consecutive pairs of one head's slice by t * theta_i.
  function rope(v, t, off, dk) {
    for (var i = 0; i < dk / 2; i++) {
      var th = t * Math.pow(10000, -2 * i / dk);
      var c = Math.cos(th), s = Math.sin(th);
      var a = v[off + 2 * i], b = v[off + 2 * i + 1];
      v[off + 2 * i] = a * c - b * s;
      v[off + 2 * i + 1] = a * s + b * c;
    }
  }

  // Returns per head: { logits: TxT, weights: TxT } (masked logits = -Inf).
  function attention(opts) {
    var H = opts.heads, dk = D / H;
    var Q = [], K = [], t, h, i, j;
    for (t = 0; t < T; t++) {
      var x = new Float64Array(VOCAB[TOKENS[t]]);
      if (opts.pe === "sin") {
        var pe = sinusoid(t, D);
        for (i = 0; i < D; i++) x[i] += pe[i];
      }
      var q = project(x, WQ), k = project(x, WK);
      if (opts.pe === "rope") {
        for (h = 0; h < H; h++) { rope(q, t, h * dk, dk); rope(k, t, h * dk, dk); }
      }
      Q.push(q); K.push(k);
    }
    var scale = (opts.scaled ? Math.sqrt(dk) : 1) * opts.temp;
    var out = [];
    for (h = 0; h < H; h++) {
      var off = h * dk, logits = [], weights = [];
      for (i = 0; i < T; i++) {
        var lr = new Float64Array(T), wr = new Float64Array(T), mx = -Infinity;
        for (j = 0; j < T; j++) {
          if (opts.causal && j > i) { lr[j] = -Infinity; continue; }
          var s = 0;
          for (var c = 0; c < dk; c++) s += Q[i][off + c] * K[j][off + c];
          lr[j] = s / scale;
          if (lr[j] > mx) mx = lr[j];
        }
        var z = 0;
        for (j = 0; j < T; j++) { wr[j] = Math.exp(lr[j] - mx); z += wr[j]; }
        for (j = 0; j < T; j++) wr[j] /= z;
        logits.push(lr); weights.push(wr);
      }
      out.push({ logits: logits, weights: weights, dk: dk });
    }
    return out;
  }

  function entropyBits(p) {
    var e = 0;
    for (var i = 0; i < p.length; i++) if (p[i] > 0) e -= p[i] * Math.log2(p[i]);
    return e;
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
      ".attn-opts{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;margin-bottom:.9rem}" +
      ".attn-group{display:flex;flex-direction:column;gap:.35rem}" +
      ".attn-group .widget-label{margin-bottom:0}" +
      ".attn-seg{display:flex;flex-wrap:wrap;gap:.3rem}" +
      ".attn-seg .widget-btn{padding:.3rem .65rem}" +
      ".attn-plot{cursor:pointer}" +
      ".attn-sliders{max-width:28rem;margin-top:.2rem}" +
      ".attn-sliders .widget-slider-row>span:first-child{width:6.4em}";
    document.head.appendChild(s);
  }

  function build(mount) {
    injectStyle();
    var opts = { heads: 1, pe: "sin", causal: false, scaled: true, temp: 1 };
    var sel = { row: 6, head: 0 };     // pinned query row ("it")
    var hover = null;                  // transient {row, head} under pointer
    var result = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Scaled dot-product attention";
    var meta = el("div", "widget-meta", head);
    mount.parentNode.insertBefore(head, mount);

    // --- option toggles ----------------------------------------------------
    var optsRow = el("div", "attn-opts", mount);
    function segmented(label, choices, get, set) {
      var g = el("div", "attn-group", optsRow);
      el("div", "widget-label", g).textContent = label;
      var seg = el("div", "attn-seg", g);
      var btns = choices.map(function (c) {
        var b = el("button", "widget-btn", seg);
        b.type = "button";
        b.textContent = c[1];
        b.addEventListener("click", function () { set(c[0]); sync(); refresh(); });
        return { b: b, v: c[0] };
      });
      function sync() {
        btns.forEach(function (x) {
          var on = x.v === get();
          if (on) x.b.setAttribute("data-primary", ""); else x.b.removeAttribute("data-primary");
          x.b.setAttribute("aria-pressed", on ? "true" : "false");
        });
      }
      sync();
    }
    segmented("Positions", [["none", "none"], ["sin", "sinusoidal"], ["rope", "RoPE"]],
      function () { return opts.pe; }, function (v) { opts.pe = v; });
    segmented("Heads", HEAD_CHOICES.map(function (h) { return [h, String(h)]; }),
      function () { return opts.heads; },
      function (v) { opts.heads = v; sel.head = Math.min(sel.head, v - 1); });
    segmented("Mask", [[false, "bidirectional"], [true, "causal"]],
      function () { return opts.causal; }, function (v) { opts.causal = v; });
    segmented("Scale", [[true, "÷ √dₖ"], [false, "none"]],
      function () { return opts.scaled; }, function (v) { opts.scaled = v; });

    var sliders = el("div", "attn-sliders", mount);
    var srow = el("div", "widget-slider-row", sliders);
    el("span", "attn-slabel", srow).textContent = "softmax τ";
    var tIn = el("input", "widget-slider", srow);
    tIn.type = "range"; tIn.min = "0"; tIn.max = "1000"; tIn.step = "1";
    var TMIN = 0.1, TMAX = 4;
    function tPos(v) { return Math.round(1000 * Math.log(v / TMIN) / Math.log(TMAX / TMIN)); }
    function tVal(p) { return TMIN * Math.pow(TMAX / TMIN, p / 1000); }
    tIn.value = String(tPos(1));
    tIn.setAttribute("aria-label", "Softmax temperature");
    var tShow = el("span", "widget-slider-val", srow);
    tIn.addEventListener("input", function () {
      opts.temp = tVal(Number(tIn.value));
      refresh();
    });

    var canvas = el("canvas", "widget-plot attn-plot", mount);
    canvas.setAttribute("aria-label",
      "Attention heatmaps, one per head. Rows are queries, columns are keys. " +
      "Hover or tap a row to inspect its weights.");

    var cols = el("div", "widget-cols", mount);
    var left = el("div", null, cols);
    var right = el("div", null, cols);
    var rowLabel = el("div", "widget-label", left);
    var bars = TOKENS.map(function (tok, j) {
      var row = el("div", "widget-row widget-row--wide", left);
      el("span", "widget-row-name", row).textContent = j + " " + tok;
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", row);
      return { fill: fill, val: val, row: row };
    });
    el("div", "widget-label", right).textContent = "This row";
    function stat(name) {
      var row = el("div", "widget-row widget-row--wide", right);
      el("span", "widget-row-name", row).textContent = name;
      var v = el("span", "widget-row-val", row);
      v.style.width = "auto";
      v.style.flex = "1 1 auto";
      return v;
    }
    var stEnt = stat("entropy");
    var stMax = stat("top key");
    var stSpread = stat("logit range");
    var stDk = stat("d_k");
    var note = el("div", "widget-note", mount);

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");
    var layout = null;

    function palette() {
      return {
        accent: cssVar(mount, "--accent", "#ba3925"),
        fg: cssVar(mount, "--fg", "#1a1a1a"),
        muted: cssVar(mount, "--muted", "#6a6a6a"),
        border: cssVar(mount, "--border", "#e2e2dd"),
        block: cssVar(mount, "--block-bg", "#f3f3f2"),
        bg: cssVar(mount, "--card-bg", "#fff"),
      };
    }

    function computeLayout(W) {
      var H = opts.heads;
      var perRow = Math.min(H, 2);
      var gapX = 14, labW = 30;
      var avail = (W - gapX * (perRow - 1)) / perRow - labW;
      var cell = Math.max(10, Math.min(H === 1 ? 44 : 40, Math.floor(avail / T)));
      // Narrow cells cannot fit a horizontal key label; stand them on end.
      var rotate = cell < 27;
      var topH = rotate ? 38 : 30;
      var mapW = labW + cell * T;
      var totalW = mapW * perRow + gapX * (perRow - 1);
      var x0 = Math.max(0, (W - totalW) / 2);
      var rowsN = Math.ceil(H / perRow);
      var mapH = topH + cell * T;
      var maps = [];
      for (var h = 0; h < H; h++) {
        var r = Math.floor(h / perRow), c = h % perRow;
        maps.push({
          x: x0 + c * (mapW + gapX) + labW,
          y: r * (mapH + 12) + topH,
          labW: labW, topH: topH,
        });
      }
      return {
        cell: cell, maps: maps, rotate: rotate,
        height: rowsN * mapH + (rowsN - 1) * 12 + 4,
      };
    }

    function draw() {
      var W = mount.clientWidth - 2 || 600;
      if (W > 2000) W = 2000;
      layout = computeLayout(W);
      var Hpx = layout.height;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = Hpx + "px";
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(Hpx * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, Hpx);

      var p = palette();
      var cell = layout.cell;
      var active = hover || sel;
      var fs = cell >= 26 ? 11 : 10;
      ctx.font = fs + 'px "Open Sans", sans-serif';

      layout.maps.forEach(function (m, h) {
        var A = result[h].weights;
        // head title
        ctx.fillStyle = h === active.head ? p.accent : p.muted;
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        if (opts.heads > 1) {
          ctx.font = "600 " + fs + 'px "Open Sans", sans-serif';
          ctx.fillText("head " + (h + 1), m.x - m.labW, m.y - m.topH + 10);
          ctx.font = fs + 'px "Open Sans", sans-serif';
        }
        // column labels (keys)
        ctx.fillStyle = p.muted;
        for (var j = 0; j < T; j++) {
          if (layout.rotate) {
            ctx.save();
            ctx.translate(m.x + (j + 0.5) * cell, m.y - 3);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(TOKENS[j], 0, 0);
            ctx.restore();
          } else {
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";
            ctx.fillText(TOKENS[j], m.x + (j + 0.5) * cell, m.y - 3);
          }
        }
        // row labels (queries)
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (var i = 0; i < T; i++) {
          ctx.fillStyle = (h === active.head && i === active.row) ? p.accent : p.muted;
          ctx.fillText(TOKENS[i], m.x - 5, m.y + (i + 0.5) * cell);
        }
        // cells
        ctx.fillStyle = p.block;
        ctx.fillRect(m.x, m.y, cell * T, cell * T);
        for (i = 0; i < T; i++) {
          for (j = 0; j < T; j++) {
            var cx = m.x + j * cell, cy = m.y + i * cell;
            if (opts.causal && j > i) {
              ctx.strokeStyle = p.border;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(cx + 2, cy + cell - 2);
              ctx.lineTo(cx + cell - 2, cy + 2);
              ctx.stroke();
              continue;
            }
            ctx.globalAlpha = Math.min(1, A[i][j]);
            ctx.fillStyle = p.accent;
            ctx.fillRect(cx, cy, cell, cell);
            ctx.globalAlpha = 1;
            if (cell >= 34) {
              ctx.fillStyle = A[i][j] > 0.55 ? "#fff" : p.fg;
              ctx.textAlign = "center";
              ctx.font = '9px "JetBrains Mono", monospace';
              ctx.fillText(A[i][j].toFixed(2).replace(/^0/, ""), cx + cell / 2, cy + cell / 2);
              ctx.font = fs + 'px "Open Sans", sans-serif';
            }
          }
        }
        // grid lines
        ctx.strokeStyle = p.bg;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var g = 1; g < T; g++) {
          ctx.moveTo(m.x + g * cell + 0.5, m.y);
          ctx.lineTo(m.x + g * cell + 0.5, m.y + T * cell);
          ctx.moveTo(m.x, m.y + g * cell + 0.5);
          ctx.lineTo(m.x + T * cell, m.y + g * cell + 0.5);
        }
        ctx.stroke();
        // highlighted row
        if (h === active.head) {
          ctx.strokeStyle = p.fg;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(m.x + 0.75, m.y + active.row * cell + 0.75,
            T * cell - 1.5, cell - 1.5);
        }
      });
    }

    function readout() {
      var a = hover || sel;
      var res = result[a.head];
      var w = res.weights[a.row], lg = res.logits[a.row];
      rowLabel.textContent = "Query “" + TOKENS[a.row] + "” (position " + a.row + ")"
        + (opts.heads > 1 ? " · head " + (a.head + 1) : "") + " attends to";
      var best = 0, lo = Infinity, hi = -Infinity;
      for (var j = 0; j < T; j++) {
        var masked = !isFinite(lg[j]);
        bars[j].row.style.opacity = masked ? "0.35" : "1";
        bars[j].fill.style.width = (100 * w[j]).toFixed(1) + "%";
        bars[j].val.textContent = masked ? "masked" : w[j].toFixed(3);
        if (w[j] > w[best]) best = j;
        if (!masked) { lo = Math.min(lo, lg[j]); hi = Math.max(hi, lg[j]); }
      }
      var n = 0;
      for (j = 0; j < T; j++) if (isFinite(lg[j])) n++;
      stEnt.textContent = entropyBits(w).toFixed(2) + " / " + Math.log2(n).toFixed(2) + " bits";
      stMax.textContent = "“" + TOKENS[best] + "” @" + best + " · " + w[best].toFixed(2);
      stSpread.textContent = (hi - lo).toFixed(2);
      stDk.textContent = res.dk + (opts.scaled ? " (÷" + Math.sqrt(res.dk).toFixed(2) + ")" : " (unscaled)");
    }

    function explain() {
      var msg;
      if (opts.pe === "none") {
        msg = "No positional information: the two “the” tokens have identical "
          + "embeddings, so their rows and columns are identical in every head. "
          + "Without a mask, attention cannot tell word order apart.";
        if (opts.causal) {
          msg = "No positional encoding, but the causal mask still breaks the "
            + "symmetry: each query only sees its prefix, so order leaks in "
            + "through the mask alone.";
        }
      } else if (opts.pe === "sin") {
        msg = "Sinusoidal encodings are added to the embeddings before the "
          + "projections, so the two “the” tokens now differ and get different "
          + "rows and columns.";
      } else {
        msg = "RoPE rotates each (q, k) pair by an angle proportional to the "
          + "position, so qₘ·kₙ depends only on the offset m − n. Nothing is "
          + "added to the embeddings themselves.";
      }
      if (!opts.scaled) {
        msg += " Without ÷√dₖ the logits grow with dₖ and the softmax saturates.";
      }
      note.textContent = msg;
    }

    function refresh() {
      result = attention(opts);
      meta.textContent = T + " tokens · d = " + D + " · " + opts.heads
        + (opts.heads === 1 ? " head" : " heads") + " × dₖ = " + D / opts.heads;
      tShow.textContent = opts.temp.toFixed(2);
      draw();
      readout();
      explain();
    }

    // --- interaction -------------------------------------------------------
    function hit(ev) {
      if (!layout) return null;
      var r = canvas.getBoundingClientRect();
      var x = ev.clientX - r.left, y = ev.clientY - r.top;
      for (var h = 0; h < layout.maps.length; h++) {
        var m = layout.maps[h];
        if (x >= m.x - m.labW && x <= m.x + layout.cell * T &&
            y >= m.y && y < m.y + layout.cell * T) {
          return { head: h, row: Math.floor((y - m.y) / layout.cell) };
        }
      }
      return null;
    }
    canvas.addEventListener("pointermove", function (ev) {
      if (ev.pointerType !== "mouse") return;
      var h = hit(ev);
      if (!h && !hover) return;
      if (h && hover && h.head === hover.head && h.row === hover.row) return;
      hover = h;
      draw(); readout();
    });
    canvas.addEventListener("pointerleave", function () {
      if (!hover) return;
      hover = null;
      draw(); readout();
    });
    canvas.addEventListener("click", function (ev) {
      var h = hit(ev);
      if (!h) return;
      sel = h; hover = null;
      draw(); readout();
    });

    refresh();
    window.addEventListener("resize", function () { draw(); });
    new MutationObserver(function () { draw(); }).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="attention"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

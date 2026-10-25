// A live Mandelbrot / Julia explorer: the escape-time loop from the post,
// one fragment-shader invocation per pixel.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-mandelbrot
//   :::
//
// Every pixel iterates z <- z^2 + c until |z| > R (R = 256) or the iteration
// budget runs out, exactly like the C code in the post. Colouring is the
// continuous ("smooth") escape count
//   mu = n + 1 - log2(log|z_n|),
// which removes the integer banding: for large R, log|z| roughly doubles per
// step after escape, so the fractional part tracks how far past the bailout
// the orbit went.
//
// Julia mode fixes c and iterates from z_0 = pixel. c follows the mouse,
// read through the Mandelbrot view you were looking at when you switched,
// which is drawn as an inset with c marked. Click to pin c.
//
// Precision: WebGL 1 fragment shaders compute in float32 (24-bit mantissa),
// so once a pixel is smaller than about 1e-7 of the coordinate magnitude the
// image breaks into blocks. That is a property of the arithmetic, not of the
// set. The 2-D canvas fallback (no WebGL) uses float64 but is far slower.
(function () {
  "use strict";

  var BAILOUT = 256;
  var MAX_LOOP = 5000;          // compile-time loop bound in the shader
  var HOME = { cx: -0.6, cy: 0, half: 1.35 };       // half-height of view
  var JULIA_HOME = { cx: 0, cy: 0, half: 1.45 };
  var INSET = 0.3;              // Julia-mode inset size, as a fraction of the plot
  var PRESETS = [
    { name: "Seahorse valley", cx: -0.7453, cy: 0.1127, half: 0.0065 },
    { name: "Elephant valley", cx: 0.2822, cy: 0.0103, half: 0.006 },
    { name: "Mini-brot", cx: -1.7686, cy: 0.0017, half: 0.0045 },
  ];

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

  function rgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (isNaN(n)) return [0.5, 0.5, 0.5];
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  function isDark() {
    return document.documentElement.getAttribute("data-theme") === "dark";
  }

  // Theme-derived palette: four stops cycled smoothly, plus the interior.
  function palette(mount) {
    var dark = isDark();
    var bg = rgb(cssVar(mount, "--card-bg", "#ffffff"));
    var accent = rgb(cssVar(mount, "--accent", "#ba3925"));
    var link = rgb(cssVar(mount, "--link", "#2156a5"));
    var deep = dark ? [0.05, 0.06, 0.09] : [0.07, 0.09, 0.2];
    var light = dark ? [0.96, 0.9, 0.78] : [1.0, 0.95, 0.82];
    return {
      stops: [deep, link, light, accent],
      inside: dark ? [0.03, 0.035, 0.05] : [0.04, 0.04, 0.06],
      bg: bg,
      marker: accent,
    };
  }

  // --- shaders ------------------------------------------------------------------
  var VS = [
    "attribute vec2 p;",
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }",
  ].join("\n");

  var FS = [
    "precision highp float;",
    "uniform vec2 u_origin;     // pixel offset of this viewport",
    "uniform vec2 u_res;        // viewport size in device pixels",
    "uniform vec2 u_center;",
    "uniform float u_scale;     // plane units per device pixel",
    "uniform int u_iter;",
    "uniform int u_julia;",
    "uniform vec2 u_c;",
    "uniform vec3 u_s0; uniform vec3 u_s1; uniform vec3 u_s2; uniform vec3 u_s3;",
    "uniform vec3 u_inside;",
    "uniform float u_mark;      // >0: draw a ring around u_c (inset)",
    "uniform vec3 u_markCol;",
    "vec3 pal(float t) {",
    "  t = fract(t) * 4.0;",
    "  float f = fract(t);",
    "  f = f * f * (3.0 - 2.0 * f);",
    "  if (t < 1.0) return mix(u_s0, u_s1, f);",
    "  if (t < 2.0) return mix(u_s1, u_s2, f);",
    "  if (t < 3.0) return mix(u_s2, u_s3, f);",
    "  return mix(u_s3, u_s0, f);",
    "}",
    "void main() {",
    "  vec2 px = gl_FragCoord.xy - u_origin - 0.5 * u_res;",
    "  vec2 q = u_center + px * u_scale;",
    "  vec2 z = u_julia == 1 ? q : vec2(0.0);",
    "  vec2 c = u_julia == 1 ? u_c : q;",
    "  float n = 0.0;",
    "  bool esc = false;",
    "  for (int i = 0; i < " + MAX_LOOP + "; i++) {",
    "    if (i >= u_iter) break;",
    "    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;",
    "    if (dot(z, z) > " + (BAILOUT * BAILOUT).toFixed(1) + ") { esc = true; break; }",
    "    n += 1.0;",
    "  }",
    "  vec3 col;",
    "  if (!esc) {",
    "    col = u_inside;",
    "  } else {",
    "    float mu = n + 1.0 - log2(log(length(z)));",
    "    col = pal(0.3 * sqrt(max(mu, 1.0)) - 0.3);",
    "  }",
    "  if (u_mark > 0.0) {",
    "    float d = length(q - u_c) / u_scale;",
    "    float ring = smoothstep(1.6 * u_mark, 0.5 * u_mark, abs(d - 5.0 * u_mark));",
    "    col = mix(col, u_markCol, ring);",
    "    col = mix(col, vec3(1.0), smoothstep(1.8 * u_mark, 0.7 * u_mark, d));",
    "  }",
    "  gl_FragColor = vec4(col, 1.0);",
    "}",
  ].join("\n");

  function makeGL(canvas) {
    var gl = null;
    try {
      gl = canvas.getContext("webgl", { antialias: false, alpha: false }) ||
        canvas.getContext("experimental-webgl", { antialias: false, alpha: false });
    } catch (e) { gl = null; }
    if (!gl) return null;
    function sh(type, src) {
      var s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(s) || "shader compile failed");
      }
      return s;
    }
    try {
      var prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
      gl.useProgram(prog);
      var buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      var loc = gl.getAttribLocation(prog, "p");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      var U = {};
      ["u_origin", "u_res", "u_center", "u_scale", "u_iter", "u_julia", "u_c",
        "u_s0", "u_s1", "u_s2", "u_s3", "u_inside", "u_mark", "u_markCol"]
        .forEach(function (n) { U[n] = gl.getUniformLocation(prog, n); });
      return { gl: gl, U: U };
    } catch (e) {
      return null;
    }
  }

  // --- CPU fallback (float64), same maths as the shader -------------------------
  function palJS(stops, t) {
    t = (t - Math.floor(t)) * 4;
    var k = Math.floor(t), f = t - k;
    f = f * f * (3 - 2 * f);
    var a = stops[k % 4], b = stops[(k + 1) % 4];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  function renderCPU(ctx2, w, h, step, view, iter, julia, c, pal) {
    var bw = Math.ceil(w / step), bh = Math.ceil(h / step);
    var img = ctx2.createImageData(bw, bh);
    var d = img.data, B2 = BAILOUT * BAILOUT;
    for (var j = 0; j < bh; j++) {
      for (var i = 0; i < bw; i++) {
        var qx = view.cx + ((i + 0.5) * step - w / 2) * view.scale;
        var qy = view.cy - ((j + 0.5) * step - h / 2) * view.scale;
        var zx = julia ? qx : 0, zy = julia ? qy : 0;
        var cx = julia ? c[0] : qx, cy = julia ? c[1] : qy;
        var n = 0, esc = false;
        for (; n < iter; n++) {
          var t = zx * zx - zy * zy + cx;
          zy = 2 * zx * zy + cy; zx = t;
          if (zx * zx + zy * zy > B2) { esc = true; break; }
        }
        var col;
        if (!esc) col = pal.inside;
        else {
          var mu = n + 1 - Math.log2(Math.log(Math.sqrt(zx * zx + zy * zy)));
          col = palJS(pal.stops, 0.3 * Math.sqrt(Math.max(mu, 1)) - 0.3);
        }
        var o = (j * bw + i) * 4;
        d[o] = col[0] * 255; d[o + 1] = col[1] * 255; d[o + 2] = col[2] * 255;
        d[o + 3] = 255;
      }
    }
    return img;
  }

  // --- widget --------------------------------------------------------------------
  function build(mount) {
    var julia = false, iter = 300, locked = false;
    var view = { cx: HOME.cx, cy: HOME.cy, scale: 0 };   // scale: units / device px
    var mView = null;              // Mandelbrot view frozen when entering Julia
    var c = [-0.7269, 0.1889];
    var W = 0, Hh = 0, dpr = 1, dirty = true, raf = null, interacting = false;
    var idleTimer = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Mandelbrot explorer";
    var meta = el("div", "widget-meta", head);
    mount.parentNode.insertBefore(head, mount);

    var canvas = el("canvas", "widget-plot", mount);
    canvas.setAttribute("aria-label",
      "Mandelbrot set: drag to pan, scroll or pinch to zoom");
    canvas.style.borderRadius = "6px";

    var engine = makeGL(canvas);
    var ctx2 = null, off = null;
    if (!engine) {
      ctx2 = canvas.getContext("2d");
      off = document.createElement("canvas");
    }
    meta.textContent = engine
      ? "WebGL fragment shader · float32"
      : "2-D canvas fallback · float64";

    var cols = el("div", "widget-cols", mount);
    var colA = el("div", null, cols), colB = el("div", null, cols);
    function readRow(parent, name) {
      var r = el("div", "widget-row widget-row--wide", parent);
      el("span", "widget-row-name", r).textContent = name;
      var v = el("span", "widget-row-val", r);
      v.style.width = "auto";
      v.style.flex = "1 1 auto";
      v.style.textAlign = "right";
      v.style.whiteSpace = "nowrap";
      v.style.overflow = "hidden";
      v.style.textOverflow = "ellipsis";
      return v;
    }
    var rRe = readRow(colA, "centre re");
    var rIm = readRow(colA, "centre im");
    var rC = readRow(colA, "c");
    var rZoom = readRow(colB, "zoom");
    var rPx = readRow(colB, "pixel size");
    var rIt = readRow(colB, "iterations");

    var sRow = el("div", "widget-slider-row", mount);
    sRow.style.marginTop = "0.8rem";
    el("span", null, sRow).textContent = "max iter";
    var sIn = el("input", "widget-slider", sRow);
    sIn.type = "range"; sIn.min = "20"; sIn.max = String(MAX_LOOP);
    sIn.step = "10"; sIn.value = String(iter);
    sIn.setAttribute("aria-label", "Maximum iterations");
    var sVal = el("span", "widget-slider-val", sRow);
    sIn.addEventListener("input", function () {
      iter = Number(sIn.value);
      request();
    });

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
    var modeBtn = button("Julia mode", function () { setJulia(!julia); }, true);
    var lockBtn = button("Pin c", function () { locked = !locked; request(); });
    button("Reset view", function () {
      var h = julia ? JULIA_HOME : HOME;
      setView(h.cx, h.cy, h.half);
    });
    PRESETS.forEach(function (p) {
      button(p.name, function () {
        if (julia) setJulia(false);
        setView(p.cx, p.cy, p.half);
        if (iter < 600) { iter = 600; sIn.value = "600"; }
        request();
      });
    });

    // --- view maths -------------------------------------------------------------
    function setView(cx, cy, half) {
      view.cx = cx; view.cy = cy;
      view.scale = (2 * half) / Math.max(1, Hh);
      request();
    }

    function planeAt(clientX, clientY, v) {
      var r = canvas.getBoundingClientRect();
      var x = (clientX - r.left) * dpr, y = (clientY - r.top) * dpr;
      v = v || view;
      return [v.cx + (x - W / 2) * v.scale, v.cy - (y - Hh / 2) * v.scale];
    }

    function zoomAt(clientX, clientY, factor) {
      var before = planeAt(clientX, clientY);
      // float32 cannot resolve much below one ulp per pixel, so the GL path
      // stops a little past that point (deep enough to see the blocks); the
      // float64 fallback can go much further
      var mag = Math.max(Math.abs(view.cx), Math.abs(view.cy), 0.25);
      var minScale = engine ? 0.5 * 1.19e-7 * mag : 1e-15;
      view.scale = Math.max(minScale, Math.min(0.05, view.scale * factor));
      var after = planeAt(clientX, clientY);
      view.cx += before[0] - after[0];
      view.cy += before[1] - after[1];
      request();
    }

    function setJulia(on) {
      julia = on;
      if (on) {
        mView = { cx: view.cx, cy: view.cy, scale: view.scale };
        setView(JULIA_HOME.cx, JULIA_HOME.cy, JULIA_HOME.half);
      } else if (mView) {
        view.cx = mView.cx; view.cy = mView.cy; view.scale = mView.scale;
      }
      locked = false;
      modeBtn.textContent = julia ? "Mandelbrot mode" : "Julia mode";
      request();
    }

    // --- sizing -----------------------------------------------------------------
    function resize() {
      var cw = Math.max(200, canvas.clientWidth || mount.clientWidth);
      var ch = Math.round(Math.max(240, Math.min(520, cw * 0.64)));
      canvas.style.height = ch + "px";
      var oldDpr = dpr;
      dpr = Math.min(2, window.devicePixelRatio || 1);
      // keep plane units per CSS pixel fixed if the pixel ratio changes
      if (view.scale && oldDpr !== dpr) view.scale *= oldDpr / dpr;
      if (mView && oldDpr !== dpr) mView.scale *= oldDpr / dpr;
      W = Math.round(cw * dpr); Hh = Math.round(ch * dpr);
      if (!engine) {
        // the CPU path renders at reduced resolution; keep the backing store
        // at CSS size so the upscale is cheap
        canvas.width = cw; canvas.height = ch;
      } else {
        canvas.width = W; canvas.height = Hh;
      }
      if (!view.scale) setView(HOME.cx, HOME.cy, HOME.half);
      request();
    }

    // --- rendering --------------------------------------------------------------
    function request() {
      dirty = true;
      if (!raf) raf = requestAnimationFrame(frame);
    }

    function frame() {
      raf = null;
      if (!dirty) return;
      dirty = false;
      var pal = palette(mount);
      if (engine) drawGL(pal); else drawCPU(pal);
      readout();
    }

    function drawGL(pal) {
      var gl = engine.gl, U = engine.U;
      gl.viewport(0, 0, W, Hh);
      gl.disable(gl.SCISSOR_TEST);
      gl.uniform3fv(U.u_s0, pal.stops[0]);
      gl.uniform3fv(U.u_s1, pal.stops[1]);
      gl.uniform3fv(U.u_s2, pal.stops[2]);
      gl.uniform3fv(U.u_s3, pal.stops[3]);
      gl.uniform3fv(U.u_inside, pal.inside);
      gl.uniform3fv(U.u_markCol, pal.marker);
      pass(gl, U, 0, 0, W, Hh, view, julia, 0);
      if (julia && mView) {
        // inset: the Mandelbrot frame c is read from, with c marked
        var iw = Math.round(W * INSET), ih = Math.round(Hh * INSET);
        var m = Math.round(8 * dpr), bw = Math.max(1, Math.round(dpr));
        var x0 = W - iw - m, y0 = m;               // GL origin is bottom-left
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(x0 - bw, y0 - bw, iw + 2 * bw, ih + 2 * bw);
        gl.clearColor(pal.bg[0], pal.bg[1], pal.bg[2], 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.scissor(x0, y0, iw, ih);
        gl.viewport(x0, y0, iw, ih);
        var insetView = { cx: mView.cx, cy: mView.cy, scale: mView.scale / INSET };
        pass(gl, U, x0, y0, iw, ih, insetView, false, dpr);
        gl.disable(gl.SCISSOR_TEST);
      }
    }

    function pass(gl, U, ox, oy, w, h, v, jul, mark) {
      gl.uniform2f(U.u_origin, ox, oy);
      gl.uniform2f(U.u_res, w, h);
      gl.uniform2f(U.u_center, v.cx, v.cy);
      gl.uniform1f(U.u_scale, v.scale);
      gl.uniform1i(U.u_iter, mark ? Math.min(iter, 400) : iter);
      gl.uniform1i(U.u_julia, jul ? 1 : 0);
      gl.uniform2f(U.u_c, c[0], c[1]);
      gl.uniform1f(U.u_mark, mark);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    function drawCPU(pal) {
      var cw = canvas.width, ch = canvas.height;
      // coarse while dragging, full resolution once the view settles on
      // small canvases
      var step = interacting ? 4 : (cw * ch <= 250000 ? 1 : 2);
      var v = { cx: view.cx, cy: view.cy, scale: view.scale * dpr };
      var img = renderCPU(ctx2, cw, ch, step, v, iter, julia, c, pal);
      off.width = img.width; off.height = img.height;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx2.imageSmoothingEnabled = false;
      ctx2.drawImage(off, 0, 0, img.width * step, img.height * step);
      if (julia && mView) {
        var iw = Math.round(cw * INSET), ih = Math.round(ch * INSET), m = 8;
        var x0 = cw - iw - m, y0 = ch - ih - m;
        var iv = { cx: mView.cx, cy: mView.cy, scale: mView.scale * dpr / INSET };
        var inset = renderCPU(ctx2, iw, ih, 2, iv, Math.min(iter, 300), false, c, pal);
        off.width = inset.width; off.height = inset.height;
        off.getContext("2d").putImageData(inset, 0, 0);
        ctx2.fillStyle = "rgb(" + pal.bg.map(function (x) { return Math.round(x * 255); }) + ")";
        ctx2.fillRect(x0 - 1, y0 - 1, iw + 2, ih + 2);
        ctx2.drawImage(off, x0, y0, inset.width * 2, inset.height * 2);
        var mx = x0 + iw / 2 + (c[0] - iv.cx) / iv.scale;
        var my = y0 + ih / 2 - (c[1] - iv.cy) / iv.scale;
        ctx2.strokeStyle = "rgb(" + pal.marker.map(function (x) { return Math.round(x * 255); }) + ")";
        ctx2.lineWidth = 1.5;
        ctx2.beginPath();
        ctx2.arc(mx, my, 5, 0, 6.284);
        ctx2.stroke();
      }
    }

    function fmt(v, digits) {
      return (v < 0 ? "−" : "+") + Math.abs(v).toFixed(digits);
    }

    function readout() {
      var pxCss = view.scale * dpr;                   // plane units per CSS px
      var digits = Math.max(3, Math.min(16, Math.ceil(-Math.log10(pxCss)) + 1));
      rRe.textContent = fmt(view.cx, digits);
      rIm.textContent = fmt(view.cy, digits) + "i";
      var home = julia ? JULIA_HOME : HOME;
      var zoom = (2 * home.half / Math.max(1, Hh)) / view.scale;
      rZoom.textContent = "×" + (zoom < 1e4 ? zoom.toFixed(zoom < 10 ? 2 : 0)
        : zoom.toExponential(2));
      rPx.textContent = pxCss.toExponential(2);
      rC.textContent = julia
        ? fmt(c[0], 4) + (c[1] < 0 ? " − " : " + ") + Math.abs(c[1]).toFixed(4)
          + "i" + (locked ? " (pinned)" : "")
        : "—";
      sVal.textContent = String(iter);
      rIt.textContent = String(iter);
      lockBtn.disabled = !julia;
      lockBtn.textContent = locked ? "Unpin c" : "Pin c";

      // float32 has a 24-bit mantissa: spacing ~ 1.2e-7 * |coordinate|.
      // Below a few of those per device pixel the image goes blocky.
      var mag = Math.max(Math.abs(view.cx), Math.abs(view.cy), 0.25);
      var ratio = view.scale / (1.19e-7 * mag);
      if (engine && ratio < 4) {
        note.textContent = "Float32 limit: a pixel is now only " + ratio.toFixed(1)
          + " float32 steps wide at this coordinate, so neighbouring pixels "
          + "round to the same point and the image turns into blocks. Going "
          + "deeper needs double or arbitrary precision (perturbation theory), "
          + "which a WebGL 1 shader does not have.";
      } else if (julia) {
        note.textContent = "Julia set for the c shown in the inset. Move the "
          + "mouse to change c (on touch screens, tap); click to pin it. c just "
          + "inside the Mandelbrot set gives a connected Julia set, just outside "
          + "gives dust.";
      } else {
        note.textContent = "Drag to pan, scroll or pinch to zoom, double-click "
          + "to zoom in. Deep zooms need more iterations: the boundary is where "
          + "orbits take longest to decide. Each pixel's colour is the smooth "
          + "escape count n + 1 − log₂ log|zₙ|.";
      }
    }

    // --- interaction ------------------------------------------------------------
    var pointers = {}, downAt = null, moved = false, pinch = null;

    function busy() {
      interacting = true;
      clearTimeout(idleTimer);
      idleTimer = setTimeout(function () { interacting = false; request(); }, 160);
    }

    function setCFrom(ev) {
      if (!julia || !mView || locked) return;
      // read c through the Mandelbrot view frozen on entering Julia mode
      var p = planeAt(ev.clientX, ev.clientY, mView);
      c[0] = p[0]; c[1] = p[1];
      request();
    }

    canvas.addEventListener("pointerdown", function (ev) {
      canvas.setPointerCapture(ev.pointerId);
      pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
      downAt = { x: ev.clientX, y: ev.clientY, id: ev.pointerId };
      moved = false;
      var ids = Object.keys(pointers);
      if (ids.length === 2) {
        var a = pointers[ids[0]], b = pointers[ids[1]];
        pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) };
      }
    });

    canvas.addEventListener("pointermove", function (ev) {
      var p = pointers[ev.pointerId];
      if (!p) {
        if (ev.pointerType === "mouse") setCFrom(ev);
        return;
      }
      ev.preventDefault();
      var ids = Object.keys(pointers);
      if (ids.length >= 2 && pinch) {
        pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
        var a = pointers[ids[0]], b = pointers[ids[1]];
        var d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d > 0 && pinch.d > 0) {
          zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, pinch.d / d);
          pinch.d = d;
        }
        moved = true;
        busy();
        return;
      }
      var dx = ev.clientX - p.x, dy = ev.clientY - p.y;
      if (Math.abs(ev.clientX - downAt.x) + Math.abs(ev.clientY - downAt.y) > 3) {
        moved = true;
      }
      if (moved) {
        view.cx -= dx * dpr * view.scale;
        view.cy += dy * dpr * view.scale;
        busy();
        request();
      }
      pointers[ev.pointerId] = { x: ev.clientX, y: ev.clientY };
    });

    function up(ev) {
      if (!pointers[ev.pointerId]) return;
      delete pointers[ev.pointerId];
      if (Object.keys(pointers).length < 2) pinch = null;
      if (!moved && julia && ev.type === "pointerup") {
        if (ev.pointerType === "mouse") {
          locked = !locked;
          if (!locked) setCFrom(ev);
        } else {
          var was = locked;
          locked = false;
          setCFrom(ev);
          locked = was;
        }
        request();
      }
    }
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);

    canvas.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
      zoomAt(ev.clientX, ev.clientY, Math.exp(dy * 0.0015));
      busy();
    }, { passive: false });

    canvas.addEventListener("dblclick", function (ev) {
      ev.preventDefault();
      zoomAt(ev.clientX, ev.clientY, ev.shiftKey ? 2 : 0.5);
    });

    if (engine) {
      canvas.addEventListener("webglcontextlost", function (ev) {
        ev.preventDefault();
        note.textContent = "The WebGL context was lost; reload the page to "
          + "restore the renderer.";
      });
    }

    // --- lifecycle --------------------------------------------------------------
    resize();
    window.addEventListener("resize", resize);
    new MutationObserver(request).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="mandelbrot"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

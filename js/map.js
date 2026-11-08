// Concept map for /map.html: a force-directed graph of the posts, drawn from
// /map.json (built by compute_map in src/main.ts).
//
// Layout runs in a fixed world space and is fitted to the stage; drawing maps
// world to screen coordinates every frame, so nodes and labels keep a
// readable pixel size at any zoom. Labels are placed greedily (most connected
// first) and skipped where they would collide, which is what keeps a phone
// screen from turning into label soup: zooming in reveals the rest.
//
// Colours come from CSS classes (`.map-node--<theme>` etc. in main.css), so
// the graph follows the dark-mode toggle without redrawing.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var WORLD_W = 1000, WORLD_H = 700;
  var TICKS = 500;                 // layout steps before the first paint
  // Spring rest length and stiffness per edge type. Tag edges are weak but
  // always on, so the layout does not jump when they are toggled.
  var SPRING = {
    prereq: { len: 70, k: 0.35 },
    series: { len: 80, k: 0.25 },
    tag: { len: 150, k: 0.012 },
  };
  var LABEL_FONT = 12, LABEL_LINE = 14, LABEL_CHARS = 20;

  var stage = document.querySelector(".map-stage");
  var svg = stage && stage.querySelector(".map-svg");
  if (!svg || !window.fetch) return;

  fetch("/map.json").then(function (r) { return r.json(); }).then(init);

  function el(name, attrs, parent) {
    var e = document.createElementNS(SVG_NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // "Title: subtitle (aside)" -> "Title", wrapped to at most two short lines.
  function labelLines(title) {
    var t = title.split(/[:(?]| - | — /)[0].trim() || title;
    var words = t.split(/\s+/), lines = [""];
    for (var i = 0; i < words.length; i++) {
      var cur = lines[lines.length - 1];
      var next = cur ? cur + " " + words[i] : words[i];
      if (next.length <= LABEL_CHARS || !cur) {
        lines[lines.length - 1] = next;
      } else if (lines.length < 2) {
        lines.push(words[i]);
      } else {
        lines[1] = lines[1] + " …";
        break;
      }
    }
    return lines.map(function (l) {
      return l.length > LABEL_CHARS + 2 ? l.slice(0, LABEL_CHARS) + "…" : l;
    });
  }

  function init(data) {
    document.querySelector(".map-controls").hidden = false;
    stage.hidden = false;

    var themeIndex = {};
    data.themes.forEach(function (t, i) { themeIndex[t.key] = i; });
    var themeTitle = {};
    data.themes.forEach(function (t) { themeTitle[t.key] = t.title; });

    // Each theme gets an anchor on an ellipse; nodes start near theme anchors
    // (deterministically, so the layout is the same on every visit).
    var anchors = data.themes.map(function (_, i) {
      var a = -Math.PI / 2 + i * 2 * Math.PI / data.themes.length + Math.PI / 4;
      return { x: WORLD_W / 2 + 260 * Math.cos(a), y: WORLD_H / 2 + 190 * Math.sin(a) };
    });

    var byId = {};
    var nodes = data.nodes.map(function (n, i) {
      var anchor = anchors[themeIndex[n.theme]] || { x: WORLD_W / 2, y: WORLD_H / 2 };
      var a = i * 2.39996; // golden angle
      var node = {
        data: n,
        x: anchor.x + 60 * Math.cos(a) * Math.sqrt((i % 7) + 1) / 2,
        y: anchor.y + 60 * Math.sin(a) * Math.sqrt((i % 7) + 1) / 2,
        vx: 0, vy: 0, fx: null, fy: null,
        anchor: anchor,
        r: 5 + 2.6 * Math.sqrt(n.readingTime),
        degree: 0,
        lines: labelLines(n.title),
        neighbours: [],
      };
      byId[n.id] = node;
      return node;
    });
    var edges = data.edges.map(function (e) {
      var edge = { data: e, type: e.type, s: byId[e.source], t: byId[e.target] };
      if (e.type !== "tag") { edge.s.degree++; edge.t.degree++; }
      return edge;
    });

    // ---- layout ---------------------------------------------------------
    var alpha = 1;
    function tick() {
      var i, j, a, b, dx, dy, d2, d, f;
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        for (j = i + 1; j < nodes.length; j++) {
          b = nodes[j];
          dx = b.x - a.x; dy = b.y - a.y;
          d2 = Math.max(dx * dx + dy * dy, 1);
          d = Math.sqrt(d2);
          f = 1800 / d2;                              // charge
          var min = a.r + b.r + 26;                   // collision + label room
          if (d < min) f += (min - d) / d * 0.5;
          f *= alpha;
          a.vx -= dx / d * f; a.vy -= dy / d * f;
          b.vx += dx / d * f; b.vy += dy / d * f;
        }
      }
      for (i = 0; i < edges.length; i++) {
        var e = edges[i], sp = SPRING[e.type];
        dx = e.t.x - e.s.x; dy = e.t.y - e.s.y;
        d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        f = (d - sp.len - e.s.r - e.t.r) / d * sp.k * alpha;
        e.s.vx += dx * f / 2; e.s.vy += dy * f / 2;
        e.t.vx -= dx * f / 2; e.t.vy -= dy * f / 2;
      }
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        a.vx += (a.anchor.x - a.x) * 0.012 * alpha;  // theme gravity
        a.vy += (a.anchor.y - a.y) * 0.012 * alpha;
        a.vx += (WORLD_W / 2 - a.x) * 0.002 * alpha; // keep it centred
        a.vy += (WORLD_H / 2 - a.y) * 0.002 * alpha;
        if (a.fx !== null) {
          a.x = a.fx; a.y = a.fy; a.vx = a.vy = 0;
        } else {
          a.vx *= 0.6; a.vy *= 0.6;
          a.x += a.vx; a.y += a.vy;
        }
      }
      alpha *= 0.99;
    }
    for (var n = 0; n < TICKS; n++) tick();

    // ---- scene ----------------------------------------------------------
    var defs = el("defs", {}, svg);
    ["", "--hot"].forEach(function (mod) {
      var m = el("marker", {
        id: "map-arrow" + mod, viewBox: "0 0 10 10", refX: "9", refY: "5",
        markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse",
        markerUnits: "userSpaceOnUse",
      }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", class: "map-arrow" + (mod ? " map-arrow--hot" : "") }, m);
    });
    var gEdges = el("g", {}, svg);
    var gNodes = el("g", {}, svg);
    var gLabels = el("g", {}, svg);

    edges.forEach(function (e) {
      e.el = el("line", { class: "map-edge map-edge--" + e.type }, gEdges);
      if (e.type === "prereq") e.el.setAttribute("marker-end", "url(#map-arrow)");
    });
    nodes.forEach(function (nd) {
      nd.el = el("circle", { class: "map-node map-node--" + nd.data.theme }, gNodes);
      nd.el.__node = nd;
      var t = el("text", { class: "map-label", "font-size": LABEL_FONT }, gLabels);
      nd.lines.forEach(function (line, i) {
        var span = el("tspan", { x: 0, dy: i ? LABEL_LINE : 0 }, t);
        span.textContent = line;
      });
      nd.label = t;
    });

    // Label widths for placement, measured once with the label font.
    var ctx = document.createElement("canvas").getContext("2d");
    ctx.font = LABEL_FONT + "px 'Open Sans', sans-serif";
    nodes.forEach(function (nd) {
      nd.labelW = Math.max.apply(null, nd.lines.map(function (l) {
        return ctx.measureText(l).width;
      })) + 4;
      nd.labelH = nd.lines.length * LABEL_LINE;
    });

    // ---- state ----------------------------------------------------------
    var view = { k: 1, x: 0, y: 0 };
    var width = 0, height = 0;
    var hiddenThemes = {};
    var hiddenEdges = { tag: true };
    var hover = null, selected = null;

    function visibleNode(nd) { return !hiddenThemes[nd.data.theme]; }
    function visibleEdge(e) {
      return !hiddenEdges[e.type] && visibleNode(e.s) && visibleNode(e.t);
    }
    function nodeRadius(nd) {
      return nd.r * Math.min(1.5, Math.max(0.75, Math.sqrt(view.k)));
    }

    function resize() {
      width = stage.clientWidth;
      height = Math.round(Math.min(Math.max(width * 0.68, 420), 720, window.innerHeight * 0.8));
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    }

    function fit() {
      var vis = nodes.filter(visibleNode);
      if (!vis.length) return;
      var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      vis.forEach(function (nd) {
        x0 = Math.min(x0, nd.x); y0 = Math.min(y0, nd.y);
        x1 = Math.max(x1, nd.x); y1 = Math.max(y1, nd.y);
      });
      var pad = width < 500 ? 24 : 60;
      var k = Math.min((width - 2 * pad) / Math.max(x1 - x0, 1),
        (height - 2 * pad) / Math.max(y1 - y0, 1), 1.6);
      view.k = k;
      view.x = width / 2 - k * (x0 + x1) / 2;
      view.y = height / 2 - k * (y0 + y1) / 2;
    }

    function sx(x) { return x * view.k + view.x; }
    function sy(y) { return y * view.k + view.y; }

    // Focus = the node under the pointer (or tapped), plus its neighbours
    // along the edge types currently shown.
    function focusSet() {
      var f = hover || selected;
      if (!f) return null;
      var set = {};
      set[f.data.id] = true;
      edges.forEach(function (e) {
        if (!visibleEdge(e)) return;
        if (e.s === f) set[e.t.data.id] = true;
        if (e.t === f) set[e.s.data.id] = true;
      });
      return set;
    }

    function draw() {
      var focus = focusSet();
      var f = hover || selected;
      svg.classList.toggle("is-focus", !!focus);

      edges.forEach(function (e) {
        var show = visibleEdge(e);
        e.el.style.display = show ? "" : "none";
        if (!show) return;
        var x1 = sx(e.s.x), y1 = sy(e.s.y), x2 = sx(e.t.x), y2 = sy(e.t.y);
        var dx = x2 - x1, dy = y2 - y1, d = Math.sqrt(dx * dx + dy * dy) || 1;
        var r1 = nodeRadius(e.s) + 1, r2 = nodeRadius(e.t) + (e.type === "prereq" ? 3 : 1);
        e.el.setAttribute("x1", x1 + dx / d * r1);
        e.el.setAttribute("y1", y1 + dy / d * r1);
        e.el.setAttribute("x2", x2 - dx / d * r2);
        e.el.setAttribute("y2", y2 - dy / d * r2);
        var hot = !!f && (e.s === f || e.t === f);
        e.el.classList.toggle("is-hot", hot);
        if (e.type === "prereq") {
          e.el.setAttribute("marker-end", hot ? "url(#map-arrow--hot)" : "url(#map-arrow)");
        }
      });

      nodes.forEach(function (nd) {
        var show = visibleNode(nd);
        nd.el.style.display = show ? "" : "none";
        nd.el.setAttribute("cx", sx(nd.x));
        nd.el.setAttribute("cy", sy(nd.y));
        nd.el.setAttribute("r", nodeRadius(nd));
        nd.el.classList.toggle("is-hot", !!focus && !!focus[nd.data.id]);
        nd.el.classList.toggle("is-selected", nd === f);
      });

      placeLabels(focus);
    }

    // Greedy label placement: focused nodes first, then by degree and
    // reading time. Try below / above / right / left of the node; skip the
    // label entirely if every slot collides.
    function placeLabels(focus) {
      var boxes = [];
      var circles = nodes.filter(visibleNode).map(function (nd) {
        return { nd: nd, x: sx(nd.x), y: sy(nd.y), r: nodeRadius(nd) };
      });
      var order = nodes.filter(visibleNode).sort(function (a, b) {
        var fa = focus && focus[a.data.id] ? 1 : 0, fb = focus && focus[b.data.id] ? 1 : 0;
        return fb - fa || b.degree - a.degree || b.data.readingTime - a.data.readingTime;
      });
      nodes.forEach(function (nd) { nd.label.style.display = "none"; });
      order.forEach(function (nd) {
        if (focus && !focus[nd.data.id]) return;
        var cx = sx(nd.x), cy = sy(nd.y), r = nodeRadius(nd);
        var w = nd.labelW, h = nd.labelH;
        var g = r * 0.7 + 2; // diagonal offset clearing the circle
        var slots = [
          { x: cx - w / 2, y: cy + r + 3 },
          { x: cx - w / 2, y: cy - r - 3 - h },
          { x: cx + r + 4, y: cy - h / 2 },
          { x: cx - r - 4 - w, y: cy - h / 2 },
          { x: cx + g, y: cy + g },
          { x: cx - g - w, y: cy + g },
          { x: cx + g, y: cy - g - h },
          { x: cx - g - w, y: cy - g - h },
        ];
        for (var i = 0; i < slots.length; i++) {
          var b = { x: slots[i].x, y: slots[i].y, w: w, h: h };
          if (b.x < 2 || b.y < 2 || b.x + w > width - 2 || b.y + h > height - 2) continue;
          if (boxes.some(function (o) { return overlaps(b, o); })) continue;
          if (circles.some(function (c) { return c.nd !== nd && hitsCircle(b, c); })) continue;
          boxes.push(b);
          nd.label.style.display = "";
          nd.label.setAttribute("transform",
            "translate(" + (b.x + w / 2) + "," + (b.y + LABEL_FONT - 1) + ")");
          nd.label.classList.toggle("is-hot", !!focus);
          return;
        }
      });
    }

    function overlaps(a, b) {
      return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    }
    function hitsCircle(b, c) {
      var nx = Math.max(b.x, Math.min(c.x, b.x + b.w));
      var ny = Math.max(b.y, Math.min(c.y, b.y + b.h));
      return (nx - c.x) * (nx - c.x) + (ny - c.y) * (ny - c.y) < c.r * c.r;
    }

    // Re-run the simulation briefly after a drag so neighbours settle.
    var running = false;
    function reheat() {
      alpha = Math.max(alpha, 0.3);
      if (running) return;
      running = true;
      (function frame() {
        for (var i = 0; i < 2; i++) tick();
        draw();
        if (alpha > 0.01 || dragging) requestAnimationFrame(frame);
        else running = false;
      })();
    }

    // ---- info card ------------------------------------------------------
    var card = stage.querySelector(".map-card");
    function showCard(nd, touch) {
      if (!nd) { card.hidden = true; return; }
      var d = nd.data;
      var before = [], after = [];
      edges.forEach(function (e) {
        if (e.type !== "prereq") return;
        if (e.t === nd) before.push(e.s.data.title);
        if (e.s === nd) after.push(e.t.data.title);
      });
      card.innerHTML = "";
      var title = document.createElement("a");
      title.className = "map-card__title";
      title.href = d.url;
      title.textContent = d.title;
      card.appendChild(title);
      var meta = document.createElement("p");
      meta.className = "muted";
      meta.textContent = d.date + " · " + d.readingTime + " min read · " + themeTitle[d.theme];
      card.appendChild(meta);
      if (d.tags.length) {
        var tags = document.createElement("p");
        tags.className = "muted";
        tags.textContent = d.tags.map(function (t) { return "#" + t; }).join(" ");
        card.appendChild(tags);
      }
      [["Read first", before], ["Read next", after]].forEach(function (row) {
        if (!row[1].length) return;
        var p = document.createElement("p");
        p.className = "map-card__row";
        p.textContent = row[0] + ": " + row[1].join(", ");
        card.appendChild(p);
      });
      var hint = document.createElement("p");
      hint.className = "muted";
      hint.textContent = touch ? "Tap again to open." : "Click to open.";
      card.appendChild(hint);
      card.hidden = false;
    }

    // ---- interaction ----------------------------------------------------
    var pointers = {};
    var dragging = null, dragMoved = false, downNode = null, pinch = null;
    var pan = null;

    function local(ev) {
      var rect = svg.getBoundingClientRect();
      return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
    }
    function zoomAt(px, py, factor) {
      var k = Math.min(6, Math.max(0.2, view.k * factor));
      view.x = px - (px - view.x) * k / view.k;
      view.y = py - (py - view.y) * k / view.k;
      view.k = k;
      draw();
    }

    svg.addEventListener("pointerdown", function (ev) {
      var p = local(ev);
      pointers[ev.pointerId] = p;
      svg.setPointerCapture(ev.pointerId);
      var ids = Object.keys(pointers);
      if (ids.length === 2) {
        // Second finger: switch from drag/pan to pinch zoom.
        if (dragging) { dragging.fx = dragging.fy = null; dragging = null; }
        pan = null; downNode = null;
        var a = pointers[ids[0]], b = pointers[ids[1]];
        pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) };
        return;
      }
      var nd = ev.target.__node;
      dragMoved = false;
      if (nd) {
        downNode = nd;
        dragging = nd;
        dragging.start = p;
      } else {
        pan = { x: p.x, y: p.y, vx: view.x, vy: view.y };
      }
    });

    svg.addEventListener("pointermove", function (ev) {
      var p = local(ev);
      if (pointers[ev.pointerId]) pointers[ev.pointerId] = p;
      if (pinch) {
        var ids = Object.keys(pointers);
        if (ids.length < 2) return;
        var a = pointers[ids[0]], b = pointers[ids[1]];
        var d = Math.hypot(a.x - b.x, a.y - b.y);
        zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d / pinch.d);
        pinch.d = d;
        return;
      }
      if (dragging) {
        if (!dragMoved && Math.hypot(p.x - dragging.start.x, p.y - dragging.start.y) < 5) return;
        dragMoved = true;
        dragging.fx = (p.x - view.x) / view.k;
        dragging.fy = (p.y - view.y) / view.k;
        reheat();
        return;
      }
      if (pan) {
        if (Math.hypot(p.x - pan.x, p.y - pan.y) > 3) dragMoved = true;
        view.x = pan.vx + p.x - pan.x;
        view.y = pan.vy + p.y - pan.y;
        draw();
        return;
      }
      if (ev.pointerType === "mouse") {
        var nd = ev.target.__node || null;
        if (nd !== hover) {
          hover = nd;
          svg.style.cursor = nd ? "pointer" : "";
          showCard(hover || selected, false);
          draw();
        }
      }
    });

    function release(ev) {
      delete pointers[ev.pointerId];
      if (pinch) {
        if (Object.keys(pointers).length === 0) pinch = null;
        return;
      }
      var nd = downNode;
      if (dragging) {
        dragging.fx = dragging.fy = null;
        dragging = null;
        reheat();
      }
      downNode = null;
      if (ev.type === "pointercancel") { pan = null; return; }
      if (nd && !dragMoved) {
        // Mouse: click opens. Touch: first tap selects, second tap opens.
        if (ev.pointerType === "mouse" || selected === nd) {
          if (ev.ctrlKey || ev.metaKey) window.open(nd.data.url, "_blank");
          else location.href = nd.data.url;
          return;
        }
        selected = nd;
        showCard(nd, true);
        draw();
      } else if (pan && !dragMoved) {
        selected = null;
        showCard(hover, false);
        draw();
      }
      pan = null;
    }
    svg.addEventListener("pointerup", release);
    svg.addEventListener("pointercancel", release);
    svg.addEventListener("pointerleave", function (ev) {
      if (ev.pointerType !== "mouse" || dragging || pan) return;
      hover = null;
      showCard(selected, false);
      draw();
    });

    svg.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var p = local(ev);
      zoomAt(p.x, p.y, Math.exp(-ev.deltaY * 0.0015));
    }, { passive: false });

    stage.querySelectorAll("[data-zoom]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var z = btn.getAttribute("data-zoom");
        if (z === "reset") { fit(); draw(); return; }
        zoomAt(width / 2, height / 2, z === "in" ? 1.3 : 1 / 1.3);
      });
    });

    document.querySelectorAll(".map-filter").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var key = btn.getAttribute("data-theme-key");
        hiddenThemes[key] = !hiddenThemes[key];
        btn.setAttribute("aria-pressed", hiddenThemes[key] ? "false" : "true");
        if (selected && !visibleNode(selected)) { selected = null; showCard(null); }
        draw();
      });
    });
    document.querySelectorAll(".map-edge-filter").forEach(function (btn) {
      var type = btn.getAttribute("data-edge");
      btn.addEventListener("click", function () {
        hiddenEdges[type] = !hiddenEdges[type];
        btn.setAttribute("aria-pressed", hiddenEdges[type] ? "false" : "true");
        draw();
      });
    });

    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        var w = stage.clientWidth;
        if (w === width) return;
        resize(); fit(); draw();
      }, 100);
    });

    resize();
    fit();
    draw();
  }
})();

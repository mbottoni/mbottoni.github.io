// Tabular Q-learning on the FrozenLake 4x4 map, learning in the browser.
//
// Embedded from a post with:
//   {cap="..."}
//   ::: widget-gridworld
//   :::
//
// The default layout is gym's FrozenLake-v1 4x4 map and the reward structure is
// the same: +1 for reaching the goal, 0 everywhere else, and falling in a hole
// ends the episode. Transitions here are deterministic rather than slippery --
// slippery converges too slowly to watch, and the value gradient is the point.
//
// Cell shade is V(s) = max_a Q(s,a); the arrow is the greedy action.
(function () {
  "use strict";

  var SIDE = 4;
  var NS = SIDE * SIDE;
  var NA = 4;                              // up, right, down, left
  var DR = [-1, 0, 1, 0];
  var DC = [0, 1, 0, -1];

  var ALPHA = 0.5, GAMMA = 0.95;
  var EPS0 = 1.0, EPS_MIN = 0.05, EPS_DECAY = 0.99;
  var MAX_STEPS = 100;
  var EPISODES_PER_FRAME = 4;
  var TRAIN_BATCH = 400;

  // Small optimistic initialisation. With Q starting at exactly zero and reward
  // only at the goal, epsilon decays faster than successful trajectories
  // accumulate and the greedy policy solves this map only ~14% of the time in
  // any budget short enough to watch. At 0.05 an unvisited action always looks
  // slightly better than a visited dead end, which is enough to drive
  // systematic exploration: 100% solved, usually by episode 40. It is still
  // small enough that the goal-driven gradient (1, 0.95, 0.90, ...) dominates
  // the picture.
  var OPTIMISTIC = 0.05;

  function freshQ() {
    var q = new Float64Array(NS * NA);
    q.fill(OPTIMISTIC);
    return q;
  }

  var DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
  ].join("").split("");

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

  function mix(a, b, t) {
    function hex(h) {
      h = h.replace("#", "");
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      var n = parseInt(h, 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var A = hex(a), B = hex(b);
    return "rgb(" + A.map(function (v, i) {
      return Math.round(v + (B[i] - v) * t);
    }).join(",") + ")";
  }

  function build(mount) {
    var map = DEFAULT_MAP.slice();
    var Q = freshQ();
    var episodes = 0, eps = EPS0;
    var wins = [];                          // rolling outcome window
    var training = false, remaining = 0, raf = null;
    var agent = -1, runTimer = null;

    var head = el("div", "widget-head");
    el("div", "widget-title", head).textContent = "Q-learning on FrozenLake";
    el("div", "widget-meta", head).textContent = "16 states · 4 actions";
    mount.parentNode.insertBefore(head, mount);

    var body = el("div", "widget-body", mount);
    var stage = el("div", "widget-stage", body);
    var canvas = el("canvas", "widget-canvas", stage);
    canvas.setAttribute("aria-label",
      "Gridworld: cell shade is the learned state value, the arrow is the greedy action");
    var side = el("div", "widget-side", body);

    function stat(name, fmt) {
      var row = el("div", "widget-row", side);
      var n = el("span", "widget-row-name", row);
      n.style.width = "auto";
      n.style.flex = "1 1 auto";
      n.textContent = name;
      var v = el("span", "widget-row-val", row);
      v.style.width = "auto";
      return function (x) { v.textContent = fmt(x); };
    }

    el("div", "widget-label", side).textContent = "Training";
    var setEp = stat("episodes", function (v) { return String(v); });

    function meterRow(name) {
      var row = el("div", "widget-row", side);
      var n = el("span", "widget-row-name", row);
      n.style.width = "5.4em";
      n.textContent = name;
      var meter = el("div", "widget-meter widget-meter--plain", row);
      var fill = el("span", "widget-meter-fill", meter);
      fill.style.left = "0";
      fill.setAttribute("data-pos", "");
      var val = el("span", "widget-row-val", row);
      return function (v) {
        fill.style.width = (100 * Math.max(0, Math.min(1, v))).toFixed(0) + "%";
        val.textContent = v.toFixed(2);
      };
    }
    var setWin = meterRow("success");
    var setEps = meterRow("explore ε");

    var note = el("div", "widget-note", side);
    var controls = el("div", "widget-controls", mount);

    function button(text, fn, primary) {
      var b = el("button", "widget-btn", controls);
      b.type = "button";
      b.textContent = text;
      if (primary) b.setAttribute("data-primary", "");
      b.addEventListener("click", fn);
      return b;
    }

    // --- environment -------------------------------------------------------
    function startState() {
      var i = map.indexOf("S");
      return i < 0 ? 0 : i;
    }

    function step(s, a) {
      var r = Math.floor(s / SIDE) + DR[a];
      var c = (s % SIDE) + DC[a];
      if (r < 0 || r >= SIDE || c < 0 || c >= SIDE) return { s: s, rew: 0, done: false };
      var ns = r * SIDE + c;
      if (map[ns] === "G") return { s: ns, rew: 1, done: true };
      if (map[ns] === "H") return { s: ns, rew: 0, done: true };
      return { s: ns, rew: 0, done: false };
    }

    function greedy(s) {
      var best = 0;
      for (var a = 1; a < NA; a++) if (Q[s * NA + a] > Q[s * NA + best]) best = a;
      return best;
    }

    function value(s) {
      var v = Q[s * NA];
      for (var a = 1; a < NA; a++) if (Q[s * NA + a] > v) v = Q[s * NA + a];
      return v;
    }

    function episode() {
      var s = startState(), won = 0;
      for (var t = 0; t < MAX_STEPS; t++) {
        var a = Math.random() < eps ? (Math.random() * NA) | 0 : greedy(s);
        var r = step(s, a);
        var target = r.rew + (r.done ? 0 : GAMMA * value(r.s));
        Q[s * NA + a] += ALPHA * (target - Q[s * NA + a]);
        s = r.s;
        if (r.done) { won = r.rew; break; }
      }
      episodes++;
      eps = Math.max(EPS_MIN, eps * EPS_DECAY);
      wins.push(won);
      if (wins.length > 50) wins.shift();
    }

    // --- drawing -----------------------------------------------------------
    var ctx = canvas.getContext("2d");

    function draw() {
      var w = stage.clientWidth || 260;
      var dpr = window.devicePixelRatio || 1;
      canvas.style.height = w + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);

      var base = cssVar(mount, "--card-bg", "#fff");
      var accent = cssVar(mount, "--accent", "#ba3925");
      var border = cssVar(mount, "--border", "#e2e2dd");
      var muted = cssVar(mount, "--muted", "#6a6a6a");
      var fg = cssVar(mount, "--fg", "#1a1a1a");
      var cell = w / SIDE;

      for (var s = 0; s < NS; s++) {
        var cx = (s % SIDE) * cell, cy = Math.floor(s / SIDE) * cell;
        var kind = map[s];

        if (kind === "H") {
          ctx.fillStyle = mix(base, muted, 0.55);
        } else if (kind === "G") {
          ctx.fillStyle = accent;
        } else {
          ctx.fillStyle = mix(base, accent, Math.max(0, Math.min(1, value(s))) * 0.8);
        }
        ctx.fillRect(cx + 1, cy + 1, cell - 2, cell - 2);

        ctx.strokeStyle = border;
        ctx.lineWidth = 1;
        ctx.strokeRect(cx + 0.5, cy + 0.5, cell - 1, cell - 1);

        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        if (kind === "H") {
          ctx.fillStyle = base;
          ctx.font = "600 " + (cell * 0.2 | 0) + "px 'Open Sans', sans-serif";
          ctx.fillText("hole", cx + cell / 2, cy + cell / 2);
        } else if (kind === "G") {
          ctx.fillStyle = "#fff";
          ctx.font = "600 " + (cell * 0.2 | 0) + "px 'Open Sans', sans-serif";
          ctx.fillText("goal", cx + cell / 2, cy + cell / 2);
        } else {
          var v = value(s);
          if (v > OPTIMISTIC + 1e-4) {
            arrow(cx + cell / 2, cy + cell / 2 - cell * 0.06, cell * 0.2,
              greedy(s), v > 0.45 ? "#fff" : fg);
          }
          ctx.fillStyle = v > 0.45 ? "rgba(255,255,255,0.85)" : muted;
          ctx.font = (cell * 0.15 | 0) + "px 'JetBrains Mono', monospace";
          ctx.fillText(v.toFixed(2), cx + cell / 2, cy + cell * 0.8);
          if (kind === "S") {
            ctx.fillStyle = v > 0.45 ? "rgba(255,255,255,0.85)" : muted;
            ctx.font = "600 " + (cell * 0.14 | 0) + "px 'Open Sans', sans-serif";
            ctx.fillText("start", cx + cell / 2, cy + cell * 0.2);
          }
        }
      }

      if (agent >= 0) {
        var ax = (agent % SIDE) * cell + cell / 2;
        var ay = Math.floor(agent / SIDE) * cell + cell / 2;
        ctx.beginPath();
        ctx.arc(ax, ay, cell * 0.17, 0, 6.284);
        ctx.fillStyle = cssVar(mount, "--link", "#2156a5");
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = base;
        ctx.stroke();
      }
    }

    function arrow(x, y, r, a, colour) {
      var dx = DC[a], dy = DR[a];
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;
      ctx.lineWidth = Math.max(1.5, r * 0.22);
      ctx.beginPath();
      ctx.moveTo(x - dx * r, y - dy * r);
      ctx.lineTo(x + dx * r * 0.35, y + dy * r * 0.35);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x + dx * r, y + dy * r);
      ctx.lineTo(x + (-dx * 0.4 + dy * 0.45) * r, y + (-dy * 0.4 - dx * 0.45) * r);
      ctx.lineTo(x + (-dx * 0.4 - dy * 0.45) * r, y + (-dy * 0.4 + dx * 0.45) * r);
      ctx.closePath();
      ctx.fill();
    }

    // The rolling success rate is capped by exploration (with eps = 0.05 and a
    // six-step path, even a perfect policy only wins ~73% of training
    // episodes), so "solved" is judged on a greedy rollout instead.
    function greedySolves() {
      var s = startState();
      for (var t = 0; t < 40; t++) {
        var r = step(s, greedy(s));
        if (r.s === s) return false;          // walked into a wall forever
        s = r.s;
        if (r.done) return map[s] === "G";
      }
      return false;
    }

    function readout() {
      setEp(episodes);
      var w = wins.length
        ? wins.reduce(function (a, b) { return a + b; }, 0) / wins.length
        : 0;
      setWin(w);
      setEps(eps);
      return w;
    }

    function refresh() { draw(); return readout(); }

    // --- training loop -----------------------------------------------------
    function frame() {
      var n = Math.min(EPISODES_PER_FRAME, remaining);
      for (var i = 0; i < n; i++) episode();
      remaining -= n;
      var w = refresh();
      if (remaining > 0) {
        note.textContent = "Training… " + episodes + " episodes. Value is "
          + "spreading backwards from the goal, one step per update.";
        raf = requestAnimationFrame(frame);
      } else {
        training = false;
        trainBtn.disabled = false;
        note.textContent = greedySolves()
          ? "Solved: the greedy policy reaches the goal. Shading is roughly "
            + "0.95^(steps to goal), so it fades with distance. The success "
            + "rate stays below 1 because training still explores."
          : w > 0.2
            ? "Getting there — " + (100 * w).toFixed(0) + "% of recent episodes "
              + "reach the goal, but the greedy policy still fails. Train more."
            : "Not solved yet. Either it needs more episodes, or the holes you "
              + "placed leave no path from start to goal.";
      }
    }

    function train(n) {
      if (training) return;
      stopRun();
      training = true;
      trainBtn.disabled = true;
      remaining = n;
      raf = requestAnimationFrame(frame);
    }

    function stopTrain() {
      training = false;
      remaining = 0;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      trainBtn.disabled = false;
    }

    function stopRun() {
      if (runTimer) clearTimeout(runTimer);
      runTimer = null;
      agent = -1;
    }

    // --- controls ----------------------------------------------------------
    var trainBtn = button("Train " + TRAIN_BATCH + " episodes",
      function () { train(TRAIN_BATCH); }, true);

    button("Run the policy", function () {
      stopTrain();
      stopRun();
      agent = startState();
      var steps = 0;
      refresh();
      (function walk() {
        runTimer = setTimeout(function () {
          var r = step(agent, greedy(agent));
          agent = r.s;
          steps++;
          refresh();
          if (r.done || steps >= 30) {
            note.textContent = map[agent] === "G"
              ? "Reached the goal in " + steps + " steps, following the greedy "
                + "policy with no exploration."
              : map[agent] === "H"
                ? "Fell in a hole. The policy is not good enough yet — train more."
                : "Wandered without terminating; the policy is still mostly noise.";
            setTimeout(function () { agent = -1; refresh(); }, 900);
            return;
          }
          walk();
        }, 260);
      })();
      note.textContent = "Running the greedy policy from the start.";
    });

    button("Reset learning", function () {
      stopTrain();
      stopRun();
      Q = freshQ();
      episodes = 0;
      eps = EPS0;
      wins = [];
      refresh();
      note.textContent = "Q reset to the optimistic " + OPTIMISTIC + ". Only "
        + "the goal carries reward, so value has to propagate backwards from it.";
    });

    button("Reset map", function () {
      stopTrain();
      stopRun();
      map = DEFAULT_MAP.slice();
      Q = freshQ();
      episodes = 0;
      eps = EPS0;
      wins = [];
      refresh();
      note.textContent = "Back to the standard FrozenLake 4x4 map.";
    });

    // --- editing the map ---------------------------------------------------
    canvas.addEventListener("pointerdown", function (ev) {
      var r = canvas.getBoundingClientRect();
      var c = Math.floor((ev.clientX - r.left) / (r.width / SIDE));
      var row = Math.floor((ev.clientY - r.top) / (r.height / SIDE));
      if (row < 0 || row >= SIDE || c < 0 || c >= SIDE) return;
      var s = row * SIDE + c;
      if (map[s] === "S" || map[s] === "G") {
        note.textContent = "The start and the goal stay put — toggle the other "
          + "cells to move the holes around.";
        return;
      }
      ev.preventDefault();
      stopTrain();
      stopRun();
      map[s] = map[s] === "H" ? "F" : "H";
      Q = freshQ();
      episodes = 0;
      eps = EPS0;
      wins = [];
      refresh();
      note.textContent = "Map changed, so the learning restarts. Train again to "
        + "see the new value landscape.";
    });

    note.textContent = "Nothing learned yet. Only the goal pays a reward, and "
      + "every other action starts at a small optimistic " + OPTIMISTIC
      + " so the agent bothers to try them. Press Train and watch value flow "
      + "backwards from the goal. Click any cell to add or remove a hole.";
    refresh();

    window.addEventListener("resize", refresh);
    new MutationObserver(refresh).observe(document.documentElement, {
      attributes: true, attributeFilter: ["data-theme"],
    });
  }

  function init() {
    var mounts = document.querySelectorAll(
      '.widget[data-widget="gridworld"] .widget-mount');
    for (var i = 0; i < mounts.length; i++) build(mounts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

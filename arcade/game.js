/* Gradient Descent — a one-button downhill surfer.
 * Ride the loss landscape: hold to dive (accelerate down slopes), release to
 * launch off crests and fly. Outrun the rising deadline; go as far as you can.
 * Pure canvas, no dependencies. */
(function () {
  "use strict";
  var canvas = document.getElementById("gd-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  // ---- tunables ----------------------------------------------------------
  var GRAV = 1300;        // gravity (px/s^2)
  var DIVE = 1500;        // extra downward pull while holding
  var MAXV = 1300;        // speed cap
  var START_X = 120;      // start on a downhill stretch
  var START_VX = 360;
  var FRICTION = 0.04;    // gentle speed bleed per second
  var CAM_X = 0.34;       // ball horizontal position on screen (fraction)
  var CAM_Y = 0.44;       // ball vertical position on screen (fraction)

  // terrain h(x): a gentle net downward trend (you keep descending) + rolling
  // hills. Slopes stay below ~45° so motion is mostly forward, not plummeting.
  function h(x) {
    return -0.16 * x
      + 52 * Math.sin(0.0060 * x)
      + 15 * Math.sin(0.0180 * x + 0.8)
      + 5 * Math.sin(0.0460 * x + 2.1);
  }
  function dh(x) {
    return -0.16
      + 52 * 0.0060 * Math.cos(0.0060 * x)
      + 15 * 0.0180 * Math.cos(0.0180 * x + 0.8)
      + 5 * 0.0460 * Math.cos(0.0460 * x + 2.1);
  }

  // ---- state -------------------------------------------------------------
  var state = "start";
  var ball, holding = false, camY = 0, t = 0, deadline = 0, best = 0;
  var coins = {}, trail = [];
  var flash = 0, flashText = "";

  try { best = parseInt(localStorage.getItem("gd-best") || "0", 10) || 0; } catch (e) {}

  var prevSlope = 0;

  function reset() {
    ball = { x: START_X, y: h(START_X), v: START_VX, vx: 0, vy: 0, air: false };
    holding = false; camY = ball.y; t = 0; trail = []; coins = {};
    deadline = START_X - W; flash = 0; prevSlope = dh(START_X);
  }
  reset();

  function score() { return Math.max(0, Math.floor((ball.x - START_X) / 8)); }
  function curSpeed() { return ball.air ? Math.hypot(ball.vx, ball.vy) : ball.v; }

  // ---- simulation --------------------------------------------------------
  // Grounded: the ball rolls along the surface with a scalar speed `v`; gravity
  // along the slope accelerates it downhill (and decelerates uphill). Holding
  // adds pull → dive faster on downhills. Release at a crest to launch into the
  // air; dive to come back down for a "perfect" landing. This stays stable and
  // always makes forward progress, unlike snapping velocity each frame.
  function step(dt) {
    var g = holding ? GRAV + DIVE : GRAV;

    if (!ball.air) {
      var s = dh(ball.x);
      var L = Math.hypot(1, s);
      ball.v += (-g * s / L) * dt;            // along-slope acceleration
      ball.v *= (1 - FRICTION * dt);
      if (ball.v < 45) ball.v = 45;
      if (ball.v > MAXV) ball.v = MAXV;
      ball.x += ball.v * dt / L;              // advance along the surface
      ball.y = h(ball.x);
      var s2 = dh(ball.x);
      // launch off a crest if you let go with enough speed
      if (prevSlope > 0.03 && s2 < 0.03 && ball.v > 360 && !holding) {
        var L2 = Math.hypot(1, s2);
        ball.air = true;
        ball.vx = ball.v / L2;
        ball.vy = ball.v * s2 / L2 + 40;       // a little pop
      }
      prevSlope = s2;
    } else {
      ball.vy -= g * dt;
      ball.x += ball.vx * dt;
      ball.y += ball.vy * dt;
      if (ball.vx > MAXV) ball.vx = MAXV;
      var gy = h(ball.x);
      if (ball.y <= gy) {                       // land
        ball.y = gy;
        var sl = dh(ball.x), Ll = Math.hypot(1, sl);
        var vt = (ball.vx + ball.vy * sl) / Ll; // tangential speed
        if (holding && sl < -0.1 && ball.vy < 0 && vt > 320) {
          vt *= 1.15; flash = 1; flashText = "PERFECT";
        }
        if (vt < 45) vt = 45;
        if (vt > MAXV) vt = MAXV;
        ball.v = vt; ball.air = false; prevSlope = sl;
      }
    }

    // camera eases vertically so descents and jumps both read well
    camY += (ball.y - camY) * Math.min(1, dt * 6);

    trail.push({ x: ball.x, y: ball.y });
    if (trail.length > 60) trail.shift();

    // coins on the surface
    var spacing = 230;
    var k0 = Math.floor((ball.x - 200) / spacing);
    for (var k = k0; k < k0 + 12; k++) {
      if (k < 1 || coins[k]) continue;
      var cx = k * spacing, cy = h(cx) + 46;
      if (Math.hypot(ball.x - cx, ball.y - cy) < 32) {
        coins[k] = 1; flash = 1; flashText = "+50";
        if (ball.air) ball.vx = Math.min(MAXV, ball.vx + 60);
        else ball.v = Math.min(MAXV, ball.v + 70);
      }
    }

    // deadline creeps right and ramps up — eventually outruns even a perfect run
    deadline += (190 + t * 9) * dt;
    if (ball.x < deadline) gameOver();

    if (flash > 0) flash -= dt * 1.4;
    t += dt;
  }

  function gameOver() {
    state = "over";
    var sc = score();
    if (sc > best) { best = sc; try { localStorage.setItem("gd-best", String(best)); } catch (e) {} }
  }

  // ---- rendering ---------------------------------------------------------
  function wx(x) { return (x - ball.x) + W * CAM_X; }
  function wy(y) { return H * CAM_Y - (y - camY); }

  function draw() {
    // sky
    var sky = ctx.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0, "#1b1f2b");
    sky.addColorStop(1, "#2a2030");
    ctx.fillStyle = sky; ctx.fillRect(0, 0, W, H);

    // loss landscape (filled ground)
    var camX = ball.x - W * CAM_X;
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (var sx = 0; sx <= W; sx += 4) {
      ctx.lineTo(sx, wy(h(camX + sx)));
    }
    ctx.lineTo(W, H); ctx.closePath();
    var grd = ctx.createLinearGradient(0, 0, 0, H);
    grd.addColorStop(0, "#b9402c");
    grd.addColorStop(0.5, "#7a2f57");
    grd.addColorStop(1, "#2156a5");
    ctx.fillStyle = grd; ctx.fill();
    // surface highlight
    ctx.beginPath();
    for (var sx2 = 0; sx2 <= W; sx2 += 4) {
      var yy = wy(h(camX + sx2));
      if (sx2 === 0) ctx.moveTo(sx2, yy); else ctx.lineTo(sx2, yy);
    }
    ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 2; ctx.stroke();

    // coins
    var spacing = 230, k0 = Math.floor(camX / spacing);
    for (var k = k0; k < k0 + 16; k++) {
      if (k < 1 || coins[k]) continue;
      var cx = k * spacing;
      ctx.beginPath();
      ctx.arc(wx(cx), wy(h(cx) + 46), 7, 0, 6.2832);
      ctx.fillStyle = "#f2d11a"; ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.lineWidth = 1.5; ctx.stroke();
    }

    // deadline wall
    var dwx = wx(deadline);
    if (dwx > -40) {
      var dg = ctx.createLinearGradient(0, 0, dwx, 0);
      dg.addColorStop(0, "rgba(186,57,37,.55)");
      dg.addColorStop(1, "rgba(186,57,37,0)");
      ctx.fillStyle = dg; ctx.fillRect(0, 0, Math.max(0, dwx), H);
      ctx.strokeStyle = "rgba(186,57,37,.9)"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(dwx, 0); ctx.lineTo(dwx, H); ctx.stroke();
    }

    // trail
    ctx.beginPath();
    for (var i = 0; i < trail.length; i++) {
      var p = trail[i];
      if (i === 0) ctx.moveTo(wx(p.x), wy(p.y)); else ctx.lineTo(wx(p.x), wy(p.y));
    }
    ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 3; ctx.stroke();

    // ball
    var bx = wx(ball.x), by = wy(ball.y);
    if (holding && state === "play") {
      // "diving" aura so it's obvious the hold is registering
      ctx.beginPath(); ctx.arc(bx, by, 18, 0, 6.2832);
      ctx.fillStyle = "rgba(242,209,26,.30)"; ctx.fill();
      ctx.fillStyle = "#f2d11a"; ctx.font = "bold 12px 'Open Sans',sans-serif";
      ctx.textAlign = "center"; ctx.fillText("▼ dive", bx, by + 30);
    }
    ctx.beginPath(); ctx.arc(bx, by, 10, 0, 6.2832);
    ctx.fillStyle = "#fdfdfc"; ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = holding && state === "play" ? "#f2d11a" : "#ba3925";
    ctx.stroke();

    // HUD
    ctx.fillStyle = "#fdfdfc";
    ctx.font = "bold 20px 'Open Sans',sans-serif"; ctx.textAlign = "left";
    ctx.fillText("loss ↓ " + score(), 16, 30);
    ctx.font = "14px 'Open Sans',sans-serif"; ctx.fillStyle = "rgba(253,253,252,.7)";
    ctx.fillText("best " + best, 16, 50);
    ctx.textAlign = "right";
    ctx.fillText("speed " + Math.round(curSpeed()), W - 16, 30);

    if (flash > 0) {
      ctx.globalAlpha = Math.min(1, flash);
      ctx.fillStyle = "#f2d11a"; ctx.font = "bold 22px 'Open Sans',sans-serif";
      ctx.textAlign = "center"; ctx.fillText(flashText, bx, by - 24);
      ctx.globalAlpha = 1;
    }

    if (state !== "play") overlay();
  }

  function overlay() {
    ctx.fillStyle = "rgba(15,15,22,.6)"; ctx.fillRect(0, 0, W, H);
    ctx.textAlign = "center"; ctx.fillStyle = "#fdfdfc";
    if (state === "start") {
      ctx.font = "bold 34px 'Open Sans',sans-serif";
      ctx.fillText("Gradient Descent", W / 2, H / 2 - 36);
      ctx.font = "16px 'Open Sans',sans-serif"; ctx.fillStyle = "rgba(253,253,252,.85)";
      ctx.fillText("Ride the loss landscape. Hold to dive, release to fly.", W / 2, H / 2 + 2);
      ctx.fillText("Hold mouse / tap / Space.  Outrun the red wall.", W / 2, H / 2 + 26);
      ctx.fillStyle = "#f2d11a"; ctx.font = "bold 18px 'Open Sans',sans-serif";
      ctx.fillText("Click or press Space to start", W / 2, H / 2 + 64);
    } else {
      ctx.font = "bold 34px 'Open Sans',sans-serif";
      ctx.fillText("Run over", W / 2, H / 2 - 30);
      ctx.font = "20px 'Open Sans',sans-serif"; ctx.fillStyle = "#f2d11a";
      ctx.fillText("loss descended: " + score(), W / 2, H / 2 + 4);
      ctx.fillStyle = "rgba(253,253,252,.85)"; ctx.font = "15px 'Open Sans',sans-serif";
      ctx.fillText("best " + best + "  ·  click or Space to retry", W / 2, H / 2 + 34);
    }
  }

  // ---- loop --------------------------------------------------------------
  var last = 0;
  function frame(now) {
    var dt = last ? Math.min(0.033, (now - last) / 1000) : 0;
    last = now;
    if (state === "play") step(dt);
    draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // ---- input -------------------------------------------------------------
  function press() {
    if (state === "play") { holding = true; return; }
    reset(); state = "play"; holding = true;        // first press starts + dives
  }
  function release() { holding = false; }

  // Pointer Events unify mouse / touch / pen and avoid the double-fire you get
  // from separate mouse+touch handlers (which left the hold state stuck). Pointer
  // capture means we still get pointerup even if the finger/cursor leaves the
  // canvas; blur/cancel release so holding can never get stuck on.
  canvas.addEventListener("pointerdown", function (e) {
    e.preventDefault();
    try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
    press();
  });
  canvas.addEventListener("pointerup", function (e) { e.preventDefault(); release(); });
  canvas.addEventListener("pointercancel", release);
  window.addEventListener("blur", release);

  window.addEventListener("keydown", function (e) {
    if (e.code === "Space" || e.key === " ") { e.preventDefault(); if (!e.repeat) press(); }
  });
  window.addEventListener("keyup", function (e) {
    if (e.code === "Space" || e.key === " ") release();
  });
})();

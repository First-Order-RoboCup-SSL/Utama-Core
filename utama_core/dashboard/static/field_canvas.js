// Shared field renderer used by every dashboard view that shows robots on a
// pitch (live view, replay view). One implementation, one set of visual
// conventions, so a replay looks exactly like the live match it was
// recorded from.
//
// Usage:
//   const view = new FieldCanvas(canvasEl, geometry, { myTeamIsRight, myTeamIsYellow });
//   view.draw({ robots: {friendly:[...], enemy:[...]}, ball: {...}, designated: [x,y], tactics: {robotId: "label"} });

const FIELD_COLORS = {
  pitch: "#242832",
  line: "#4b5162",
  yellow: "#e0a527",
  blue: "#3d7fd1",
  ball: "#c98a3f",
  marker: "#e8eaf0",
  laneOpen: "#5fb87a",
  laneBlocked: "#c0524a",
  committed: "#e0a14a",
};

// Robot body radius, world meters -> px is ROBOT_RADIUS * scale so the drawn
// robot is proportional to the actual field (division-I robots are 0.09m
// radius; see utama_core/config/physical_constants.py:ROBOT_RADIUS). Do not
// hardcode a pixel radius here — it drifts out of proportion as soon as the
// canvas or field geometry changes size.
const ROBOT_RADIUS_M = 0.09;

class FieldCanvas {
  constructor(canvas, geometry, { myTeamIsRight = false, myTeamIsYellow = true } = {}) {
    this.canvas = canvas;
    this.geometry = geometry;
    this.myTeamIsRight = myTeamIsRight;
    this.myTeamIsYellow = myTeamIsYellow;
    this._lastState = {};
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(canvas.parentElement);
    this.resize();
  }

  setTeamLayout(myTeamIsRight, myTeamIsYellow) {
    this.myTeamIsRight = myTeamIsRight;
    this.myTeamIsYellow = myTeamIsYellow;
  }

  _view() {
    const g = this.geometry;
    const goalDepth = Math.max(0, Number(g.goal_depth) || 0);
    return {
      minX: -g.half_length - goalDepth,
      maxX: g.half_length + goalDepth,
      minY: -g.half_width,
      maxY: g.half_width,
      width: 2 * (g.half_length + goalDepth),
      height: 2 * g.half_width,
      goalDepth,
    };
  }

  _transform() {
    const canvas = this.canvas;
    const view = this._view();
    const M = 12;
    const usableW = Math.max(1, canvas.width - 2 * M);
    const usableH = Math.max(1, canvas.height - 2 * M);
    const scale = Math.min(usableW / view.width, usableH / view.height);
    const originX = (canvas.width - view.width * scale) / 2;
    const originY = (canvas.height - view.height * scale) / 2;
    const myTeamIsRight = this.myTeamIsRight;

    return {
      scale,
      toX(fx) {
        const displayX = myTeamIsRight ? fx : -fx;
        return originX + (displayX - view.minX) * scale;
      },
      toY(fy) {
        return originY + (view.maxY - fy) * scale;
      },
      toField(px, py) {
        let fx = view.minX + (px - originX) / scale;
        if (!myTeamIsRight) fx = -fx;
        const fy = view.maxY - (py - originY) / scale;
        return { x: fx, y: fy };
      },
    };
  }

  resize() {
    const canvas = this.canvas;
    const wrap = canvas.parentElement;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    const view = this._view();
    const fieldAspect = view.width / view.height;
    const wrapAspect = cw / ch;
    let pw, ph;
    if (wrapAspect > fieldAspect) {
      ph = ch;
      pw = Math.round(ch * fieldAspect);
    } else {
      pw = cw;
      ph = Math.round(cw / fieldAspect);
    }
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    canvas.style.left = Math.round((cw - pw) / 2) + "px";
    canvas.style.top = Math.round((ch - ph) / 2) + "px";
    canvas.style.width = pw + "px";
    canvas.style.height = ph + "px";
    this.draw(this._lastState);
  }

  draw(state) {
    this._lastState = state || {};
    const canvas = this.canvas;
    const g = this.geometry;
    const ctx = canvas.getContext("2d");
    const tx = this._transform();
    const scale = tx.scale;
    const toX = tx.toX;
    const toY = tx.toY;
    const CW = canvas.width;
    const CH = canvas.height;

    function rectBounds(x1, y1, x2, y2) {
      const px1 = toX(x1), px2 = toX(x2);
      const py1 = toY(y1), py2 = toY(y2);
      return { x: Math.min(px1, px2), y: Math.min(py1, py2), w: Math.abs(px2 - px1), h: Math.abs(py2 - py1) };
    }
    function strokeWorldRect(x1, y1, x2, y2) {
      const r = rectBounds(x1, y1, x2, y2);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
    }

    ctx.fillStyle = FIELD_COLORS.pitch;
    ctx.fillRect(0, 0, CW, CH);

    ctx.strokeStyle = FIELD_COLORS.line;
    ctx.lineWidth = 1.25;
    strokeWorldRect(-g.half_length, -g.half_width, g.half_length, g.half_width);

    ctx.beginPath();
    ctx.moveTo(toX(0), toY(-g.half_width));
    ctx.lineTo(toX(0), toY(g.half_width));
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(toX(0), toY(0), g.center_circle_radius * scale, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = FIELD_COLORS.line;
    ctx.beginPath();
    ctx.arc(toX(0), toY(0), 1.5, 0, 2 * Math.PI);
    ctx.fill();

    const dl = 2 * g.half_defense_depth, dw = g.half_defense_width;
    strokeWorldRect(-g.half_length, -dw, -g.half_length + dl, dw);
    strokeWorldRect(g.half_length - dl, -dw, g.half_length, dw);

    const goalDepth = Math.max(0, Number(g.goal_depth) || 0);
    ctx.strokeStyle = FIELD_COLORS.line;
    ctx.lineWidth = 1.5;
    strokeWorldRect(-g.half_length - goalDepth, -g.half_goal_width, -g.half_length, g.half_goal_width);
    strokeWorldRect(g.half_length, -g.half_goal_width, g.half_length + goalDepth, g.half_goal_width);

    if (state.designated) {
      const dx = toX(state.designated[0]), dy = toY(state.designated[1]);
      ctx.strokeStyle = FIELD_COLORS.marker;
      ctx.lineWidth = 1.5;
      const s = 5;
      ctx.beginPath();
      ctx.moveTo(dx - s, dy - s); ctx.lineTo(dx + s, dy + s);
      ctx.moveTo(dx + s, dy - s); ctx.lineTo(dx - s, dy + s);
      ctx.stroke();
    }

    const robots = state.robots;
    const tactics = state.tactics || {};
    if (robots) {
      const r = Math.max(2, ROBOT_RADIUS_M * scale);
      const friendlyFill = this.myTeamIsYellow ? FIELD_COLORS.yellow : FIELD_COLORS.blue;
      const enemyFill = this.myTeamIsYellow ? FIELD_COLORS.blue : FIELD_COLORS.yellow;
      const drawTeam = (list, fill, label) => {
        for (const bot of list || []) {
          const cx = toX(bot.x), cy = toY(bot.y);
          ctx.fillStyle = fill;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, 2 * Math.PI);
          ctx.fill();
          // Heading dash extends past the body edge so it reads as a
          // direction pointer rather than disappearing into the fill.
          ctx.strokeStyle = FIELD_COLORS.pitch;
          ctx.lineWidth = Math.max(1, r * 0.28);
          const headingLen = r * 1.6;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + headingLen * Math.cos(bot.orientation), cy - headingLen * Math.sin(bot.orientation));
          ctx.stroke();
          ctx.fillStyle = fill;
          ctx.font = "7px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText(bot.id, cx, cy - r - 1);
          if (label && tactics[String(bot.id)]) {
            ctx.fillStyle = FIELD_COLORS.marker;
            ctx.font = "6px ui-monospace, monospace";
            ctx.fillText(tactics[String(bot.id)], cx, cy + r + 7);
          }
        }
      };
      drawTeam(robots.enemy, enemyFill, false);
      drawTeam(robots.friendly, friendlyFill, true);
    }

    if (state.ball) {
      const bx = toX(state.ball.x), by = toY(state.ball.y);
      ctx.fillStyle = FIELD_COLORS.ball;
      ctx.beginPath();
      ctx.arc(bx, by, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    this._drawOverlays(ctx, state, robots, toX, toY);
  }

  // Tactic-specific geometric intentions (which enemy a marker covers, a
  // pass's intended receive point, ...), forward-filled by the caller from
  // sparse TraceEvents keyed by whatever string the tactic chose when
  // calling `ctx.match_log.trace_if_changed(...)`. This renderer only knows
  // the few keys below by name — a tactic's trace value is inert here until
  // a case is added for its key, same opt-in shape as everything else in
  // this file.
  _drawOverlays(ctx, state, robots, toX, toY) {
    const overlays = state.overlays;
    if (!overlays || !robots) return;

    const byId = {};
    for (const bot of robots.friendly || []) byId["f" + bot.id] = bot;
    for (const bot of robots.enemy || []) byId["e" + bot.id] = bot;

    // A robot whose tactic slot currently returns is_committed() — the
    // scheduler will not reassign it until it releases or a barrier reset
    // clears everything. Drawn first/underneath so a robot that is also
    // highlights()-called-out this tick still shows both distinctly (a
    // bigger dashed lock ring vs. the smaller solid highlight ring).
    const committedIds = overlays["committed_robot_ids"];
    if (committedIds) {
      ctx.strokeStyle = FIELD_COLORS.committed;
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 1.5;
      for (const robotId of committedIds) {
        const bot = byId["f" + robotId];
        if (!bot) continue;
        const cx = toX(bot.x), cy = toY(bot.y);
        ctx.beginPath();
        ctx.arc(cx, cy, 11, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    const marks = overlays["shadow_and_mark.marks"];
    if (marks) {
      ctx.strokeStyle = FIELD_COLORS.marker;
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      for (const markerId in marks) {
        const opponentId = marks[markerId];
        const marker = byId["f" + markerId];
        const opponent = byId["e" + opponentId];
        if (!marker || !opponent) continue;
        this._drawArrow(ctx, toX(marker.x), toY(marker.y), toX(opponent.x), toY(opponent.y));
      }
      ctx.setLineDash([]);
    }

    const shotLane = overlays["give_and_go.shot_lane"];
    if (shotLane) {
      const fx = toX(shotLane.from.x), fy = toY(shotLane.from.y);
      if (shotLane.to) {
        ctx.strokeStyle = FIELD_COLORS.laneOpen;
        ctx.setLineDash([]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(toX(shotLane.to.x), toY(shotLane.to.y));
        ctx.stroke();
      } else {
        // No open lane: a short blocked-red stub toward goal instead of a
        // full line to nowhere, so "no shot" still reads at a glance.
        ctx.strokeStyle = FIELD_COLORS.laneBlocked;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1.5;
        const goalDir = shotLane.from.x < 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(fx + goalDir * 14, fy);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    const shadowPost = overlays["defense.shadow_post"];
    if (shadowPost) {
      ctx.strokeStyle = FIELD_COLORS.laneOpen;
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(toX(shadowPost.x), toY(shadowPost.y), 4, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const screenLine = overlays["block_shape.screen_line"];
    if (screenLine) {
      ctx.strokeStyle = FIELD_COLORS.laneOpen;
      ctx.setLineDash([]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(screenLine.x), toY(screenLine.y1));
      ctx.lineTo(toX(screenLine.x), toY(screenLine.y2));
      ctx.stroke();
    }

    const runnerTarget = overlays["switch_of_play.runner_target"];
    if (runnerTarget) {
      const runner = byId["f" + runnerTarget.runner_id];
      if (runner) {
        ctx.strokeStyle = FIELD_COLORS.ball;
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 1.25;
        this._drawArrow(ctx, toX(runner.x), toY(runner.y), toX(runnerTarget.x), toY(runnerTarget.y));
        ctx.setLineDash([]);
      }
      const rtx = toX(runnerTarget.x), rty = toY(runnerTarget.y);
      ctx.strokeStyle = FIELD_COLORS.ball;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(rtx, rty, 4, 0, 2 * Math.PI);
      ctx.stroke();
    }

    const press = overlays["press_and_contain.press"];
    if (press) {
      const presser = byId["f" + press.presser_id];
      const pressed = byId["e" + press.pressed_enemy_id];
      if (presser && pressed) {
        ctx.strokeStyle = FIELD_COLORS.laneBlocked;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        this._drawArrow(ctx, toX(presser.x), toY(presser.y), toX(pressed.x), toY(pressed.y));
        ctx.setLineDash([]);
      }
    }

    const passTarget = overlays["give_and_go.pass_target"];
    if (passTarget) {
      const receiver = byId["f" + passTarget.receiver_id];
      if (receiver) {
        ctx.strokeStyle = FIELD_COLORS.ball;
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 1.25;
        this._drawArrow(ctx, toX(receiver.x), toY(receiver.y), toX(passTarget.x), toY(passTarget.y));
        ctx.setLineDash([]);
      }
      // The intended receive point itself, whether or not the receiver has
      // reached it yet — the whole point of tracing this is to see when the
      // two diverge (the exact silent-aim bug class STRATEGY_DEVELOPMENT.md
      // warns intercept_point() can produce).
      const tx = toX(passTarget.x), ty = toY(passTarget.y);
      ctx.strokeStyle = FIELD_COLORS.ball;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(tx, ty, 4, 0, 2 * Math.PI);
      ctx.stroke();
    }

    const clearTarget = overlays["clear_ball.clear_target"];
    if (clearTarget) {
      ctx.strokeStyle = FIELD_COLORS.laneOpen;
      ctx.setLineDash([]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(toX(clearTarget.from.x), toY(clearTarget.from.y));
      ctx.lineTo(toX(clearTarget.to.x), toY(clearTarget.to.y));
      ctx.stroke();
    }

    const dribbleTarget = overlays["dribble.segment_target"];
    if (dribbleTarget) {
      ctx.strokeStyle = FIELD_COLORS.marker;
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(toX(dribbleTarget.x), toY(dribbleTarget.y), 4, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const lureTarget = overlays["decoy_and_overload.lure_target"];
    if (lureTarget) {
      ctx.strokeStyle = FIELD_COLORS.laneBlocked;
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(toX(lureTarget.x), toY(lureTarget.y), 4, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const overloadTarget = overlays["decoy_and_overload.overload_target"];
    if (overloadTarget) {
      ctx.strokeStyle = FIELD_COLORS.laneOpen;
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(toX(overloadTarget.x), toY(overloadTarget.y), 4, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Any tactic's optional, purely cosmetic per-robot highlight (see
    // `Tactic.highlights()`), logged one key per active slot as
    // "highlights.<tactic_id>" -> {robot_id: label}. Not tied to any
    // specific tactic or to is_committed() — a tactic decides what's worth
    // calling out and why; this renderer just draws whatever comes through.
    for (const key in overlays) {
      if (!key.startsWith("highlights.")) continue;
      const labels = overlays[key];
      for (const robotId in labels) {
        const bot = byId["f" + robotId];
        if (!bot) continue;
        const cx = toX(bot.x), cy = toY(bot.y);
        ctx.strokeStyle = FIELD_COLORS.marker;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = FIELD_COLORS.marker;
        ctx.font = "6px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(labels[robotId], cx, cy - 8 - 3);
      }
    }
  }

  _drawArrow(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const headLen = 5;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.save();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
    ctx.restore();
  }

  canvasToField(clientX, clientY) {
    const canvas = this.canvas;
    const rect = canvas.getBoundingClientRect();
    const cw = canvas.width, ch = canvas.height;
    const tx = this._transform();
    const px = (clientX - rect.left) * (cw / rect.width);
    const py = (clientY - rect.top) * (ch / rect.height);
    return tx.toField(px, py);
  }
}

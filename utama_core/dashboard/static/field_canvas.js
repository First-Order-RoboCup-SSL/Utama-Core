// Shared field renderer used by every dashboard view that shows robots on a
// pitch (live view, replay view). One implementation, one set of visual
// conventions, so a replay looks exactly like the live match it was
// recorded from.
//
// Usage:
//   const view = new FieldCanvas(canvasEl, geometry, { myTeamIsRight, myTeamIsYellow });
//   view.draw({ robots: {friendly:[...], enemy:[...]}, ball: {...}, designated: [x,y], tactics: {robotId: "label"},
//               tactic_tags: {robotId: "attack" | "defense" | "mixed"} });

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

// Ring colour per `TacticTag` value (utama_core/engine/tactic.py) around a
// friendly robot, so the team's attack/defense split reads at a glance while
// the body keeps its jersey colour. Purely a display of the tag the robot's
// current tactic declares; nothing in the scheduler reads it.
const TACTIC_TAG_COLORS = {
  attack: "#d9694f",
  defense: "#4fb3a9",
  mixed: "#a68fd6",
};

// Robot body radius, world meters -> px is ROBOT_RADIUS * scale so the drawn
// robot is proportional to the actual field (division-I robots are 0.09m
// radius; see utama_core/config/physical_constants.py:ROBOT_RADIUS). Do not
// hardcode a pixel radius here — it drifts out of proportion as soon as the
// canvas or field geometry changes size.
const ROBOT_RADIUS_M = 0.09;

// The run-off around the field lines (Division B's boundary width), drawn so a ball or robot
// just over a line stays in view.
const RUNOFF_M = 0.3;

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
    const halfX = g.half_length + Math.max(goalDepth, RUNOFF_M);
    const halfY = g.half_width + RUNOFF_M;
    return { minX: -halfX, maxX: halfX, minY: -halfY, maxY: halfY, width: 2 * halfX, height: 2 * halfY, goalDepth };
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

  // Canvas pixels -> field metres ({x, y}), for clicks on the field.
  toField(px, py) {
    return this._transform().toField(px, py);
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

    if (state.field_check) this._drawAreaInPlay(ctx, state.field_check, toX, toY, rectBounds, CW, CH);

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

    // Labels grow with the robots but stay readable on a phone-sized field.
    const textPx = Math.max(10, Math.round(ROBOT_RADIUS_M * scale * 1.4)) + "px ui-monospace, monospace";
    const robots = state.robots;
    const tactics = state.tactics || {};
    const tacticTags = state.tactic_tags || {};
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
          const tagColor = label ? TACTIC_TAG_COLORS[tacticTags[String(bot.id)]] : undefined;
          if (tagColor) {
            ctx.strokeStyle = tagColor;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, r + 2, 0, 2 * Math.PI);
            ctx.stroke();
          }
          // Heading dash extends past the body edge so it reads as a
          // direction pointer rather than disappearing into the fill.
          ctx.strokeStyle = FIELD_COLORS.pitch;
          ctx.lineWidth = Math.max(1, r * 0.28);
          // Through toX/toY, so a mirrored view (our goal on the right) mirrors it too.
          const headingM = (r * 1.6) / scale;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(
            toX(bot.x + headingM * Math.cos(bot.orientation)),
            toY(bot.y + headingM * Math.sin(bot.orientation))
          );
          ctx.stroke();
          this._label(ctx, bot.id, cx, cy - r - 2, "bottom", "bold " + textPx, fill);
          if (label && tactics[String(bot.id)]) {
            this._label(ctx, tactics[String(bot.id)], cx, cy + r + 2, "top", textPx, FIELD_COLORS.marker);
          }
        }
      };
      drawTeam(robots.enemy, enemyFill, false);
      drawTeam(robots.friendly, friendlyFill, true);
      this._drawTagLegend(ctx, tacticTags);

      // A real robot whose ball sensor is on ("friendly:3", from the radio's feedback): a ring
      // in the ball's colour, so a sensor that never fires (or never stops) shows on the field.
      const sensorOn = new Set(state.ball_sensor || []);
      ctx.strokeStyle = FIELD_COLORS.ball;
      ctx.lineWidth = 3;
      for (const team of ["friendly", "enemy"]) {
        for (const bot of robots[team] || []) {
          if (!sensorOn.has(team + ":" + bot.id)) continue;
          ctx.beginPath();
          ctx.arc(toX(bot.x), toY(bot.y), r + 6, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }

    if (state.field_check && robots) {
      // A robot past the area in play by more than the margin has forced HALT on real robots.
      const [x1, y1, x2, y2] = state.field_check.bounds;
      const m = state.field_check.margin;
      ctx.strokeStyle = "#e06a60";
      ctx.lineWidth = 2;
      for (const bot of [...(robots.friendly || []), ...(robots.enemy || [])]) {
        if (bot.x < x1 - m || bot.x > x2 + m || bot.y < y1 - m || bot.y > y2 + m) {
          ctx.beginPath();
          ctx.arc(toX(bot.x), toY(bot.y), Math.max(2, ROBOT_RADIUS_M * scale) + 4, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    }

    if (state.ball) {
      // Out of view, the ball is drawn hollow at the nearest edge, so it never just disappears.
      const R = 3.5;
      const rawX = toX(state.ball.x), rawY = toY(state.ball.y);
      const bx = Math.min(CW - R - 1, Math.max(R + 1, rawX));
      const by = Math.min(CH - R - 1, Math.max(R + 1, rawY));
      ctx.beginPath();
      ctx.arc(bx, by, R, 0, 2 * Math.PI);
      if (bx === rawX && by === rawY) {
        ctx.fillStyle = FIELD_COLORS.ball;
        ctx.fill();
      } else {
        ctx.strokeStyle = FIELD_COLORS.ball;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    this._drawOverlays(ctx, state, robots, toX, toY, textPx, Math.max(2, ROBOT_RADIUS_M * scale));
  }

  // Text with a pitch-coloured halo, so it reads over field lines and overlays.
  _label(ctx, text, x, y, baseline, font, colour) {
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = baseline;
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = FIELD_COLORS.pitch;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);
    ctx.textBaseline = "alphabetic";
  }

  // The area in play (StrategyRunner's field_bounds): everything outside it is dimmed, so a
  // half-field test shows at a glance which half the runner thinks the camera covers.
  // The +x arrow ties the picture to ssl-vision's coordinates, since this view turns the
  // field so our team's goal is on the right.
  _drawAreaInPlay(ctx, fieldCheck, toX, toY, rectBounds, CW, CH) {
    const [x1, y1, x2, y2] = fieldCheck.bounds;
    const r = rectBounds(x1, y1, x2, y2);
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, CW, CH);
    ctx.rect(r.x + r.w, r.y, -r.w, r.h); // opposite winding cuts the area in play out
    ctx.fill();
    ctx.strokeStyle = FIELD_COLORS.laneOpen;
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]);
    const ax = toX(0), ay = CH - 6, dir = toX(1) > ax ? 1 : -1;
    ctx.strokeStyle = ctx.fillStyle = FIELD_COLORS.marker;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + dir * 24, ay);
    ctx.lineTo(ax + dir * 20, ay - 3);
    ctx.moveTo(ax + dir * 24, ay);
    ctx.lineTo(ax + dir * 20, ay + 3);
    ctx.stroke();
    ctx.font = "7px ui-monospace, monospace";
    ctx.textAlign = dir > 0 ? "left" : "right";
    ctx.fillText("+x (vision)", ax + dir * 27, ay + 2);
    ctx.restore();
  }

  // Small key for the tactic-tag rings, only when some robot has a tag.
  _drawTagLegend(ctx, tacticTags) {
    const present = new Set(Object.values(tacticTags));
    const entries = Object.keys(TACTIC_TAG_COLORS).filter((tag) => present.has(tag));
    if (!entries.length) return;
    ctx.font = "7px ui-monospace, monospace";
    ctx.textAlign = "left";
    let x = 8;
    const y = 10;
    for (const tag of entries) {
      ctx.strokeStyle = TACTIC_TAG_COLORS[tag];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x + 3, y - 2, 3, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = FIELD_COLORS.marker;
      ctx.fillText(tag, x + 9, y);
      x += 9 + ctx.measureText(tag).width + 10;
    }
  }

  // Tactic-specific geometric intentions (which enemy a marker covers, a
  // pass's intended receive point, ...), forward-filled by the caller from
  // sparse TraceEvents keyed by whatever string the tactic chose when
  // calling `ctx.match_log.trace_if_changed(...)`. This renderer only knows
  // the few keys below by name — a tactic's trace value is inert here until
  // a case is added for its key, same opt-in shape as everything else in
  // this file.
  _drawOverlays(ctx, state, robots, toX, toY, textPx, r) {
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
    // The label goes under the robot (its number is above), below its tactic label if it has one.
    const tactics = state.tactics || {};
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
        ctx.arc(cx, cy, r + 1, 0, 2 * Math.PI);
        ctx.stroke();
        const below = cy + r + 2 + (tactics[robotId] ? parseInt(textPx, 10) + 1 : 0);
        this._label(ctx, labels[robotId], cx, below, "top", textPx, FIELD_COLORS.marker);
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

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
  friendly: "#e8eaf0",
  enemy: "#6b7280",
  ball: "#c98a3f",
  marker: "#e8eaf0",
};

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
      const r = 5;
      const drawTeam = (list, fill, label) => {
        for (const bot of list || []) {
          const cx = toX(bot.x), cy = toY(bot.y);
          ctx.fillStyle = fill;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = FIELD_COLORS.pitch;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + r * Math.cos(bot.orientation), cy - r * Math.sin(bot.orientation));
          ctx.stroke();
          ctx.fillStyle = fill;
          ctx.font = "7px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText(bot.id, cx, cy - r - 1);
          if (label && tactics[String(bot.id)]) {
            ctx.fillStyle = FIELD_COLORS.friendly;
            ctx.font = "6px ui-monospace, monospace";
            ctx.fillText(tactics[String(bot.id)], cx, cy + r + 7);
          }
        }
      };
      drawTeam(robots.enemy, FIELD_COLORS.enemy, false);
      drawTeam(robots.friendly, FIELD_COLORS.friendly, true);
    }

    if (state.ball) {
      const bx = toX(state.ball.x), by = toY(state.ball.y);
      ctx.fillStyle = FIELD_COLORS.ball;
      ctx.beginPath();
      ctx.arc(bx, by, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
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

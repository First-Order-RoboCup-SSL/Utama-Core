// Match timeline: a window of the match as swimlanes, the dashboard's version of
// tools/match_timeline.py's figure, in the Replay tab around the viewed frame and in the
// Live tab up to now. Top to bottom, on one time axis:
//   - the referee command band (live play, STOP, kickoff, free kick, penalty, placement, HALT);
//   - possession: whose ball it is, in the team's colour. In a replay, one span per situation
//     (a restart going live or a side winning the ball, as goal_ledger counts them); live,
//     the side with a robot on the ball;
//   - one lane per friendly robot: a span per tactic it was in (the Tactic class from the
//     intention log), a dark tick where it switched, and a light strip where it had the ball;
//   - each friendly robot's distance to the ball, 0-3 m;
//   - events: goals, fouls, stalls, ball losses (the replay's markers) and shots (dots in the
//     shooting team's colour).
// Hovering names what is under the pointer. A cursor marks the viewed frame; in a replay,
// clicking or dragging seeks. A replay is computed once from data replay.js already holds
// (`load`); live, each streamed state extends the lanes (`append`) and the last LIVE_KEEP_S
// are kept. Drawing only clips and paints.

(function () {
  const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"];
  const OTHER = "#5f636c"; // tactics beyond the seven slots
  const TEAM = { yellow: "#e0a527", blue: "#3d7fd1" }; // field_canvas.js's robot colours
  const REF_FILL = {
    "live play": "#2b3a30",
    STOP: "#5b5f68",
    HALT: "#b8615a",
    kickoff: "#3d5a80",
    "free kick": "#7a5236",
    penalty: "#6d4560",
    placement: "#6e6436",
    other: "#33373f",
  };
  const MARKER_FILL = { goal: "#e0c14f", foul: "#d08a4a", stall: "#b8615a", loss: "#8b909b" };
  const KEEPER = "goalkeeper"; // the label the Live view's tactic status gives the keeper
  const LIVE_KEEP_S = 300;
  const MAX_DIST_M = 3;

  const GUTTER = 44; // left labels
  const REF_H = 16;
  const BALL_H = 12;
  const LANE_H = 18;
  const DIST_H = 46;
  const EVENT_H = 12;
  const AXIS_H = 14;
  const GAP = 4;

  // The legend group of a referee command name, as match_timeline.ref_group.
  function refGroup(command) {
    if (!command) return "other";
    if (command === "NORMAL_START" || command === "FORCE_START") return "live play";
    if (command === "HALT") return "HALT";
    if (command === "STOP" || command.startsWith("GOAL_") || command.startsWith("TIMEOUT_")) return "STOP";
    if (command.startsWith("PREPARE_KICKOFF")) return "kickoff";
    if (command.startsWith("DIRECT_FREE") || command.startsWith("INDIRECT_FREE")) return "free kick";
    if (command.startsWith("PREPARE_PENALTY")) return "penalty";
    if (command.startsWith("BALL_PLACEMENT")) return "placement";
    return "other";
  }

  // The commands the referee's match clock runs under (referee_command.py's CLOCK_RUNS): it stops in
  // STOP, HALT, kickoff and penalty preparation and ball placement, so a 10 min match takes longer.
  const CLOCK_RUNS = new Set([
    "NORMAL_START",
    "FORCE_START",
    "DIRECT_FREE_YELLOW",
    "DIRECT_FREE_BLUE",
    "INDIRECT_FREE_YELLOW",
    "INDIRECT_FREE_BLUE",
  ]);

  // The match clock from the referee's change rows [{sim_time, command, stage}], as the referee
  // keeps it (state_machine.py's `_stage_played`): a function of time giving the stage and its
  // playing time so far, or null before the first row.
  function matchClock(referee) {
    const played = [];
    let total = 0;
    for (let i = 0; i < referee.length; i++) {
      const prev = referee[i - 1];
      if (prev && CLOCK_RUNS.has(prev.command)) total += referee[i].sim_time - prev.sim_time;
      if (prev && referee[i].stage !== prev.stage) total = 0;
      played.push(total);
    }
    return (t) => {
      let lo = 0;
      let hi = referee.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (referee[mid].sim_time <= t) lo = mid + 1;
        else hi = mid;
      }
      if (lo === 0) return null;
      const e = referee[lo - 1];
      return { stage: e.stage, played: played[lo - 1] + (CLOCK_RUNS.has(e.command) ? t - e.sim_time : 0) };
    };
  }

  // Spans [{t0, t1, label, text}] tiling [start, end] from change points [[t, label, text]],
  // merging repeats (a merged span keeps its first text).
  function spansFrom(changes, start, end) {
    const out = [];
    for (let i = 0; i < changes.length; i++) {
      const t0 = Math.max(changes[i][0], start);
      const t1 = Math.min(i + 1 < changes.length ? changes[i + 1][0] : end, end);
      const label = changes[i][1];
      if (t1 <= t0) continue;
      const last = out[out.length - 1];
      if (last && last.label === label) last.t1 = t1;
      else out.push({ t0, t1, label, text: changes[i][2] });
    }
    return out;
  }

  // For each robot, the tactic it was in over time, as match_timeline.tactic_changes: a robot's
  // tactic is the slot whose latest row lists it, named by the row's note when that is a class
  // name; a "referee reset" row clears every slot.
  function tacticChanges(events, robotIds) {
    const members = new Map(); // slot -> Set(robot ids); insertion order = last update last
    const names = {};
    const changes = {};
    for (const id of robotIds) changes[id] = [];
    for (const e of events) {
      if (e.tactic_id === "referee reset") members.clear();
      else {
        if (e.note && /^[A-Za-z_][A-Za-z0-9_]*$/.test(e.note)) names[e.tactic_id] = e.note;
        if (!(e.tactic_id in names)) names[e.tactic_id] = e.tactic_id;
        members.delete(e.tactic_id);
        members.set(e.tactic_id, new Set(e.robot_ids));
      }
      const slots = Array.from(members.keys()).reverse();
      for (const id of robotIds) {
        const slot = slots.find((s) => members.get(s).has(id));
        const label = slot === undefined ? null : names[slot];
        const points = changes[id];
        const last = points[points.length - 1];
        if (!last && label === null) continue;
        if (last && last[1] === label) continue;
        if (last && last[0] === e.sim_time) last[1] = label;
        else points.push([e.sim_time, label]);
      }
    }
    return changes;
  }

  // Runs of `flag(i)` over the frames, as spans.
  function runs(frames, flag) {
    const out = [];
    let start = null;
    for (let i = 0; i < frames.length; i++) {
      const on = flag(i);
      if (on && start === null) start = frames[i].ts;
      else if (!on && start !== null) {
        out.push({ t0: start, t1: frames[i].ts, label: true });
        start = null;
      }
    }
    if (start !== null) out.push({ t0: start, t1: frames[frames.length - 1].ts, label: true });
    return out;
  }

  function firstAtOrAfter(frames, ts) {
    let lo = 0;
    let hi = frames.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].ts < ts) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function at(spans, t) {
    return spans.find((s) => s.t0 <= t && t <= s.t1 && s.label !== null && s.label !== false);
  }

  function short(tactic) {
    return tactic.replace(/Tactic$/, "");
  }

  function mmss(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  // A live tactic status entry ["<slot>[ (committed)]", "<tag>"] as a lane label.
  // A robot the referee drives (a restart, HALT) is one "referee" span: its job is in the Tactics
  // panel, and which restart it was is the referee lane above, so jobs don't each take a colour.
  function liveLabel(status) {
    if (!status) return null;
    return status[1] === "referee" ? "referee" : status[0].replace(/ \(committed\)$/, "");
  }

  class Timeline {
    constructor(canvas, legend, onSeek) {
      this.canvas = canvas;
      this.legend = legend;
      this.onSeek = onSeek;
      this.windowS = 60;
      this.data = null;
      this.ts = 0;
      this._view = null; // [t0, t1] last drawn
      this._rows = []; // [{y0, y1, kind, lane}] last drawn, for hovering
      this.tip = document.createElement("div");
      this.tip.className = "timeline-tip";
      canvas.parentElement.appendChild(this.tip);

      let dragging = false;
      const timeAt = (e) => {
        if (!this._view) return null;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left - GUTTER;
        const w = rect.width - GUTTER;
        if (x < 0 || w <= 0) return null;
        return this._view[0] + (x / w) * (this._view[1] - this._view[0]);
      };
      const seek = (e) => {
        const t = timeAt(e);
        if (t !== null && this.onSeek) this.onSeek(t);
      };
      canvas.addEventListener("mousedown", (e) => {
        dragging = Boolean(this.onSeek);
        seek(e);
      });
      window.addEventListener("mousemove", (e) => dragging && seek(e));
      window.addEventListener("mouseup", () => (dragging = false));
      canvas.addEventListener("mousemove", (e) => {
        const t = timeAt(e);
        const rect = canvas.getBoundingClientRect();
        const text = t === null ? null : this._describe(t, e.clientY - rect.top);
        this.tip.style.display = text ? "" : "none";
        if (!text) return;
        this.tip.textContent = text;
        const x = e.clientX - rect.left;
        this.tip.style.left = Math.min(x + 12, rect.width - this.tip.offsetWidth - 4) + "px";
        this.tip.style.top = e.clientY - rect.top + 14 + "px";
      });
      canvas.addEventListener("mouseleave", () => (this.tip.style.display = "none"));
      new ResizeObserver(() => this.draw(this.ts)).observe(canvas.parentElement);
    }

    setWindow(seconds) {
      this.windowS = seconds;
      this.draw(this.ts);
    }

    // A replay. `play` rows carry `title` and `ended` text (replay.js words them, for its list).
    load(frames, { tactics = [], referee = [], trace = [], markers = [], shots = [], play = [], myTeamIsYellow = true } = {}) {
      if (!frames.length) {
        this.data = null;
        this.legend.innerHTML = "";
        this.draw(0);
        return;
      }
      const start = frames[0].ts;
      const end = frames[frames.length - 1].ts;
      const ids = new Set();
      for (const f of frames) for (const r of f.robots.friendly) ids.add(r.id);
      const robotIds = Array.from(ids).sort((a, b) => a - b);

      const keepers = new Set();
      for (const e of trace) {
        if (e.key === "highlights.goalkeeper" && e.value) for (const k in e.value) keepers.add(Number(k));
      }
      const changes = tacticChanges(tactics, robotIds);
      const lanes = robotIds.map((id) => {
        const points = [[start, keepers.has(id) ? KEEPER : null]].concat(changes[id]);
        const spans = spansFrom(points, start, end).map((s) =>
          s.label === null && keepers.has(id) ? { ...s, label: KEEPER } : s
        );
        const has = (i) => frames[i].robots.friendly.some((r) => r.id === id && r.has_ball);
        return { id, tactics: spans, ball: runs(frames, has) };
      });

      const total = {};
      for (const lane of lanes) {
        for (const s of lane.tactics) {
          if (s.label && s.label !== KEEPER) total[s.label] = (total[s.label] || 0) + s.t1 - s.t0;
        }
      }
      const ranked = Object.keys(total).sort((a, b) => total[b] - total[a] || a.localeCompare(b));
      const colours = {};
      ranked.forEach((name, i) => (colours[name] = i < SLOTS.length ? SLOTS[i] : OTHER));

      const refPoints = referee.map((e) => [
        e.sim_time,
        refGroup(e.command),
        e.command.replace(/_/g, " ").toLowerCase() + (e.note ? ": " + e.note : ""),
      ]);
      const possession = play.map((r) => ({
        t0: r.t,
        t1: r.t_end,
        label: r.side || "none",
        text: r.title + (r.ended ? " → " + r.ended : ""),
        kind: r.kind,
      }));

      // distance to the ball per robot per frame, NaN when either is missing
      const dist = {};
      for (const id of robotIds) dist[id] = new Float32Array(frames.length);
      frames.forEach((f, i) => {
        for (const id of robotIds) {
          const r = f.robots.friendly.find((q) => q.id === id);
          dist[id][i] = r && f.ball ? Math.hypot(r.x - f.ball.x, r.y - f.ball.y) : NaN;
        }
      });

      this.data = {
        frames,
        origin: start,
        start,
        end,
        lanes,
        colours,
        referee: refPoints.length ? spansFrom(refPoints, start, end) : [],
        possession,
        dist,
        markers,
        shots,
        robotColour: {},
        team: this._teams(myTeamIsYellow),
      };
      robotIds.forEach((id, i) => (this.data.robotColour[id] = SLOTS[i % SLOTS.length]));
      this.legend.innerHTML =
        this._refKey(this.data.referee) +
        ranked.map((name) => `<span><i style="background:${colours[name]}"></i>${short(name)}</span>`).join("");
      this.draw(start);
    }

    // The referee band's colours, for the groups this match has: a short STOP or placement is
    // too narrow to carry its name.
    _refKey(spans) {
      const groups = new Set(spans.map((s) => s.label));
      const present = Object.keys(REF_FILL).filter((g) => groups.has(g));
      if (!present.length) return "";
      const keys = present.map((g) => `<span><i style="background:${REF_FILL[g]}"></i>${g}</span>`).join("");
      return `<span class="muted">ref</span>${keys}<span class="muted">tactics</span>`;
    }

    _teams(myTeamIsYellow) {
      const friendly = myTeamIsYellow ? "yellow" : "blue";
      const enemy = myTeamIsYellow ? "blue" : "yellow";
      return { friendly: TEAM[friendly], enemy: TEAM[enemy], names: { friendly, enemy } };
    }

    // Live: one state from the Live view's stream (robots, ball, command, scores,
    // tactic_status), at `ts` seconds. Spans are extended in place; the caller draws.
    append(ts, state) {
      let d = this.data;
      // a new match (the clock jumped) starts over; a state no newer than the last is skipped
      const fresh = !d || !d.live || ts < d.end - 5 || ts > d.end + 60;
      if (!fresh && ts <= d.end) return;
      if (fresh) {
        d = this.data = { live: true, frames: [], origin: ts, start: ts, end: ts, lanes: [], colours: {}, referee: [] };
        Object.assign(d, { possession: [], dist: {}, markers: [], shots: [], robotColour: {}, score: null });
        d.team = this._teams(state.my_team_is_yellow !== false);
        this.legend.innerHTML = "";
      }
      const extend = (spans, label, text) => {
        const last = spans[spans.length - 1];
        if (last) last.t1 = ts;
        if (!last || last.label !== label) spans.push({ t0: ts, t1: ts, label, text });
      };
      const friendly = (state.robots && state.robots.friendly) || [];
      const enemy = (state.robots && state.robots.enemy) || [];
      for (const r of friendly) {
        if (d.dist[r.id]) continue;
        d.dist[r.id] = d.frames.map(() => NaN);
        d.robotColour[r.id] = SLOTS[Object.keys(d.robotColour).length % SLOTS.length];
        d.lanes.push({ id: r.id, tactics: [], ball: [] });
        d.lanes.sort((a, b) => a.id - b.id);
      }
      d.frames.push({ ts });
      d.end = ts;
      const group = refGroup(state.command);
      extend(d.referee, group, (state.command || "").replace(/_/g, " ").toLowerCase());

      // possession as goal_ledger reads it: the side with a robot on the ball while the other
      // has none keeps it until the other side touches it; nobody holds it outside live play
      const last = d.possession[d.possession.length - 1];
      const ours = friendly.some((r) => r.has_ball);
      const theirs = enemy.some((r) => r.has_ball);
      const holder = group !== "live play" ? null : ours !== theirs ? (ours ? "friendly" : "enemy") : last ? last.label : null;
      extend(d.possession, holder, holder ? d.team.names[holder] + "'s ball" : null);

      const status = state.tactic_status || {};
      for (const lane of d.lanes) {
        const r = friendly.find((q) => q.id === lane.id);
        const label = r ? liveLabel(status[String(lane.id)]) : null;
        if (label && label !== KEEPER && !(label in d.colours)) {
          const n = Object.keys(d.colours).length;
          d.colours[label] = n < SLOTS.length ? SLOTS[n] : OTHER;
          this.legend.innerHTML += `<span><i style="background:${d.colours[label]}"></i>${label}</span>`;
        }
        extend(lane.tactics, label);
        extend(lane.ball, Boolean(r && r.has_ball));
        d.dist[lane.id].push(r && state.ball ? Math.hypot(r.x - state.ball.x, r.y - state.ball.y) : NaN);
      }
      const score = [state.yellow_score, state.blue_score];
      if (d.score && (score[0] > d.score[0] || score[1] > d.score[1])) {
        const scorer = score[0] > d.score[0] ? "yellow" : "blue";
        d.markers.push({ sim_time: ts, kind: "goal", label: `Goal ${scorer}, ${score[0]}-${score[1]}` });
      }
      d.score = score;

      if (ts - d.start > LIVE_KEEP_S + 30) {
        const cut = ts - LIVE_KEEP_S;
        const k = firstAtOrAfter(d.frames, cut);
        d.frames.splice(0, k);
        for (const id in d.dist) d.dist[id].splice(0, k);
        const keep = (spans) => spans.filter((s) => s.t1 >= cut);
        d.referee = keep(d.referee);
        d.possession = keep(d.possession);
        d.markers = d.markers.filter((m) => m.sim_time >= cut);
        for (const lane of d.lanes) {
          lane.tactics = keep(lane.tactics);
          lane.ball = keep(lane.ball);
        }
        d.start = d.frames[0].ts;
      }
      this.ts = ts;
    }

    // What is under the pointer at time `t`, height `y`, as one line.
    _describe(t, y) {
      const d = this.data;
      const row = this._rows.find((r) => y >= r.y0 && y < r.y1);
      if (!d || !row) return null;
      const when = mmss(t - d.origin) + "  ";
      if (row.kind === "ref") {
        const s = at(d.referee, t);
        return s ? when + (s.text || s.label) : null;
      }
      if (row.kind === "ball") {
        const s = at(d.possession, t);
        return s && s.text ? when + s.text : null;
      }
      if (row.kind === "lane") {
        const s = at(row.lane.tactics, t);
        const ball = at(row.lane.ball, t) ? ", on the ball" : "";
        return when + "#" + row.lane.id + " " + (s ? s.label : "no tactic") + ball;
      }
      if (row.kind === "dist") {
        const i = Math.min(firstAtOrAfter(d.frames, t), d.frames.length - 1);
        const parts = d.lanes
          .filter((l) => !Number.isNaN(d.dist[l.id][i]))
          .map((l) => "#" + l.id + " " + d.dist[l.id][i].toFixed(1) + " m");
        return parts.length ? when + "to the ball: " + parts.join(", ") : null;
      }
      if (row.kind === "events") {
        const tol = (4 * (this._view[1] - this._view[0])) / Math.max(1, this.canvas.clientWidth - GUTTER); // 4 px
        const hits = d.markers.filter((m) => Math.abs(m.sim_time - t) <= tol).map((m) => m.label);
        for (const s of d.shots) {
          if (Math.abs(s.t - t) <= tol) hits.push("shot by " + d.team.names[s.side]);
        }
        return hits.length ? when + hits.join("; ") : null;
      }
      return null;
    }

    _height() {
      const lanes = this.data ? this.data.lanes.length : 0;
      return REF_H + BALL_H + GAP + lanes * LANE_H + GAP + DIST_H + GAP + EVENT_H + AXIS_H;
    }

    draw(ts) {
      this.ts = ts;
      const canvas = this.canvas;
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.parentElement.clientWidth;
      const cssH = this._height();
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
        canvas.style.width = cssW + "px";
        canvas.style.height = cssH + "px";
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      const d = this.data;
      this._rows = [];
      if (!d) return;

      const span = Math.min(this.windowS || d.end - d.start, d.end - d.start) || 1;
      let t0 = ts - span / 3;
      t0 = Math.max(d.start, Math.min(t0, d.end - span));
      const t1 = t0 + span;
      this._view = [t0, t1];
      const w = cssW - GUTTER;
      const x = (t) => GUTTER + ((t - t0) / span) * w;
      ctx.font = "10px ui-monospace, monospace";
      ctx.textBaseline = "middle";

      const band = (spans, y, h, fill, label) => {
        for (const s of spans) {
          if (s.t1 < t0 || s.t0 > t1 || s.label === null || s.label === false) continue;
          const a = Math.max(x(s.t0), GUTTER);
          const b = Math.min(x(s.t1), cssW);
          ctx.fillStyle = fill(s);
          ctx.fillRect(a, y, Math.max(1, b - a - 1), h);
          const text = label && label(s);
          if (text && ctx.measureText(text).width + 6 < b - a) {
            ctx.fillStyle = "#dfe2e8";
            ctx.fillText(text, a + 3, y + h / 2);
          }
        }
      };
      const row = (kind, text, y, h, lane) => {
        this._rows.push({ kind, y0: y, y1: y + h, lane });
        ctx.fillStyle = "#8b909b";
        ctx.fillText(text, 2, y + h / 2);
      };

      let y = 0;
      row("ref", "ref", y, REF_H);
      band(d.referee, y, REF_H, (s) => REF_FILL[s.label] || REF_FILL.other, (s) => s.label);
      y += REF_H;

      row("ball", "ball", y, BALL_H);
      band(
        d.possession,
        y + 2,
        BALL_H - 3,
        (s) => d.team[s.label] || OTHER,
        (s) => (s.kind && s.kind !== "ball won" ? "from " + s.kind : null)
      );
      y += BALL_H + GAP;

      for (const lane of d.lanes) {
        row("lane", "#" + lane.id, y, LANE_H, lane);
        band(
          lane.tactics,
          y,
          LANE_H - 2,
          (s) => (s.label === KEEPER ? OTHER : d.colours[s.label] || OTHER),
          (s) => short(s.label)
        );
        // a tick where the robot switched from one tactic to another
        ctx.fillStyle = "#101114";
        let prev = null;
        for (const s of lane.tactics) {
          if (!s.label) continue;
          if (prev && prev !== s.label && s.t0 >= t0 && s.t0 <= t1) ctx.fillRect(x(s.t0) - 1, y, 2, LANE_H - 2);
          prev = s.label;
        }
        band(lane.ball, y + LANE_H - 5, 3, () => "#f2f2f2");
        y += LANE_H;
      }
      y += GAP;

      row("dist", "ball m", y, DIST_H);
      ctx.strokeStyle = "#33373f";
      ctx.beginPath();
      for (const m of [1, 2]) {
        const gy = y + DIST_H * (1 - m / MAX_DIST_M);
        ctx.moveTo(GUTTER, gy);
        ctx.lineTo(cssW, gy);
      }
      ctx.stroke();
      const i0 = Math.max(0, firstAtOrAfter(d.frames, t0) - 1);
      const i1 = Math.min(d.frames.length, firstAtOrAfter(d.frames, t1) + 1);
      const stride = Math.max(1, Math.floor((i1 - i0) / w));
      for (const lane of d.lanes) {
        const series = d.dist[lane.id];
        ctx.strokeStyle = d.robotColour[lane.id];
        ctx.beginPath();
        let pen = false;
        for (let i = i0; i < i1; i += stride) {
          const v = series[i];
          if (Number.isNaN(v)) {
            pen = false;
            continue;
          }
          const px = x(d.frames[i].ts);
          const py = y + DIST_H * (1 - Math.min(v, MAX_DIST_M) / MAX_DIST_M);
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        }
        ctx.stroke();
      }
      y += DIST_H + GAP;

      row("events", "events", y, EVENT_H);
      for (const m of d.markers) {
        if (m.sim_time < t0 || m.sim_time > t1) continue;
        ctx.fillStyle = MARKER_FILL[m.kind] || "#8b909b";
        ctx.fillRect(x(m.sim_time) - 1.5, y, 3, EVENT_H);
      }
      ctx.strokeStyle = "#101114";
      for (const s of d.shots) {
        if (s.t < t0 || s.t > t1) continue;
        ctx.fillStyle = d.team[s.side] || OTHER;
        ctx.beginPath();
        ctx.arc(x(s.t), y + EVENT_H / 2, 3.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      y += EVENT_H;

      // axis: a tick every 5 s or 10 s or 30 s, labelled as match time (m:ss)
      const step = span > 180 ? 30 : span > 60 ? 10 : 5;
      ctx.fillStyle = "#8b909b";
      ctx.textAlign = "center";
      for (let s = Math.ceil((t0 - d.origin) / step) * step; d.origin + s <= t1; s += step) {
        const px = x(d.origin + s);
        ctx.fillRect(px, y, 1, 3);
        if (px > GUTTER + 12 && px < cssW - 12) ctx.fillText(mmss(s), px, y + 9);
      }
      ctx.textAlign = "start";

      const cx = x(ts);
      ctx.fillStyle = "#dfe2e8";
      ctx.fillRect(cx - 0.5, 0, 1.5, y);
    }
  }

  if (typeof window !== "undefined") Object.assign(window, { Timeline, matchClock });
  if (typeof module !== "undefined") module.exports = { refGroup, spansFrom, tacticChanges, matchClock };
})();

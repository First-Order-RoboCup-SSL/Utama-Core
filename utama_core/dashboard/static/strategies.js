// Strategies tab: every strategy in the catalog (what it is, its tactics, its verdict, where it
// stands and shoots on the pitch, how it plays, its newest matches) and every evaluation run.
// Data: GET /strategies, /strategies/runs and /strategies/maps (dashboard/views/strategies.py).
// There are no standings: the verdict (goals, paired against the champion) is the only ranking;
// signals sit beside the other strategies' average to explain how a strategy plays (docs/signals.md).

(function () {
  let data = null; // /strategies
  let runs = null; // /strategies/runs, loaded when the Runs page is first opened
  let page = "strategies"; // "strategies" | "runs"
  let selected = null; // strategy name
  let showArchive = false;
  const maps = {}; // round run_id -> /strategies/maps answer
  const expandedRun = {}; // run_id -> bool; the newest starts open
  const expandedRound = {}; // "run_id:k" -> bool

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (x) => Math.round((x || 0) * 100) + "%";
  const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
  const signed = (x, d = 2) => (x == null ? "—" : (x > 0 ? "+" : "") + Number(x).toFixed(d));

  const pill = (text, cls, title) => `<span class="pill ${cls || ""}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
  function replayButton(path, atTime, label) {
    if (!path) return "";
    const at = atTime != null ? ` data-at="${atTime}"` : "";
    return `<button class="icon-btn view-replay-btn" data-path="${esc(path)}"${at} title="watch the replay">${esc(label || "▶")}</button>`;
  }

  // --- the strategy list --------------------------------------------------------------------------

  function renderList() {
    const groups = [
      ["Champion", (s) => s.role === "champion"],
      ["Core", (s) => s.role === "core"],
      ["Reserve", (s) => s.role === "reserve"],
    ];
    return groups
      .map(([title, keep]) => {
        const rows = data.strategies.filter(keep).sort((a, b) => a.name.localeCompare(b.name));
        if (!rows.length) return "";
        return (
          `<div class="strategy-group">${title}</div>` +
          rows
            .map((s) => {
              const v = s.verdicts[0];
              return (
                `<div class="strategy-item${s.name === selected ? " active" : ""}" data-name="${esc(s.name)}" title="${esc(s.summary)}">` +
                `<span>${esc(s.name)}</span>${v ? pill(v.verdict, v.verdict) : ""}</div>`
              );
            })
            .join("")
        );
      })
      .join("");
  }

  // --- one strategy -------------------------------------------------------------------------------

  function renderTactics(s) {
    return s.tactics
      .map((t) => `<span class="tactic-chip" title="${esc(t.description)}\n${esc(t.file)}">${esc(t.name.replace(/Tactic$/, ""))}</span>`)
      .join(" ");
  }

  function renderVerdict(s) {
    const v = s.verdicts[0];
    if (!v) return pill("not evaluated", "", `pixi run evaluate ${s.name}`);
    const detail = `decided in the ${v.decided_in} after ${v.rounds} round(s), n ${v.n}, p ${num(v.p, 3)}, ${v.date || v.run_id}`;
    return `${pill(v.verdict, v.verdict, detail)} <span class="muted">${signed(v.mean)} ± ${num(v.se)} goals/match vs ${esc(v.reference)}</span>`;
  }

  // An SVG pitch in the maps' frame (metres, attacking right), with `inner` drawn under the lines.
  function pitchSvg(p, inner) {
    const L = p.half_length, W = p.half_width, gd = p.goal_depth, pad = 0.12;
    const vb = `${-L - gd - pad} ${-W - pad} ${2 * (L + gd + pad)} ${2 * (W + pad)}`;
    const line = 'fill="none" stroke="#a5b3a7" stroke-width="1" vector-effect="non-scaling-stroke"';
    const lines =
      `<rect x="${-L}" y="${-W}" width="${2 * L}" height="${2 * W}" ${line}/>` +
      `<line x1="0" y1="${-W}" x2="0" y2="${W}" ${line}/>` +
      `<circle cx="0" cy="0" r="${p.centre_radius}" ${line}/>` +
      `<rect x="${-L}" y="${-p.box_half_width}" width="${p.box_depth}" height="${2 * p.box_half_width}" ${line}/>` +
      `<rect x="${L - p.box_depth}" y="${-p.box_half_width}" width="${p.box_depth}" height="${2 * p.box_half_width}" ${line}/>` +
      `<rect x="${-L - gd}" y="${-p.goal_half_width}" width="${gd}" height="${2 * p.goal_half_width}" fill="#a5b3a7"/>` +
      `<rect x="${L}" y="${-p.goal_half_width}" width="${gd}" height="${2 * p.goal_half_width}" fill="#a5b3a7"/>`;
    return (
      `<svg viewBox="${vb}" xmlns="http://www.w3.org/2000/svg"><g transform="scale(1,-1)">` +
      `<rect x="${-L - gd - pad}" y="${-W - pad}" width="${2 * (L + gd + pad)}" height="${2 * (W + pad)}" fill="#f4f7f3"/>` +
      `${inner}${lines}</g></svg>`
    );
  }

  function heatmap(m, grid, vmax) {
    const L = m.pitch.half_length, W = m.pitch.half_width, b = m.bin_m;
    let out = "";
    grid.share.forEach((row, i) =>
      row.forEach((v, j) => {
        if (v > 0) {
          const a = Math.min(1, v / vmax).toFixed(2);
          out += `<rect x="${-L + j * b}" y="${-W + i * b}" width="${b + 0.01}" height="${b + 0.01}" fill="#2563eb" fill-opacity="${a}"/>`;
        }
      })
    );
    for (const [x, y] of grid.means) out += `<circle cx="${x}" cy="${y}" r="0.11" fill="#1c2230" stroke="#fff" stroke-width="0.03"/>`;
    return out;
  }

  function shotMarks(shots) {
    const mark = ([x, y, scored], colour) =>
      `<circle cx="${x}" cy="${y}" r="0.09" fill="${scored ? colour : "#fff"}" fill-opacity="${scored ? 1 : 0.6}" stroke="${colour}" stroke-width="0.03"/>`;
    return (shots.faced || []).map((s) => mark(s, "#dc2626")).join("") + (shots.own || []).map((s) => mark(s, "#2563eb")).join("");
  }

  function renderMaps(s) {
    if (!s.signals) return "";
    const m = maps[s.signals.run_id];
    let body;
    if (!m || m.status === "computing") body = '<div class="muted">Drawing the maps from the replays… (about a minute, once per run)</div>';
    else if (m.status !== "ready") body = '<div class="muted">No maps: this round has no replays.</div>';
    else if (!m.strategies[s.name]) body = '<div class="muted">No replays of this strategy in this round.</div>';
    else {
      const d = m.strategies[s.name];
      const fig = (title, svg) => `<figure class="map"><figcaption>${title}</figcaption>${svg}</figure>`;
      body =
        `<div class="maps">` +
        fig("With the ball", pitchSvg(m.pitch, heatmap(m, d.attack, m.vmax.attack))) +
        fig("Without the ball", pitchSvg(m.pitch, heatmap(m, d.defence, m.vmax.defence))) +
        fig(
          `Shots <span style="color:#2563eb">●</span> ours <span style="color:#dc2626">●</span> against, hollow missed`,
          pitchSvg(m.pitch, shotMarks(d.shots))
        ) +
        `</div>`;
    }
    return `<section><h2>On the pitch <span class="muted" style="font-weight:400;">attacking →, dots are each robot's average spot</span></h2>${body}</section>`;
  }

  function renderSignals(s) {
    const sig = s.signals;
    if (!sig) return "";
    const fmt = (r, x) => (x == null ? "—" : r.key.endsWith("_share") || r.key === "stalled" ? pct(x) : num(x, r.key === "pass_progress_m" ? 2 : 1));
    const rows = sig.rows
      .filter((r) => r.value != null || r.field != null)
      .map((r) => {
        const max = Math.max(Math.abs(r.value || 0), Math.abs(r.field || 0)) * 1.15 || 1;
        const at = (x) => (Math.abs(x || 0) / max) * 100;
        const a = at(r.value), b = at(r.field);
        const bar =
          `<div class="dumbbell"><div class="track"></div>` +
          (r.value != null && r.field != null ? `<div class="link" style="left:${Math.min(a, b)}%; width:${Math.abs(a - b)}%;"></div>` : "") +
          (r.field != null ? `<div class="dot others" style="left:${b}%;"></div>` : "") +
          (r.value != null ? `<div class="dot this" style="left:${a}%;"></div>` : "") +
          `</div>`;
        return `<span>${esc(r.label)}</span>${bar}<span class="num">${fmt(r, r.value)}</span><span class="num muted">${fmt(r, r.field)}</span>`;
      })
      .join("");
    const title = `${sig.matches} matches in ${sig.run_id}. More is not always better: these explain how it plays (docs/signals.md).`;
    return (
      `<section><h2 title="${esc(title)}">How it plays <span class="muted" style="font-weight:400;">per match</span></h2>` +
      `<div class="signals"><span></span><span></span><span class="head num" style="color:var(--accent)">this</span><span class="head num">others</span>${rows}</div></section>`
    );
  }

  function renderMatches(s) {
    if (!s.matches.length) return "";
    const newest = s.matches[0].run_id; // older runs played older code: not shown here
    const cards = s.matches
      .filter((m) => m.run_id === newest)
      .map((m) => {
        const [f, a] = m.score;
        const res = f > a ? "win" : f < a ? "loss" : "draw";
        return (
          `<div class="match"><span title="${esc(m.opponent)}">${esc(m.opponent)}</span>` +
          `<span class="row" style="align-items:center; gap:8px;"><span class="score ${res}">${f}–${a}</span>` +
          `${m.stalled ? pill("stalled", "dropped") : ""}${replayButton(m.replay)}</span></div>`
        );
      })
      .join("");
    return `<section><h2>Matches <span class="muted" style="font-weight:400;">${esc(s.matches[0].date || "")}</span></h2><div class="matches">${cards}</div></section>`;
  }

  function renderStrategy(s) {
    const old = s.old_names.length ? ` <span class="muted" style="font-size:.85rem;">was ${s.old_names.map(esc).join(", ")}</span>` : "";
    const role = s.role === "champion" ? pill("champion", "champion") : s.role === "core" ? pill("core", "core") : pill("reserve");
    return `
      <div class="panel strategy-page stack">
        <div>
          <div class="row" style="align-items:center; gap:10px;"><h1>${esc(s.name)}</h1>${role}${old}</div>
          <div class="lead">${esc(s.summary)}</div>
          <div class="meta-row"><span>${renderTactics(s)}</span><span>${renderVerdict(s)}</span>
            <code class="muted" title="watch it: pixi run main ${esc(s.name)} wolfpack">${esc(s.file)}</code></div>
        </div>
        ${renderMaps(s)}
        ${renderSignals(s)}
        ${renderMatches(s)}
      </div>`;
  }

  function renderStrategiesPage() {
    if (!data) return '<div class="muted" style="padding:14px;">Loading…</div>';
    if (!selected || !data.strategies.find((s) => s.name === selected)) selected = data.champion || data.strategies[0]?.name;
    const s = data.strategies.find((x) => x.name === selected);
    if (s) loadMaps(s);
    return `
      <div class="strategies-layout">
        <div class="panel strategy-list">${renderList()}</div>
        ${s ? renderStrategy(s) : ""}
      </div>`;
  }

  // --- runs ---------------------------------------------------------------------------------------

  function renderVerdictCards(run) {
    if (!run.report) return '<div class="muted">Did not finish.</div>';
    const cands = Object.entries(run.report.candidates || {});
    if (!cands.length) return '<div class="muted">No candidate: a health check only.</div>';
    return cands
      .map(([name, r]) => {
        const t = r[r.decided_in] || {};
        const h2h = r.vs_todays_champion;
        const detail = `in the ${r.decided_in} after ${r.rounds} round(s), n ${t.n ?? "—"}, p ${num(t.p, 3)}`;
        return (
          `<div class="verdict-card"><div class="row" style="justify-content:space-between;"><b>${esc(name)}</b>${pill(r.verdict, r.verdict, detail)}</div>` +
          `<div>${signed(t.mean)} ± ${num(t.se)} goals/match vs ${esc(run.reference)} at the last release` +
          (h2h ? `<span class="muted"> · vs today's ${esc(h2h.name)} ${signed(h2h.mean)} ± ${num(h2h.se)}</span>` : "") +
          `</div></div>`
        );
      })
      .join("");
  }

  // Find the replay of the match named "<a>_vs_<b>" (how stall incidents name matches).
  function replayOfMatch(round, stem) {
    const r = (round.results || []).find((x) => x.replay && x.replay.split("/").pop().startsWith(stem + "."));
    return r ? r.replay : null;
  }

  function renderHealth(round) {
    const parts = [];
    const incidents = round.stall_incidents || [];
    if (round.stall_incidents) {
      const rows = incidents
        .map(
          (inc) =>
            `<div class="ref-row"><span>${esc(inc.kind)} at ${inc.sim_time.toFixed(1)}s for ${inc.duration_s.toFixed(0)}s</span><span class="row">` +
            inc.matches.map((m) => replayButton(replayOfMatch(round, m), inc.sim_time, "▶ " + m)).join("") +
            `</span></div>`
        )
        .join("");
      parts.push(`<div><b>Stalls</b> <span class="muted">${incidents.length}</span>${rows}</div>`);
    }
    const rs = round.restarts;
    if (rs && rs.restarts) parts.push(`<div><b>Restarts</b> <span class="muted">${pct(rs.reached_normal_start / rs.restarts)} went live</span></div>`);
    if (round.fouls) {
      const byRule = Object.entries(round.fouls)
        .map(([rule, f]) => [rule, f.total])
        .sort((a, b) => b[1] - a[1])
        .map(([rule, n]) => `${n} ${rule.replace(/_/g, " ")}`)
        .join(", ");
      parts.push(`<div><b>Fouls</b> <span class="muted">${byRule || "none"}</span></div>`);
    }
    if ((round.failed || []).length) parts.push(`<div class="check-fail">${round.failed.length} match(es) crashed</div>`);
    return parts.join("");
  }

  function renderRoundMatches(round) {
    const rows = (round.results || [])
      .map(
        (r) =>
          `<tr><td>${esc(r.display_a)} vs ${esc(r.display_b)}</td><td class="num">${r.score_a}–${r.score_b}</td>` +
          `<td>${r.stalled ? pill("stalled", "dropped") : ""}</td><td>${replayButton(r.replay)}</td></tr>`
      )
      .join("");
    return `<table><tbody>${rows}</tbody></table>`;
  }

  function renderRun(run) {
    const nMatches = run.rounds.reduce((n, s) => n + (s.results || []).length, 0);
    const open = expandedRun[run.run_id];
    const header =
      `<div class="panel-title run-header" data-run-id="${esc(run.run_id)}"><span>${open ? "▾" : "▸"}</span>` +
      `<span>${esc(run.date || run.run_id)}</span><span class="muted" style="font-weight:400;">${run.rounds.length} round(s), ${nMatches} matches</span></div>`;
    if (!open) return `<div class="panel run-card">${header}</div>`;
    const rounds = run.rounds
      .map((round) => {
        const key = `${run.run_id}:${round.round ?? 1}`;
        const o = expandedRound[key];
        return (
          `<div class="round-header" data-key="${esc(key)}">${o ? "▾" : "▸"} Round ${round.round ?? 1}` +
          `${round.seed != null ? ` · seed ${round.seed}` : ""} · ${(round.results || []).length} matches</div>` +
          (o ? renderRoundMatches(round) : "")
        );
      })
      .join("");
    return `
      <div class="panel run-card">
        ${header}
        <div class="stack run-body">
          ${run.run_id.startsWith("eval_") ? `<div>${renderVerdictCards(run)}</div>` : ""}
          ${run.rounds[0] ? renderHealth(run.rounds[0]) : ""}
          <div>${rounds}</div>
        </div>
      </div>`;
  }

  function renderRunsPage() {
    if (!runs) return '<div class="muted" style="padding:14px;">Loading…</div>';
    const current = runs.filter((r) => r.run_id.startsWith("eval_"));
    const archive = runs.filter((r) => !r.run_id.startsWith("eval_"));
    const empty = current.length ? "" : '<div class="muted" style="padding:6px 2px;">No runs yet: <code>pixi run evaluate NAME</code></div>';
    const older = archive.length
      ? `<div class="archive-header">${showArchive ? "▾" : "▸"} Older round robins (${archive.length}): older code, not comparable</div>` +
        (showArchive ? archive.map(renderRun).join("") : "")
      : "";
    return `<div style="max-width:1000px;">${empty}${current.map(renderRun).join("")}${older}</div>`;
  }

  // --- shell --------------------------------------------------------------------------------------

  function render() {
    const root = document.getElementById("strategies-root");
    if (!root) return;
    const tab = (p, label) => `<button class="subtab${page === p ? " active" : ""}" data-page="${p}">${label}</button>`;
    root.innerHTML =
      `<div class="subtabs">${tab("strategies", "Strategies")}${tab("runs", "Evaluation runs")}</div>` +
      (page === "strategies" ? renderStrategiesPage() : renderRunsPage());

    for (const b of root.querySelectorAll(".subtab")) {
      b.addEventListener("click", () => {
        page = b.dataset.page;
        if (page === "runs" && !runs) loadRuns();
        render();
      });
    }
    for (const el of root.querySelectorAll(".strategy-item")) {
      el.addEventListener("click", () => {
        selected = el.dataset.name;
        render();
      });
    }
    for (const h of root.querySelectorAll(".archive-header")) {
      h.addEventListener("click", () => {
        showArchive = !showArchive;
        render();
      });
    }
    for (const h of root.querySelectorAll(".run-header")) {
      h.addEventListener("click", () => {
        expandedRun[h.dataset.runId] = !expandedRun[h.dataset.runId];
        render();
      });
    }
    for (const h of root.querySelectorAll(".round-header")) {
      h.addEventListener("click", () => {
        expandedRound[h.dataset.key] = !expandedRound[h.dataset.key];
        render();
      });
    }
    for (const btn of root.querySelectorAll(".view-replay-btn")) {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const at = btn.dataset.at != null ? Number(btn.dataset.at) : undefined;
        Dashboard.showReplay(btn.dataset.path, at);
      });
    }
  }

  // The maps of the round a strategy's signals come from: asked once, polled while computing.
  function loadMaps(s) {
    const round = s.signals && s.signals.run_id;
    const known = round && maps[round];
    if (!round || (known && (known.status !== "computing" || known.pending || known.retryAt > Date.now()))) return;
    maps[round] = { status: "computing", pending: true };
    fetch("/strategies/maps?round=" + encodeURIComponent(round))
      .then((r) => r.json())
      .then((m) => {
        const computing = m.status === "computing";
        maps[round] = computing ? { ...m, retryAt: Date.now() + 3000 } : m;
        if (computing) setTimeout(() => page === "strategies" && render(), 3100);
        render();
      })
      .catch((err) => {
        maps[round] = { status: "failed" };
        console.error("maps fetch error:", err);
      });
  }

  function load() {
    fetch("/strategies")
      .then((r) => r.json())
      .then((d) => {
        data = d;
        render();
      })
      .catch((err) => console.error("strategies fetch error:", err));
    if (page === "runs") loadRuns();
  }

  function loadRuns() {
    fetch("/strategies/runs")
      .then((r) => r.json())
      .then((d) => {
        runs = d;
        const first = runs.find((r) => r.run_id.startsWith("eval_")) || runs[0];
        if (first && !Object.keys(expandedRun).length) expandedRun[first.run_id] = true;
        render();
      })
      .catch((err) => console.error("runs fetch error:", err));
  }

  Dashboard.registerView("strategies", { mount: render, onShow: load });
})();

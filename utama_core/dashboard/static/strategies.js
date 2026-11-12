// Strategies tab: every strategy in the catalog (what it is, the tactics it uses, its verdicts, how it
// plays, its matches with their replays), and every evaluation run, verdicts first.
// Data: GET /strategies and /strategies/runs (dashboard/views/strategies.py), read from replays/.
// There are no standings: the verdict (goals, paired against the champion) is the only ranking, and
// signals are shown beside the other strategies' average to explain how a strategy plays, never to
// rank it (docs/signals.md).

(function () {
  let data = null; // /strategies
  let runs = null; // /strategies/runs, loaded when the Runs page is first opened
  let page = "strategies"; // "strategies" | "runs"
  let selected = null; // strategy name
  let showAllMatches = false;
  const expandedRun = {}; // run_id -> bool; the newest starts open
  const expandedRound = {}; // "run_id:k" -> bool

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (x) => Math.round((x || 0) * 1000) / 10 + "%";
  const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
  const signed = (x, d = 2) => (x == null ? "—" : (x > 0 ? "+" : "") + Number(x).toFixed(d));

  const VERDICT_COLOUR = { better: "check-ok", dropped: "check-fail", undecided: "check-warn" };
  function verdictBadge(v) {
    return `<span class="${VERDICT_COLOUR[v] || "muted"}" style="font-weight:600; text-transform:uppercase; font-size:.72rem;">${esc(v)}</span>`;
  }
  function roleBadge(role) {
    const label = role === "champion" ? "★ champion" : role;
    const colour = role === "champion" ? "accent" : role === "core" ? "" : "muted";
    return `<span class="${colour}" style="font-size:.68rem; text-transform:uppercase; letter-spacing:.04em;">${label}</span>`;
  }
  function replayButton(path, atTime, label) {
    if (!path) return `<span class="muted" style="font-size:.72rem;">no replay</span>`;
    const at = atTime != null ? ` data-at="${atTime}"` : "";
    return `<button class="btn-accent view-replay-btn" data-path="${esc(path)}"${at}>${esc(label || "▶ replay")}</button>`;
  }
  function testLine(t) {
    if (!t) return "";
    return `mean diff <b>${signed(t.mean)}</b> ± ${num(t.se)} goals/match (n ${t.n ?? "—"}, p ${num(t.p, 3)})`;
  }

  // --- the strategy list and one strategy ---------------------------------------------------------

  function renderList() {
    const order = { champion: 0, core: 1, reserve: 2 };
    const rows = [...data.strategies].sort((a, b) => order[a.role] - order[b.role] || a.name.localeCompare(b.name));
    return rows
      .map((s) => {
        const last = s.verdicts[0];
        return (
          `<div class="strategy-item${s.name === selected ? " active" : ""}" data-name="${esc(s.name)}">` +
          `<div class="row" style="justify-content:space-between; align-items:baseline;"><b>${esc(s.name)}</b>${roleBadge(s.role)}</div>` +
          `<div class="muted strategy-summary">${esc(s.summary) || "no description"}</div>` +
          (last ? `<div style="font-size:.7rem;">${verdictBadge(last.verdict)} <span class="muted">${esc(last.date || "")}</span></div>` : "") +
          `</div>`
        );
      })
      .join("");
  }

  function renderTactics(s) {
    if (!s.tactics.length) return '<div class="muted">No tactics imported from utama_core/tactics (it defines its own).</div>';
    return s.tactics
      .map((t) => `<div class="ref-row"><span><b>${esc(t.name)}</b> <code class="muted">${esc(t.file)}</code></span></div>` +
        (t.description ? `<div class="muted" style="font-size:.72rem; margin:0 0 4px 8px;">${esc(t.description)}</div>` : ""))
      .join("");
  }

  function renderVerdicts(s) {
    if (!s.verdicts.length) {
      return `<div class="muted">Never evaluated. Run <code>pixi run evaluate ${esc(s.name)}</code> to see if it beats the champion.</div>`;
    }
    return s.verdicts
      .map((v) =>
        `<div class="ref-row"><span>${verdictBadge(v.verdict)} in the ${esc(v.decided_in)} after ${v.rounds} round(s) &middot; ` +
        `${testLine(v)} &middot; vs ${esc(v.reference)}</span><span class="muted">${esc(v.date || v.run_id)}</span></div>`)
      .join("");
  }

  function renderSignals(s) {
    const sig = s.signals;
    if (!sig) return '<div class="muted">No round with this strategy yet.</div>';
    const max = (r) => Math.max(Math.abs(r.value || 0), Math.abs(r.field || 0)) || 1;
    const fmt = (r, x) => (x == null ? "—" : r.key.endsWith("_share") || r.key === "stalled" ? pct(x) : num(x, r.key === "pass_progress_m" ? 2 : 1));
    const rows = sig.rows
      .map((r) => {
        const m = max(r);
        const w = r.value == null ? 0 : (Math.abs(r.value) / m) * 100;
        const fw = r.field == null ? null : (Math.abs(r.field) / m) * 100;
        const tick = fw == null ? "" : `<div class="signal-field" style="left:${fw}%;" title="other strategies' average"></div>`;
        return (
          `<div class="signal-row"><span class="signal-label">${esc(r.label)}</span>` +
          `<div class="signal-bar"><div class="signal-fill" style="width:${w}%;"></div>${tick}</div>` +
          `<span class="num">${fmt(r, r.value)}</span><span class="muted num">${fmt(r, r.field)}</span></div>`
        );
      })
      .join("");
    return (
      `<div class="muted" style="font-size:.72rem; margin-bottom:4px;">Per match, ${sig.matches} matches in ${esc(sig.run_id)} ` +
      `(${esc(sig.date || "")}). The tick and the grey number: the other strategies' average in that round. ` +
      `More is not better: these explain how it plays (docs/signals.md).</div>` +
      `<div class="signal-row muted" style="font-size:.66rem;"><span class="signal-label"></span><span></span><span class="num">this</span><span class="num">others</span></div>` +
      rows
    );
  }

  function renderMatchList(s) {
    if (!s.matches.length) return '<div class="muted">No recorded matches.</div>';
    const shown = showAllMatches ? s.matches : s.matches.slice(0, 30);
    const rows = shown
      .map((m) => {
        const [f, a] = m.score;
        const res = f > a ? "check-ok" : f < a ? "check-fail" : "muted";
        return (
          `<tr><td>vs ${esc(m.opponent)}</td><td class="num ${res}">${f} – ${a}</td>` +
          `<td class="num muted">${m.seed ?? "—"}</td><td class="muted">${esc(m.date || "")}</td>` +
          `<td>${m.stalled ? '<span class="marker-kind marker-stall">stalled</span>' : ""}</td>` +
          `<td>${replayButton(m.replay)}</td></tr>`
        );
      })
      .join("");
    const more =
      s.matches.length > shown.length
        ? `<button class="show-all-matches" style="margin-top:6px;">Show all ${s.matches.length}</button>`
        : "";
    return (
      `<table><thead><tr><th>Opponent</th><th class="num">Score</th><th class="num">Seed</th><th>Date</th><th></th><th></th></tr></thead>` +
      `<tbody>${rows}</tbody></table>${more}`
    );
  }

  function renderStrategy(s) {
    const old = s.old_names.length ? ` <span class="muted" style="font-size:.72rem;">(was ${s.old_names.map(esc).join(", ")})</span>` : "";
    const rest = s.description.split("\n\n").slice(1).join("\n\n");
    return `
      <div class="stack" style="padding:12px 16px; gap:14px;">
        <div>
          <div class="row" style="justify-content:space-between; align-items:baseline;"><h2 style="margin:0;">${esc(s.name)}${old}</h2>${roleBadge(s.role)}</div>
          <div style="margin-top:4px;">${esc(s.summary)}</div>
          ${rest ? `<div class="muted strategy-doc">${esc(rest)}</div>` : ""}
          <div class="muted" style="font-size:.72rem; margin-top:4px;"><code>${esc(s.file)}</code> &middot; watch it: <code>pixi run main ${esc(s.name)} wolfpack</code></div>
        </div>
        <div><div class="ref-section-title">Tactics it uses</div>${renderTactics(s)}</div>
        <div><div class="ref-section-title">Verdicts (against the champion)</div>${renderVerdicts(s)}</div>
        <div><div class="ref-section-title">How it plays</div>${renderSignals(s)}</div>
        <div><div class="ref-section-title">Matches (${s.matches.length})</div>${renderMatchList(s)}</div>
      </div>`;
  }

  function renderStrategiesPage() {
    if (!data) return '<div class="muted" style="padding:14px;">Loading…</div>';
    if (!selected || !data.strategies.find((s) => s.name === selected)) selected = data.champion || data.strategies[0]?.name;
    const s = data.strategies.find((x) => x.name === selected);
    return `
      <div class="strategies-layout">
        <div class="panel strategy-list">${renderList()}</div>
        <div class="panel" style="flex:1; overflow-y:auto;">${s ? renderStrategy(s) : ""}</div>
      </div>`;
  }

  // --- runs ---------------------------------------------------------------------------------------

  function renderVerdictCards(run) {
    const cands = Object.entries((run.report || {}).candidates || {});
    if (!run.report) {
      return run.run_id.startsWith("eval_")
        ? '<div class="muted">No report.json: the run did not finish.</div>'
        : '<div class="muted">An older round robin: no verdicts.</div>';
    }
    if (!cands.length) return '<div class="muted">No candidate: a health check of the field only.</div>';
    return cands
      .map(([name, r]) => {
        const decided = r[r.decided_in] || {};
        const h2h = r.vs_todays_champion;
        return (
          `<div class="verdict-card"><div class="row" style="justify-content:space-between;"><b>${esc(name)}</b>${verdictBadge(r.verdict)}</div>` +
          `<div>in the ${esc(r.decided_in)} after ${r.rounds} round(s) &middot; ${testLine(decided)}</div>` +
          `<div class="muted" style="font-size:.72rem;">against ${esc(run.reference)} (the champion at the last release)` +
          (h2h ? `; against today's ${esc(h2h.name)}: ${signed(h2h.mean)} ± ${num(h2h.se)}` : "") +
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
            `<div class="ref-row"><span><span class="marker-kind marker-stall">stall</span>${esc(inc.kind)} at ${inc.sim_time.toFixed(1)}s ` +
            `for ${inc.duration_s.toFixed(0)}s</span><span class="row">` +
            inc.matches.map((m) => replayButton(replayOfMatch(round, m), inc.sim_time, m)).join("") +
            `</span></div>`
        )
        .join("");
      parts.push(`<div class="ref-section-title">Stalls: ${incidents.length} incidents</div>` + (rows || '<div class="muted">none</div>'));
    }
    const rs = round.restarts;
    if (rs && rs.restarts) {
      parts.push(
        `<div class="ref-section-title">Restarts: ${rs.reached_normal_start}/${rs.restarts} reached NORMAL_START (${pct(rs.reached_normal_start / rs.restarts)})</div>`
      );
    }
    if (round.fouls) {
      const byRule = Object.entries(round.fouls)
        .map(([rule, f]) => [rule, f.total])
        .sort((a, b) => b[1] - a[1])
        .map(([rule, n]) => `${n} ${rule.replace(/_/g, " ")}`)
        .join(", ");
      parts.push(`<div class="ref-section-title">Fouls</div><div>${byRule || "none"}</div>`);
    }
    if ((round.failed || []).length) {
      parts.push(`<div class="ref-section-title check-fail">${round.failed.length} match(es) crashed</div>`);
    }
    return parts.join("") || '<div class="muted">No health data in this round.</div>';
  }

  function renderRoundMatches(round) {
    const rows = (round.results || [])
      .map((r) => {
        return (
          `<tr><td>${esc(r.display_a)} <span class="muted">(yellow)</span> vs ${esc(r.display_b)}</td>` +
          `<td class="num">${r.score_a} – ${r.score_b}</td>` +
          `<td>${r.stalled ? '<span class="marker-kind marker-stall">stalled</span>' : ""}${r.reused ? '<span class="muted" style="font-size:.68rem;">from store</span>' : ""}</td>` +
          `<td>${replayButton(r.replay)}</td></tr>`
        );
      })
      .join("");
    return `<table><tbody>${rows}</tbody></table>`;
  }

  function renderRun(run) {
    const meta = (run.report || {}).run || {};
    const commit = meta.git_commit ? ` &middot; <code>${esc(meta.git_commit.slice(0, 8))}</code>${meta.git_dirty ? " (dirty)" : ""}` : "";
    const nMatches = run.rounds.reduce((n, s) => n + (s.results || []).length, 0);
    const open = expandedRun[run.run_id];
    const header =
      `<div class="panel-title run-header" data-run-id="${esc(run.run_id)}" style="cursor:pointer;">${open ? "▼" : "▶"} ` +
      `${esc(run.run_id)} &middot; ${run.rounds.length} round(s) &middot; ${nMatches} matches${commit}</div>`;
    if (!open) return `<div class="panel" style="margin-bottom:8px;">${header}</div>`;
    const rounds = run.rounds
      .map((round) => {
        const key = `${run.run_id}:${round.round ?? 1}`;
        const o = expandedRound[key];
        return (
          `<div class="round-header" data-key="${esc(key)}" style="cursor:pointer; margin-top:6px;">${o ? "▼" : "▶"} ` +
          `Round ${round.round ?? 1}${round.seed != null ? ` (seed ${round.seed})` : ""} &middot; ${(round.results || []).length} matches</div>` +
          (o ? renderRoundMatches(round) : "")
        );
      })
      .join("");
    return `
      <div class="panel" style="margin-bottom:12px;">
        ${header}
        <div class="stack" style="padding:8px 14px 12px; gap:8px; font-size:.8rem;">
          <div><div class="ref-section-title">Verdicts</div>${renderVerdictCards(run)}</div>
          <div><div class="ref-section-title">Health check (round 1)</div>${run.rounds[0] ? renderHealth(run.rounds[0]) : ""}</div>
          <div><div class="ref-section-title">Matches</div>${rounds}</div>
        </div>
      </div>`;
  }

  function renderRunsPage() {
    if (!runs) return '<div class="muted" style="padding:14px;">Loading…</div>';
    if (!runs.length) return '<div class="muted" style="padding:14px;">No runs yet. Start one with <code>pixi run evaluate NAME</code>.</div>';
    return runs.map(renderRun).join("");
  }

  // --- shell --------------------------------------------------------------------------------------

  function render() {
    const root = document.getElementById("strategies-root");
    if (!root) return;
    const tab = (p, label) => `<button class="subtab${page === p ? " active" : ""}" data-page="${p}">${label}</button>`;
    root.innerHTML =
      `<div class="row" style="margin-bottom:8px;">${tab("strategies", "Strategies")}${tab("runs", "Evaluation runs")}</div>` +
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
        showAllMatches = false;
        render();
      });
    }
    for (const b of root.querySelectorAll(".show-all-matches")) {
      b.addEventListener("click", () => {
        showAllMatches = true;
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
        if (runs.length && !Object.keys(expandedRun).length) expandedRun[runs[0].run_id] = true;
        render();
      })
      .catch((err) => console.error("runs fetch error:", err));
  }

  Dashboard.registerView("strategies", { mount: render, onShow: load });
})();

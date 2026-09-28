// Tournament view: standings + per-match results, sortable/expandable.
// Successor concern to hand-reading full_match_tournament.py log output —
// reads replays/<run_id>/summary.json via the tournament view's Python side.
// A smoke_tournament.py summary also carries a per-strategy table, stall
// incidents, restart outcomes and fouls (docs/STRATEGY_DEVELOPMENT.md
// "Reading a tournament run"); those are shown when present. Each result's
// `replay` is the path of its replay file, or null (added server-side).
// `full_match_tournament.py` writes `winner` as the literal string "draw" on
// a tie (never null), and `standings` only carries wins/draws per config —
// losses/GF/GA/win% are derived here from `results`, not shipped precomputed.

(function () {
  let runs = [];
  let sortKey = { column: "wins", dir: -1 };
  let expandedMatch = {}; // "runIndex:matchIndex" -> bool
  let expandedRun = {}; // run_id -> bool; only the newest run starts open

  function short(name) {
    return name.replace(/^build_/, "").replace(/_kernel_strategy$/, "");
  }

  function pct(x) {
    return Math.round((x || 0) * 1000) / 10 + "%";
  }

  // Derive losses/games/GF/GA/win% per config from `results`, since
  // `standings` in summary.json only ever carries wins/draws.
  function standingsRows(run) {
    if (run.strategies) return sortRows(strategyRows(run));
    const base = {};
    for (const name of run.config_names || []) {
      base[name] = { name, wins: 0, draws: 0, losses: 0, games: 0, gf: 0, ga: 0 };
    }
    for (const r of run.results || []) {
      const a = base[r.config_a] || (base[r.config_a] = { name: r.config_a, wins: 0, draws: 0, losses: 0, games: 0, gf: 0, ga: 0 });
      const b = base[r.config_b] || (base[r.config_b] = { name: r.config_b, wins: 0, draws: 0, losses: 0, games: 0, gf: 0, ga: 0 });
      a.games++;
      b.games++;
      a.gf += r.score_a;
      a.ga += r.score_b;
      b.gf += r.score_b;
      b.ga += r.score_a;
      if (r.winner === "draw") {
        a.draws++;
        b.draws++;
      } else if (r.winner === r.config_a) {
        a.wins++;
        b.losses++;
      } else if (r.winner === r.config_b) {
        b.wins++;
        a.losses++;
      }
    }
    const rows = Object.values(base).map((s) => ({
      ...s,
      gd: s.gf - s.ga,
      winPct: s.games ? s.wins / s.games : 0,
    }));
    return sortRows(rows);
  }

  function sortRows(rows) {
    const { column, dir } = sortKey;
    rows.sort((x, y) => ((x[column] ?? 0) - (y[column] ?? 0)) * dir);
    return rows;
  }

  // smoke_tournament.py's per-strategy table (summary.json `strategies`).
  function strategyRows(run) {
    return Object.entries(run.strategies).map(([name, s]) => ({
      name,
      wins: s.wins,
      draws: s.draws,
      losses: s.losses,
      games: s.matches,
      gf: s.goals_for,
      ga: s.goals_against,
      gd: s.goals_for - s.goals_against,
      winPct: s.matches ? s.wins / s.matches : 0,
      shots: s.shots,
      passes: s.completed_passes,
      entries: s.attacking_third_entries,
      fouls: s.fouls,
      realLosses: s.real_losses_as_a,
      stalled: s.stalled,
    }));
  }

  function renderStandings(run, runIndex) {
    const rows = standingsRows(run);
    const arrow = (col) => (sortKey.column === col ? (sortKey.dir === -1 ? " ▼" : " ▲") : "");
    const cols = [
      ["wins", "W"],
      ["draws", "D"],
      ["losses", "L"],
      ["games", "Games"],
      ["gf", "GF"],
      ["ga", "GA"],
      ["gd", "GD"],
      ["winPct", "Win%"],
    ];
    if (run.strategies) {
      cols.push(
        ["shots", "Shots"],
        ["passes", "Passes"],
        ["entries", "Entries"],
        ["fouls", "Fouls"],
        ["realLosses", "Losses as A"],
        ["stalled", "Stalled"]
      );
    }
    const cell = (r, col) => {
      if (col === "gd") return `${r.gd > 0 ? "+" : ""}${r.gd}`;
      if (col === "winPct") return pct(r.winPct);
      return r[col] ?? "—";
    };
    const trs = rows
      .map((r) => `<tr><td>${short(r.name)}</td>` + cols.map(([col]) => `<td class="num">${cell(r, col)}</td>`).join("") + `</tr>`)
      .join("");
    const ths = cols
      .map(([col, label]) => `<th class="sortable" data-run="${runIndex}" data-col="${col}" style="cursor:pointer;">${label}${arrow(col)}</th>`)
      .join("");
    return `
      <table>
        <thead><tr><th>Config</th>${ths}</tr></thead>
        <tbody>${trs}</tbody>
      </table>`;
  }

  function replayButton(path, atTime, label) {
    if (!path) return `<span class="muted" style="font-size:.72rem;">no replay file</span>`;
    const at = atTime != null ? ` data-at="${atTime}"` : "";
    return `<button class="btn-accent view-replay-btn" data-path="${path}"${at}>${label || "View replay"}</button>`;
  }

  function teamAvg(perRobot, prefix) {
    const vals = Object.entries(perRobot || {})
      .filter(([k]) => k.startsWith(prefix))
      .map(([, v]) => v);
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  function renderMatchDetail(r) {
    const stats = r.stats || {};
    const rec = stats.rule_event_counts || {};
    const recParts = Object.entries(rec).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`);
    const shots = stats.shots || {};
    const friendlyMotion = teamAvg(stats.robot_motion_pct, "friendly_");
    const enemyMotion = teamAvg(stats.robot_motion_pct, "enemy_");

    return `
      <div class="stack" style="padding:8px 14px 12px; gap:4px; background:var(--surface-2);">
        <div class="ref-row"><span class="muted">Rule events</span><span>${recParts.length ? recParts.join(", ") : "none"}</span></div>
        <div class="ref-row"><span class="muted">Shots</span><span>${shots.friendly ?? 0} - ${shots.enemy ?? 0}</span></div>
        <div class="ref-row"><span class="muted">Ball travel</span><span>${stats.ball_travel_m != null ? stats.ball_travel_m.toFixed(1) + " m" : "—"}</span></div>
        <div class="ref-row"><span class="muted">Avg motion</span><span>${friendlyMotion != null ? pct(friendlyMotion) : "—"} - ${enemyMotion != null ? pct(enemyMotion) : "—"}</span></div>
        <div class="row">${replayButton(r.replay)}</div>
      </div>`;
  }

  // Fixed, predictable ordering for the 4 side x kickoff cells within a
  // pair's group — (a_is_right, a_kicks_off) in this order — rather than
  // whatever order concurrent workers happened to finish in.
  const CELL_ORDER = [
    [true, true],
    [true, false],
    [false, true],
    [false, false],
  ];

  function cellLabel(r, pairA) {
    // smoke_tournament.py plays each pair once, config_a as yellow
    if (r.a_is_right === undefined) return `${short(r.config_a)} (yellow) vs ${short(r.config_b)}`;
    // r.config_a may be either physical config depending on which side of
    // the 4-cell product ran; re-express relative to pairA (the group's
    // canonical/alphabetically-first config) so labels read consistently
    // within a group regardless of which config was "a" for this row.
    const aIsPairA = r.config_a === pairA;
    const isRight = aIsPairA ? r.a_is_right : !r.a_is_right;
    const kicksOff = aIsPairA ? r.a_kicks_off : !r.a_kicks_off;
    return `${short(pairA)} ${isRight ? "right" : "left"}, ${kicksOff ? short(pairA) : "opponent"} kicks off`;
  }

  // Group flat `results` into one entry per unordered {config_a, config_b}
  // pair, each holding up to 4 cell rows ordered by CELL_ORDER. Groups are
  // ordered alphabetically by the pair's sorted names for a stable render
  // independent of match-completion order.
  function groupedMatches(run) {
    const groups = {}; // "confA|confB" (sorted) -> { pairA, pairB, rows: [] }
    (run.results || []).forEach((r, i) => {
      const [pairA, pairB] = [r.config_a, r.config_b].sort();
      const key = pairA + "|" + pairB;
      if (!groups[key]) groups[key] = { pairA, pairB, rows: [] };
      groups[key].rows.push({ r, resultIndex: i });
    });

    return Object.keys(groups)
      .sort()
      .map((key) => {
        const g = groups[key];
        g.rows.sort((x, y) => {
          const cellOf = (row) => {
            const aIsPairA = row.r.config_a === g.pairA;
            const isRight = aIsPairA ? row.r.a_is_right : !row.r.a_is_right;
            const kicksOff = aIsPairA ? row.r.a_kicks_off : !row.r.a_kicks_off;
            return CELL_ORDER.findIndex(([r_, k_]) => r_ === isRight && k_ === kicksOff);
          };
          return cellOf(x) - cellOf(y);
        });
        return g;
      });
  }

  function renderMatches(run, runIndex) {
    const groups = groupedMatches(run);
    const rows = groups
      .map((g) => {
        const groupHeader =
          `<tr><td colspan="5" class="muted" style="padding:8px 10px 4px; border-bottom:none; font-size:.68rem; letter-spacing:.04em; text-transform:uppercase;">` +
          `${short(g.pairA)} vs ${short(g.pairB)}</td></tr>`;
        const matchRows = g.rows
          .map(({ r, resultIndex }) => {
            const key = runIndex + ":" + resultIndex;
            const isDraw = r.winner === "draw";
            const winnerLabel = isDraw ? "draw" : short(r.winner);
            const poss = (r.stats || {}).possession_pct;
            const possLabel = poss ? `${pct(poss.friendly)} - ${pct(poss.enemy)}` : "—";
            const expanded = !!expandedMatch[key];
            const detailRow = expanded
              ? `<tr><td colspan="5" style="padding:0;">${renderMatchDetail(r)}</td></tr>`
              : "";
            return (
              `<tr class="match-row" data-key="${key}" style="cursor:pointer;">` +
              `<td>${cellLabel(r, g.pairA)}</td>` +
              `<td class="num">${r.score_a} - ${r.score_b}</td>` +
              `<td>${winnerLabel}${((r.stats || {}).stall_events || []).length ? ' <span class="marker-kind marker-stall">stalled</span>' : ""}</td>` +
              `<td class="num">${possLabel}</td>` +
              `<td class="muted" style="text-align:center;">${expanded ? "▲" : "▼"}</td>` +
              `</tr>${detailRow}`
            );
          })
          .join("");
        return groupHeader + matchRows;
      })
      .join("");

    return `
      <div style="overflow-x:auto; padding:8px 14px;">
        <table>
          <thead>
            <tr><th>Cell</th><th class="num">Score</th><th>Winner</th><th class="num">Possession</th><th></th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // Find the replay of the match named "<a>_vs_<b>" (how stall incidents name matches).
  function replayOfMatch(run, stem) {
    const r = (run.results || []).find((x) => x.replay && x.replay.split("/").pop().startsWith(stem + "."));
    return r ? r.replay : null;
  }

  function renderDiagnostics(run) {
    const parts = [];
    const incidents = run.stall_incidents || [];
    if (run.stall_incidents) {
      const rows = incidents
        .map(
          (inc) =>
            `<div class="ref-row"><span><span class="marker-kind marker-stall">stall</span>${inc.kind} at ${inc.sim_time.toFixed(1)}s ` +
            `for ${inc.duration_s.toFixed(0)}s</span><span class="row">` +
            inc.matches.map((m) => replayButton(replayOfMatch(run, m), inc.sim_time, m)).join("") +
            `</span></div>`
        )
        .join("");
      parts.push(
        `<div class="ref-section-title">Stalls: ${run.stalled_match_count ?? incidents.length} matches, ${incidents.length} incidents</div>` +
          (rows || '<div class="muted">none</div>')
      );
    }
    const rs = run.restarts;
    if (rs && rs.restarts) {
      const kinds = Object.entries(rs.by_kind || {})
        .map(([kind, k]) => {
          const bad = Object.entries(k.outcomes || {})
            .filter(([o]) => o !== "taken")
            .map(([o, n]) => `${n} ${o.replace(/_/g, " ")}`)
            .join(", ");
          return `<div class="ref-row"><span class="muted">${kind.replace(/_/g, " ").toLowerCase()}</span>` +
            `<span>${k.reached_normal_start}/${k.n} reached NORMAL_START${bad ? " (" + bad + ")" : ""}</span></div>`;
        })
        .join("");
      parts.push(
        `<div class="ref-section-title">Restarts: ${rs.reached_normal_start}/${rs.restarts} reached NORMAL_START (${pct(rs.reached_normal_start / rs.restarts)})</div>` +
          kinds
      );
    }
    if (run.fouls) {
      const byRule = Object.entries(run.fouls)
        .map(([rule, f]) => [rule, f.total])
        .sort((a, b) => b[1] - a[1])
        .map(([rule, n]) => `${n} ${rule.replace(/_/g, " ")}`)
        .join(", ");
      parts.push(`<div class="ref-section-title">Fouls</div><div>${byRule || "none"}</div>`);
    }
    return parts.length ? `<div class="stack" style="padding:4px 14px 8px; gap:4px; font-size:.78rem;">${parts.join("")}</div>` : "";
  }

  function renderRunHeader(run) {
    const meta = run.run || {};
    const commit = meta.git_commit ? ` &middot; <code>${meta.git_commit.slice(0, 8)}</code>${meta.git_dirty ? " (dirty)" : ""}` : "";
    const argv = meta.argv && meta.argv.length ? ` &middot; <span class="muted">${meta.argv.join(" ")}</span>` : "";
    const open = expandedRun[run.run_id];
    return `<div class="panel-title run-header" data-run-id="${run.run_id}" style="cursor:pointer;">${open ? "▼" : "▶"} ${run.run_id} &middot; ` +
      `${(run.config_names || []).length} configs &middot; ${(run.results || []).length} matches${commit}${argv}</div>`;
  }

  function renderRun(run, index) {
    if (!expandedRun[run.run_id]) return `<div class="panel" style="margin-bottom:8px;">${renderRunHeader(run)}</div>`;
    return `
      <div class="panel" style="margin-bottom:12px;">
        ${renderRunHeader(run)}
        <div style="overflow-x:auto; padding:8px 14px 0;">${renderStandings(run, index)}</div>
        ${renderDiagnostics(run)}
        <div class="ref-section-title" style="margin:8px 14px 0;">Matches</div>
        ${renderMatches(run, index)}
      </div>`;
  }

  function render() {
    const container = document.getElementById("tournament-runs");
    if (!container) return;
    if (!runs.length) {
      container.innerHTML = '<div class="muted" style="padding:14px;">No tournament runs found.</div>';
      return;
    }
    container.innerHTML = runs.map(renderRun).join("");

    for (const th of container.querySelectorAll("th.sortable")) {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (sortKey.column === col) {
          sortKey.dir *= -1;
        } else {
          sortKey = { column: col, dir: -1 };
        }
        render();
      });
    }
    for (const header of container.querySelectorAll(".run-header")) {
      header.addEventListener("click", () => {
        const id = header.dataset.runId;
        expandedRun[id] = !expandedRun[id];
        render();
      });
    }
    for (const row of container.querySelectorAll("tr.match-row")) {
      row.addEventListener("click", (e) => {
        if (e.target.closest(".view-replay-btn")) return;
        const key = row.dataset.key;
        expandedMatch[key] = !expandedMatch[key];
        render();
      });
    }
    for (const btn of container.querySelectorAll(".view-replay-btn")) {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const path = btn.dataset.path;
        const at = btn.dataset.at != null ? Number(btn.dataset.at) : undefined;
        Dashboard.showReplay(path, at);
      });
    }
  }

  function load() {
    fetch("/tournament/runs")
      .then((r) => r.json())
      .then((data) => {
        runs = data;
        if (runs.length && !Object.keys(expandedRun).length) expandedRun[runs[0].run_id] = true;
        render();
      })
      .catch((err) => console.error("tournament fetch error:", err));
  }

  function mount() {
    load();
  }

  function onShow() {
    load();
  }

  Dashboard.registerView("tournament", { mount, onShow });
})();

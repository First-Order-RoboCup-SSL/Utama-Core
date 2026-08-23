// Tournament view: standings + per-match results, sortable/expandable.
// Successor concern to hand-reading full_match_tournament.py log output —
// reads replays/<run_id>/summary.json via the tournament view's Python side.
// `full_match_tournament.py` writes `winner` as the literal string "draw" on
// a tie (never null), and `standings` only carries wins/draws per config —
// losses/GF/GA/win% are derived here from `results`, not shipped precomputed.

(function () {
  let runs = [];
  let sortKey = { column: "wins", dir: -1 };
  let expandedMatch = {}; // "runIndex:matchIndex" -> bool

  function short(name) {
    return name.replace(/^build_/, "").replace(/_kernel_strategy$/, "");
  }

  function pct(x) {
    return Math.round((x || 0) * 1000) / 10 + "%";
  }

  // Derive losses/games/GF/GA/win% per config from `results`, since
  // `standings` in summary.json only ever carries wins/draws.
  function standingsRows(run) {
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
    const { column, dir } = sortKey;
    rows.sort((x, y) => (x[column] - y[column]) * dir);
    return rows;
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
    const trs = rows
      .map(
        (r) =>
          `<tr><td>${short(r.name)}</td>` +
          `<td class="num">${r.wins}</td><td class="num">${r.draws}</td><td class="num">${r.losses}</td>` +
          `<td class="num">${r.games}</td><td class="num">${r.gf}</td><td class="num">${r.ga}</td>` +
          `<td class="num">${r.gd > 0 ? "+" : ""}${r.gd}</td><td class="num">${pct(r.winPct)}</td></tr>`
      )
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

  // Replay filename convention, mirrored from full_match_tournament.py's
  // `run_match_cell`: "{short_a}_vs_{short_b}_{side_tag}{kickoff_tag}.pkl",
  // side_tag = R/L for a_is_right, kickoff_tag = K/k for a_kicks_off.
  function replayPath(run, r) {
    const sideTag = r.a_is_right ? "R" : "L";
    const kickoffTag = r.a_kicks_off ? "K" : "k";
    return `${run.run_id}/${short(r.config_a)}_vs_${short(r.config_b)}_${sideTag}${kickoffTag}.pkl`;
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
        <div class="row">
          <button class="btn-accent view-replay-btn" data-path="${replayPath(r.__run, r)}">View replay</button>
        </div>
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
            r.__run = run; // attach for renderMatchDetail's replayPath lookup
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
              `<td>${winnerLabel}</td>` +
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

  function renderRun(run, index) {
    return `
      <div class="panel" style="margin-bottom:12px;">
        <div class="panel-title">${run.run_id} &middot; ${(run.config_names || []).length} configs &middot; ${(run.results || []).length} matches</div>
        <div style="overflow-x:auto; padding:8px 14px 0;">${renderStandings(run, index)}</div>
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
        if (window.Dashboard && Dashboard.showReplay) Dashboard.showReplay(path);
      });
    }
  }

  function load() {
    fetch("/tournament/runs")
      .then((r) => r.json())
      .then((data) => {
        runs = data;
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

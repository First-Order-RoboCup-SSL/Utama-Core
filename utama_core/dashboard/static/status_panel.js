// Shared robot/tactic status-panel rendering, used by both the Live view
// (live SSE state) and the Replay view (indexed frame from a loaded array) —
// same DOM structure, same markup, so a replay looks like the live match it
// was recorded from, not a second implementation with its own drift.

// Renders the score/command/stage header block both views share. `ids` is
// {yellowScore, blueScore, command, stage} element ids; `state` is
// {yellow_score, blue_score, command, stage} or null (all placeholders).
function renderRefereeHeaderInto(ids, state) {
  const yellowEl = document.getElementById(ids.yellowScore);
  const blueEl = document.getElementById(ids.blueScore);
  const commandEl = document.getElementById(ids.command);
  const stageEl = document.getElementById(ids.stage);
  if (yellowEl) yellowEl.textContent = state && state.yellow_score !== undefined ? state.yellow_score : "—";
  if (blueEl) blueEl.textContent = state && state.blue_score !== undefined ? state.blue_score : "—";
  if (commandEl) {
    commandEl.textContent = state && state.command ? state.command.replace(/_/g, " ") : "—";
    commandEl.className = state && state.command === "HALT" ? "accent" : "";
  }
  if (stageEl) stageEl.textContent = state && state.stage ? state.stage.replace(/_/g, " ") : "—";
}

function renderTacticStatusInto(containerId, tacticStatus, options) {
  const c = document.getElementById(containerId);
  if (!c) return;
  const status = tacticStatus || {};
  const robotIds = Object.keys(status).sort((a, b) => Number(a) - Number(b));
  const opts = options || {};

  if (robotIds.length === 0) {
    c.innerHTML = '<div class="muted" style="font-size:.72rem;">' + (opts.emptyLabel || "no tactic data") + "</div>";
    return;
  }

  let html = "";
  for (const robotId of robotIds) {
    const labels = status[robotId] || [];
    const committed = labels.some((l) => l.includes("committed"));
    html +=
      '<div class="ref-row"><span class="' +
      (committed ? "accent" : "") +
      '">' +
      robotId +
      "</span><span>" +
      labels.map((l) => l.replace(/\s*\(committed\)/, "")).join(", ") +
      "</span>" +
      (committed ? '<span class="muted" style="font-size:.65rem;">committed</span>' : "") +
      "</div>";
  }
  c.innerHTML = html;
}

function renderRobotStatusInto(containerId, d, options) {
  const c = document.getElementById(containerId);
  if (!c) return;
  const opts = options || {};
  let html = "";

  if (!opts.hideFeedback) {
    const feedback = d.robot_feedback || [];
    html += '<div class="ref-section-title">Controller feedback (' + feedback.length + ")</div>";
    for (const row of feedback) {
      const label = row.team_color ? row.team_color + " " + row.vision_id : "port " + row.port_id;
      html +=
        '<div class="ref-row"><span>' +
        label +
        "</span><span class=\"muted\">" +
        (row.connected ? (row.has_ball ? "ball" : "connected") : "no data") +
        "</span></div>";
    }
  }

  html += '<div class="ref-section-title">Friendly</div>';
  for (const bot of (d.robots && d.robots.friendly) || []) {
    html +=
      '<div class="ref-row"><span>' +
      bot.id +
      '</span><span class="muted">(' +
      bot.x.toFixed(2) +
      ", " +
      bot.y.toFixed(2) +
      ")</span></div>";
  }

  c.innerHTML = html;
}

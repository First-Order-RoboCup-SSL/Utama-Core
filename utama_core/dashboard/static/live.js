// Live view: field canvas + referee command buttons + profile config readout.
// Successor to custom_referee/gui.py's single-page HTML/JS — same behavior,
// registered as one tab against the shared dashboard server. Named "Live"
// (not "Referee") because it's the one view for watching+controlling a
// running match — a tactic-assignment panel lives here too, not in a
// separate tab, since both read off the same running match at the same
// instant.

(function () {
  let fieldView = null;
  let cfg = null;
  let currentCmd = null;

  function send(command, designated) {
    const body = designated ? { command, designated } : { command };
    fetch("/referee/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch((err) => console.error("command error:", err));
  }

  const REF_HEADER_IDS = {
    yellowScore: "ref-yellow-score",
    blueScore: "ref-blue-score",
    command: "ref-command",
    stage: "ref-stage",
  };

  function onEvent(d) {
    renderRefereeHeaderInto(REF_HEADER_IDS, d);
    currentCmd = d.command;

    let layoutChanged = false;
    // With a camera (field_check present), draw the field as ssl-vision sees it, +x to the
    // right, so the picture matches the pitch in front of the operator; otherwise our goal
    // on the right, as in replays.
    const drawRight = d.field_check ? true : d.my_team_is_right;
    if (drawRight !== undefined && drawRight !== window._refMyTeamIsRight) {
      window._refMyTeamIsRight = drawRight;
      layoutChanged = true;
    }
    if (d.my_team_is_yellow !== undefined && d.my_team_is_yellow !== window._refMyTeamIsYellow) {
      window._refMyTeamIsYellow = d.my_team_is_yellow;
      layoutChanged = true;
    }
    if (fieldView && layoutChanged) fieldView.setTeamLayout(drawRight, d.my_team_is_yellow);

    renderFieldCheck(d.field_check);
    renderRobotStatusInto("ref-status-entries", d, { hideFeedback: !!d.field_check }); // the field check has the radio rows
    renderTacticStatusInto("ref-tactic-entries", d.tactic_status);
    d.tactic_tags = tacticTagsFromStatus(d.tactic_status);
    if (fieldView) fieldView.draw(d);
  }

  // The runner's field checks (utama_core/run/field_check.py): the failures first, so a
  // red row is never scrolled away; on real robots any failure has already forced HALT.
  function renderFieldCheck(fc) {
    const box = document.getElementById("ref-field-check");
    const banner = document.getElementById("ref-auto-halt");
    if (!fc) {
      box.innerHTML = "";
      banner.style.display = "none";
      return;
    }
    banner.style.display = fc.auto_halt ? "" : "none";
    banner.textContent = fc.auto_halt ? "Auto-HALT " + fc.auto_halt : "";
    const order = { fail: 0, warn: 1, ok: 2 };
    const rows = fc.rows.slice().sort((a, b) => order[a.status] - order[b.status]);
    const fails = rows.filter((r) => r.status === "fail").length;
    let html =
      '<div class="ref-section-title">Field check — ' +
      (fails ? '<span class="check-fail">' + fails + " failing</span>" : '<span class="check-ok">ready</span>') +
      "</div>";
    for (const r of rows) {
      html +=
        '<div class="ref-row"><span class="check-' + r.status + '">' + (r.status === "ok" ? "●" : r.status === "warn" ? "▲" : "✖") +
        " " + r.name + '</span><span class="muted">' + r.detail + "</span></div>";
    }
    box.innerHTML = html;
  }

  function mount() {
    fetch("/referee/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((c) => {
        if (!c) {
          // dashboard_server.py alone serves Replay and Tournament; a live match attaches this view
          document.getElementById("ref-profile-name").textContent = "No live match attached";
          return;
        }
        cfg = c;
        const canvas = document.getElementById("ref-field-canvas");
        fieldView = new FieldCanvas(canvas, c.geometry, {});
        document.getElementById("ref-profile-name").textContent = c.profile_name;
        Dashboard.showView("live"); // a running match is what this page is open for
      });

    const es = new EventSource("/events/referee");
    es.addEventListener("referee", (ev) => onEvent(JSON.parse(ev.data)));
    es.onopen = () => document.getElementById("ref-conn-dot").classList.add("live");
    es.onerror = () => document.getElementById("ref-conn-dot").classList.remove("live");

    for (const btn of document.querySelectorAll("#view-live button[data-cmd]")) {
      btn.addEventListener("click", () => send(btn.dataset.cmd));
    }

    document.addEventListener("keydown", (e) => {
      if (!document.getElementById("view-live").classList.contains("active")) return;
      // Halt only: a stray key press must never start the robots.
      if (e.key === " ") {
        e.preventDefault();
        send("HALT");
      }
    });
  }

  Dashboard.registerView("live", { mount });
})();

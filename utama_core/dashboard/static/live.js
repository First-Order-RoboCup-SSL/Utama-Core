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
    if (d.my_team_is_right !== undefined && d.my_team_is_right !== window._refMyTeamIsRight) {
      window._refMyTeamIsRight = d.my_team_is_right;
      layoutChanged = true;
    }
    if (d.my_team_is_yellow !== undefined && d.my_team_is_yellow !== window._refMyTeamIsYellow) {
      window._refMyTeamIsYellow = d.my_team_is_yellow;
      layoutChanged = true;
    }
    if (fieldView && layoutChanged) fieldView.setTeamLayout(d.my_team_is_right, d.my_team_is_yellow);

    renderRobotStatusInto("ref-status-entries", d);
    renderTacticStatusInto("ref-tactic-entries", d.tactic_status);
    if (fieldView) fieldView.draw(d);
  }

  function mount() {
    fetch("/referee/config")
      .then((r) => r.json())
      .then((c) => {
        cfg = c;
        const canvas = document.getElementById("ref-field-canvas");
        fieldView = new FieldCanvas(canvas, c.geometry, {});
        document.getElementById("ref-profile-name").textContent = c.profile_name;
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
      if (e.key === " ") {
        e.preventDefault();
        send(currentCmd === "HALT" ? "FORCE_START" : "HALT");
      }
    });
  }

  Dashboard.registerView("live", { mount });
})();

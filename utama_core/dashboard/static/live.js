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
  let timeline = null; // Timeline (timeline.js), the last minutes up to now
  // States arrive at the runner's rate (60 Hz); the page draws the latest once per screen frame
  // and refreshes the side panel's text a few times a second (rebuilt every state, a click on
  // it could land on an element already replaced).
  let latest = null;
  let drawQueued = false;
  let panelDrawnAt = 0;
  let panelPending = false;
  const PANEL_EVERY_MS = 250;

  // A command the server refused (a placement outside the area in play) or never got (the
  // network) is shown under the buttons, so a Halt that did not go through is seen at once.
  let sendErrorTimer = null;
  function showSendError(command, reason) {
    const el = document.getElementById("ref-send-error");
    el.textContent = command + " not sent: " + reason;
    clearTimeout(sendErrorTimer);
    sendErrorTimer = setTimeout(() => (el.textContent = ""), 6000);
  }
  function send(command, designated) {
    const body = designated ? { command, designated } : { command };
    fetch("/referee/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => (r.ok ? null : r.text().then((t) => showSendError(command, t || "HTTP " + r.status))))
      .catch((err) => showSendError(command, err.message || "no connection"));
  }

  // States come at the runner's rate; none for half a second means the page is showing the
  // past (the match stopped, the network dropped): the dot goes grey and the gap is shown.
  let lastEventAt = null;
  let streamOpen = false;
  function checkStale() {
    const age = lastEventAt == null ? 0 : (performance.now() - lastEventAt) / 1000;
    const stale = age > 0.5;
    document.getElementById("ref-conn-dot").classList.toggle("live", streamOpen && !stale);
    document.getElementById("ref-stale").textContent = stale ? "no update for " + Math.floor(age) + " s" : "";
  }

  const REF_HEADER_IDS = {
    yellowScore: "ref-yellow-score",
    blueScore: "ref-blue-score",
    command: "ref-command",
    stage: "ref-stage",
  };

  function onEvent(d) {
    lastEventAt = performance.now();
    currentCmd = d.command;
    d.tactic_tags = tacticTagsFromStatus(d.tactic_status);
    if (timeline && d.ts != null) timeline.append(d.ts, d);
    latest = d;
    if (!drawQueued) {
      drawQueued = true;
      requestAnimationFrame(render);
    }
  }

  function render() {
    drawQueued = false;
    const d = latest;
    // The command and the auto-HALT banner are safety-relevant: drawn first, every frame, so
    // neither waits on the panel's throttle or fails with a drawing error further down.
    renderRefereeHeaderInto(REF_HEADER_IDS, d);
    renderAutoHalt(d.field_check);
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
    if (fieldView) fieldView.draw(d);
    if (timeline && timeline.ts != null) timeline.draw(timeline.ts);

    const now = performance.now();
    if (now - panelDrawnAt < PANEL_EVERY_MS) {
      // Skipped: come back for it, so the panel shows the last state even if the stream stops.
      if (!panelPending) {
        panelPending = true;
        setTimeout(() => {
          panelPending = false;
          if (!drawQueued) {
            drawQueued = true;
            requestAnimationFrame(render);
          }
        }, PANEL_EVERY_MS);
      }
      return;
    }
    panelDrawnAt = now;
    renderFieldCheck(d.field_check);
    // With a field check, its rows already give each robot's position and radio.
    if (d.field_check) document.getElementById("ref-status-entries").innerHTML = "";
    else renderRobotStatusInto("ref-status-entries", d);
    renderTacticStatusInto("ref-tactic-entries", d.tactic_status);
  }

  // Right-click (long-press on a phone) on the field: ball placement to that spot, by either
  // team. In a simulator the ball is moved there; on real robots the placing team's robots
  // (or a person) bring it.
  function mountPlacementMenu(canvas) {
    const menu = document.getElementById("ref-place-menu");
    let target = null;
    let openedAt = 0;
    const hide = () => (menu.style.display = "none");
    const openAt = (clientX, clientY) => {
      if (!fieldView) return;
      const rect = canvas.getBoundingClientRect();
      const p = fieldView.toField(
        (clientX - rect.left) * (canvas.width / rect.width),
        (clientY - rect.top) * (canvas.height / rect.height)
      );
      target = [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100];
      menu.querySelector(".place-at").textContent = "Ball placement to (" + target[0].toFixed(2) + ", " + target[1].toFixed(2) + ")";
      menu.style.display = "block";
      openedAt = performance.now();
      menu.style.left = Math.min(clientX, window.innerWidth - menu.offsetWidth - 4) + "px";
      menu.style.top = Math.min(clientY, window.innerHeight - menu.offsetHeight - 4) + "px";
    };
    canvas.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      openAt(e.clientX, e.clientY);
    });
    // iPhone Safari fires no contextmenu on a long press: a held, still touch opens it too.
    let press = null;
    canvas.addEventListener("touchstart", (e) => {
      const t = e.touches[0];
      clearTimeout(press && press.timer);
      press = { x: t.clientX, y: t.clientY, timer: setTimeout(() => openAt(t.clientX, t.clientY), 600) };
    }, { passive: true });
    canvas.addEventListener("touchmove", (e) => {
      const t = e.touches[0];
      if (press && Math.hypot(t.clientX - press.x, t.clientY - press.y) > 10) clearTimeout(press.timer);
    }, { passive: true });
    canvas.addEventListener("touchend", () => press && clearTimeout(press.timer));
    for (const btn of menu.querySelectorAll("button[data-place]")) {
      btn.addEventListener("click", () => {
        hide();
        if (target) send("BALL_PLACEMENT_" + btn.dataset.place, target);
      });
    }
    document.addEventListener("click", (e) => {
      // A click right after opening is the lifted finger of the press that opened it.
      if (performance.now() - openedAt > 400 && !menu.contains(e.target)) hide();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hide();
    });
  }

  // The runner's field checks (utama_core/run/field_check.py): only the problems are listed,
  // failures first; the passing rows fold into one "N ok" line that opens on click (kept open
  // across updates). On real robots any failure has already forced HALT.
  let okOpen = false;
  function rowHtml(r) {
    const mark = r.status === "ok" ? "●" : r.status === "warn" ? "▲" : "✖";
    return (
      '<div class="ref-row"><span class="check-' + r.status + '">' + mark + " " + r.name +
      '</span><span class="muted">' + r.detail + "</span></div>"
    );
  }
  // Shown while the HALT a failing check forced lasts (the runner clears it once play resumes).
  function renderAutoHalt(fc) {
    const banner = document.getElementById("ref-auto-halt");
    const text = fc && fc.auto_halt ? "Auto-HALT " + fc.auto_halt : "";
    banner.style.display = text ? "" : "none";
    if (banner.textContent !== text) banner.textContent = text;
  }
  function renderFieldCheck(fc) {
    const box = document.getElementById("ref-field-check");
    if (!fc) {
      box.innerHTML = "";
      return;
    }
    const fails = fc.rows.filter((r) => r.status === "fail");
    const warns = fc.rows.filter((r) => r.status === "warn");
    const oks = fc.rows.filter((r) => r.status === "ok");
    let html =
      '<div class="ref-section-title">Field check — ' +
      (fails.length ? '<span class="check-fail">' + fails.length + " failing</span>" : '<span class="check-ok">ready</span>') +
      "</div>";
    for (const r of [...fails, ...warns]) html += rowHtml(r);
    html +=
      '<details id="ref-check-ok"' + (okOpen ? " open" : "") + '><summary class="check-ok">' + oks.length +
      " ok</summary>" + oks.map(rowHtml).join("") + "</details>";
    box.innerHTML = html;
    box.querySelector("#ref-check-ok").addEventListener("toggle", (e) => (okOpen = e.target.open));
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
        for (const row of document.querySelectorAll("#view-live .restart-row")) {
          row.style.display = c.restarts ? "" : "none";
        }
        const canvas = document.getElementById("ref-field-canvas");
        fieldView = new FieldCanvas(canvas, c.geometry, {});
        mountPlacementMenu(canvas);
        document.getElementById("ref-profile-name").textContent = c.profile_name;
        Dashboard.showView("live"); // a running match is what this page is open for
      });

    timeline = new Timeline(
      document.getElementById("live-timeline"),
      document.getElementById("live-timeline-legend"),
      null
    );
    document.getElementById("live-timeline-window").addEventListener("change", (e) => {
      timeline.setWindow(Number(e.target.value));
    });

    const es = new EventSource("/events/referee");
    es.addEventListener("referee", (ev) => onEvent(JSON.parse(ev.data)));
    es.onopen = () => {
      streamOpen = true;
      checkStale();
    };
    es.onerror = () => {
      streamOpen = false;
      checkStale();
    };
    setInterval(checkStale, 250);

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

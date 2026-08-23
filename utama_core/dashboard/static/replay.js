// Replay view: pick a recorded .pkl, load it once, then scrub/play entirely
// client-side. Reuses the exact same FieldCanvas renderer and status-panel
// helpers (status_panel.js) as the Live view, so a replay shows the same
// field + robots + tactic/referee panel a live match would.
//
// Tactic assignments and referee state (score/command/stage) are recorded
// sparsely — one event per *change*, not per frame — in a sibling
// `.intentions.jsonl` (see `dashboard/views/replay.py`). Rather than have
// the backend forward-fill those onto every one of tens of thousands of
// frames (cheap on the backend, wasteful over the wire — hundreds of real
// events vs. tens of thousands of redundant copies), this file does the
// same forward-fill client-side, driven by two cursors that advance as the
// viewed frame's `ts` advances. Playback only moves forward, so the common
// case is a cheap cursor bump; scrubbing backward resets both cursors to 0
// and replays forward to the new index — at event counts in the low
// hundreds this is imperceptibly fast, so there's no need for anything
// fancier (e.g. binary search) at this scale.
// Video-player conventions: mm:ss timestamps (not frame counts), a speed
// selector, frame-step buttons, and space/arrow-key transport shortcuts.

(function () {
  let fieldView = null;
  let frames = [];
  let index = 0;
  let playing = false;
  let speed = 1;
  let rafId = null;
  let lastTickMs = 0;
  let carryMs = 0;

  let tacticEvents = [];
  let refereeEvents = [];
  let tacticIdx = 0;
  let refereeIdx = 0;
  let slotRobots = {}; // tactic_id -> currently-assigned robot_ids
  let lastIndex = 0;
  let refereeState = null; // last-seen referee event, or null if none yet

  function frameTs(i) {
    return frames.length ? frames[i].ts : 0;
  }

  function tacticStatusFromSlots() {
    const status = {};
    for (const tacticId in slotRobots) {
      for (const robotId of slotRobots[tacticId]) {
        status[String(robotId)] = [tacticId];
      }
    }
    return status;
  }

  function resetEventCursors() {
    tacticIdx = 0;
    refereeIdx = 0;
    slotRobots = {};
    refereeState = null;
  }

  function advanceEventCursors(ts) {
    while (tacticIdx < tacticEvents.length && tacticEvents[tacticIdx].sim_time <= ts) {
      const e = tacticEvents[tacticIdx];
      slotRobots[e.tactic_id] = e.robot_ids;
      tacticIdx++;
    }
    while (refereeIdx < refereeEvents.length && refereeEvents[refereeIdx].sim_time <= ts) {
      refereeState = refereeEvents[refereeIdx];
      refereeIdx++;
    }
  }

  function fmtTime(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return m + ":" + String(rem).padStart(2, "0");
  }

  function updateReadouts() {
    const label = document.getElementById("replay-frame-label");
    if (!label) return;
    if (!frames.length) {
      label.textContent = "0:00 / 0:00";
      return;
    }
    const elapsed = frameTs(index) - frameTs(0);
    const total = frameTs(frames.length - 1) - frameTs(0);
    label.textContent = fmtTime(elapsed) + " / " + fmtTime(total);
  }

  const REPLAY_HEADER_IDS = {
    yellowScore: "replay-yellow-score",
    blueScore: "replay-blue-score",
    command: "replay-command",
    stage: "replay-stage",
  };

  function renderIntentionLog() {
    const c = document.getElementById("replay-intention-log");
    if (!c) return;
    if (!tacticEvents.length) {
      c.innerHTML = '<div class="muted" style="font-size:.72rem;">no intention log for this replay</div>';
      return;
    }
    const t0 = frameTs(0);
    let html = "";
    for (let i = 0; i < tacticEvents.length; i++) {
      const e = tacticEvents[i];
      const label = e.robot_ids.length ? e.tactic_id + " → " + e.robot_ids.join(",") : e.tactic_id + " (released)";
      html +=
        '<div class="ref-row" style="cursor:pointer;" data-event-idx="' +
        i +
        '"><span class="muted" style="min-width:44px;">' +
        fmtTime(e.sim_time - t0) +
        "</span><span>" +
        label +
        "</span></div>";
    }
    c.innerHTML = html;
  }

  function jumpToEvent(sim_time) {
    // Nearest frame whose ts >= sim_time (falls back to the last frame if
    // the event is at/after the final recorded ts).
    let target = frames.length - 1;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].ts >= sim_time) {
        target = i;
        break;
      }
    }
    stopPlayback();
    setIndex(target);
  }

  function setIndex(i) {
    const newIndex = Math.max(0, Math.min(i, frames.length - 1));
    if (newIndex < lastIndex) resetEventCursors();
    index = newIndex;
    lastIndex = index;
    const slider = document.getElementById("replay-slider");
    if (slider) slider.value = String(index);
    updateReadouts();
    if (!frames.length) return;
    const frame = frames[index];
    if (fieldView) fieldView.draw(frame);
    renderRobotStatusInto("replay-status-entries", frame, { hideFeedback: true });
    advanceEventCursors(frame.ts);
    renderTacticStatusInto("replay-tactic-entries", tacticStatusFromSlots(), {
      emptyLabel: "no tactic data at this frame",
    });
    renderRefereeHeaderInto(REPLAY_HEADER_IDS, refereeState);
  }

  function setPlayButtonLabel() {
    const btn = document.getElementById("replay-play");
    if (btn) btn.textContent = playing ? "Pause" : "Play";
  }

  function stopPlayback() {
    playing = false;
    setPlayButtonLabel();
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  function tick(nowMs) {
    if (!playing) return;
    const dtMs = (nowMs - lastTickMs) * speed;
    lastTickMs = nowMs;
    carryMs += dtMs;

    // Advance by however many frame-intervals fit in the elapsed time,
    // using each frame's own `ts` gap rather than assuming a fixed rate.
    while (index < frames.length - 1) {
      const gapMs = (frameTs(index + 1) - frameTs(index)) * 1000;
      if (gapMs <= 0 || carryMs < gapMs) break;
      carryMs -= gapMs;
      index += 1;
    }
    setIndex(index);

    if (index >= frames.length - 1) {
      stopPlayback();
      return;
    }
    rafId = requestAnimationFrame(tick);
  }

  function startPlayback() {
    if (!frames.length || index >= frames.length - 1) return;
    playing = true;
    setPlayButtonLabel();
    lastTickMs = performance.now();
    carryMs = 0;
    rafId = requestAnimationFrame(tick);
  }

  function togglePlayback() {
    if (playing) stopPlayback();
    else startPlayback();
  }

  function step(delta) {
    stopPlayback();
    setIndex(index + delta);
  }

  function loadList() {
    return fetch("/replay/list")
      .then((r) => r.json())
      .then((paths) => {
        const select = document.getElementById("replay-select");
        const current = select.value;
        select.innerHTML = '<option value="">Select a replay…</option>';
        for (const path of paths) {
          const opt = document.createElement("option");
          opt.value = path;
          opt.textContent = path;
          select.appendChild(opt);
        }
        if (paths.includes(current)) select.value = current;
      });
  }

  const TRANSPORT_CONTROL_IDS = [
    "replay-play",
    "replay-step-back",
    "replay-step-fwd",
    "replay-slider",
    "replay-speed",
  ];

  function setLoading(isLoading) {
    const overlay = document.getElementById("replay-loading");
    if (overlay) overlay.style.display = isLoading ? "" : "none";
    for (const id of TRANSPORT_CONTROL_IDS) {
      const el = document.getElementById(id);
      if (el) el.disabled = isLoading;
    }
  }

  function loadReplay(path) {
    stopPlayback();
    frames = [];
    tacticEvents = [];
    refereeEvents = [];
    lastIndex = 0;
    resetEventCursors();
    setIndex(0);
    setLoading(true);
    fetch("/replay/frames?path=" + encodeURIComponent(path))
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          console.error("replay load error:", data.error);
          return;
        }
        frames = data.frames || [];
        tacticEvents = data.tactic_events || [];
        refereeEvents = data.referee_events || [];
        lastIndex = 0;
        resetEventCursors();
        const canvas = document.getElementById("replay-field-canvas");
        fieldView = new FieldCanvas(canvas, data.geometry, {
          myTeamIsRight: data.my_team_is_right,
          myTeamIsYellow: data.my_team_is_yellow,
        });
        const slider = document.getElementById("replay-slider");
        slider.max = String(Math.max(0, frames.length - 1));
        const tacticBanner = document.getElementById("replay-no-tactics");
        if (tacticBanner) tacticBanner.style.display = data.has_tactic_data ? "none" : "";
        const refBanner = document.getElementById("replay-no-referee");
        if (refBanner) refBanner.style.display = data.has_referee_data ? "none" : "";
        renderIntentionLog();
        setIndex(0);
      })
      .catch((err) => console.error("replay fetch error:", err))
      .finally(() => setLoading(false));
  }

  function loadReplayByPath(path) {
    // Entry point for other views (Tournament's "view replay" links) via
    // Dashboard.showReplay(path) — same load, but also syncs the dropdown.
    // The dropdown's options are only populated once loadList()'s fetch
    // resolves, so the value must be set inside that callback, not after
    // firing it, or `select.value = path` silently no-ops (no matching
    // <option> exists yet).
    loadList().then(() => {
      const select = document.getElementById("replay-select");
      if (select) select.value = path;
    });
    loadReplay(path);
  }

  function mount() {
    loadList();
    Dashboard.registerReplayLoader(loadReplayByPath);

    document.getElementById("replay-select").addEventListener("change", (e) => {
      if (e.target.value) loadReplay(e.target.value);
    });
    document.getElementById("replay-play").addEventListener("click", togglePlayback);
    document.getElementById("replay-step-back").addEventListener("click", () => step(-1));
    document.getElementById("replay-step-fwd").addEventListener("click", () => step(1));
    document.getElementById("replay-slider").addEventListener("input", (e) => {
      stopPlayback();
      setIndex(Number(e.target.value));
    });
    document.getElementById("replay-speed").addEventListener("change", (e) => {
      speed = Number(e.target.value);
    });
    document.getElementById("replay-intention-log").addEventListener("click", (e) => {
      const row = e.target.closest("[data-event-idx]");
      if (!row) return;
      const event = tacticEvents[Number(row.dataset.eventIdx)];
      if (event) jumpToEvent(event.sim_time);
    });

    document.addEventListener("keydown", (e) => {
      if (!document.getElementById("view-replay").classList.contains("active")) return;
      if (e.target.tagName === "SELECT") return;
      if (e.key === " ") {
        e.preventDefault();
        togglePlayback();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(-1);
      }
    });
  }

  function onShow() {
    loadList();
  }

  Dashboard.registerView("replay", { mount, onShow });
})();

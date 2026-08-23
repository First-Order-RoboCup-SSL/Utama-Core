// Tab shell. Each view module (referee.js, live.js, replay.js, tournament.js)
// registers itself via `Dashboard.registerView(name, { mount, onShow })`.
// `mount` runs once; `onShow` runs each time the tab becomes active (views
// that poll on an interval only should start when visible).

const Dashboard = (() => {
  const views = {};
  let active = null;

  function registerView(name, handlers) {
    views[name] = handlers;
  }

  function showView(name) {
    if (active === name) return;
    for (const key of Object.keys(views)) {
      document.getElementById("view-" + key).classList.toggle("active", key === name);
      document.getElementById("tab-" + key).classList.toggle("active", key === name);
    }
    active = name;
    if (views[name] && views[name].onShow) views[name].onShow();
  }

  function init(defaultView) {
    for (const name of Object.keys(views)) {
      document.getElementById("tab-" + name).addEventListener("click", () => showView(name));
      if (views[name].mount) views[name].mount();
    }
    showView(defaultView);
  }

  // Cross-view command: switch to the Replay tab and preload a specific
  // path (used by the Tournament tab's "view replay" links). `replay.js`
  // registers this itself, since only it knows how to load a replay — this
  // is just the addressable hook other views call into.
  let _showReplay = null;
  function registerReplayLoader(fn) {
    _showReplay = fn;
  }
  function showReplay(path) {
    showView("replay");
    if (_showReplay) _showReplay(path);
  }

  return { registerView, showView, init, registerReplayLoader, showReplay };
})();

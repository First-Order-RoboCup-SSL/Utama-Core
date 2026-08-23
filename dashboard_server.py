"""dashboard_server.py — standalone dashboard (Replay + Tournament only).

Run:
    pixi run python dashboard_server.py
    # open http://localhost:8080 in a browser

No live match, no CustomReferee, no StrategyRunner — `attach_dashboard()`
already wires up Replay and Tournament unconditionally (both are stateless
readers of `replays/*/summary.json` and `*.pkl` files on disk), so this
script's only job is to start the HTTP server and keep the process alive.
Live/Referee views need a running match to attach to and so are still
opt-in — see `demo_referee_gui_rsim.py` for a script that starts one.

Use this when you just want to browse recorded replays or tournament
standings without also starting a live rsim match.
"""

import time

from utama_core.dashboard import attach_dashboard

DASHBOARD_PORT = 8080


def main() -> None:
    attach_dashboard(port=DASHBOARD_PORT)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

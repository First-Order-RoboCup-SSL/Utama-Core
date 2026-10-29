"""Teleop view — the Live tab's Teleop panel drives a robot by keyboard (`tactics/teleop.py`).

`/teleop/config` lists the robots the page can drive; `/teleop/drive` takes what the keys ask
for (`TeleopDrive.set`). The robots move only while play runs (Force Start): Halt stops them, and
so does the page going quiet.
"""

from __future__ import annotations

import json

from utama_core.dashboard.server import DashboardServer
from utama_core.tactics.teleop import TeleopDrive


def attach(server: DashboardServer, drive: TeleopDrive) -> None:
    def _handle_drive(payload: dict) -> None:
        drive.set(
            str(payload["robot"]),
            forward=payload.get("forward", 0.0),
            left=payload.get("left", 0.0),
            turn=payload.get("turn", 0.0),
            kick=payload.get("kick", False),
            dribble=payload.get("dribble", False),
        )

    server.add_route("/teleop/config", lambda: json.dumps({"robots": drive.robots}).encode())
    server.add_command_handler("/teleop/drive", _handle_drive)

"""A dashboard whose port is taken (another dashboard already running) must fail at start: it
used to fail in its server thread, and the match ran on with no Live tab while the page open on
that port showed the other dashboard."""

from __future__ import annotations

import socket

import pytest

from utama_core.dashboard.server import DashboardServer


def test_starting_a_dashboard_on_a_taken_port_raises():
    with socket.socket() as taken:
        taken.bind(("", 0))
        taken.listen()
        with pytest.raises(OSError):
            DashboardServer(taken.getsockname()[1]).start()

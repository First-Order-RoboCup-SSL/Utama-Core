from unittest.mock import patch

from utama_core.dashboard.server import DashboardServer


def test_explicit_host_is_used_for_listening():
    server = DashboardServer(8080, host="127.0.0.1")
    with patch("utama_core.dashboard.server.ThreadingHTTPServer") as http_server:
        with patch("threading.Thread.start"):
            server.start()
    assert http_server.call_args.args[0] == ("127.0.0.1", 8080)

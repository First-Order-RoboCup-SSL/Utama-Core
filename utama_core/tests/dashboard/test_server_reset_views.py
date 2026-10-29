"""`DashboardServer.reset_views` drops the routes and commands a match added, and keeps the
dashboard's own: after a demo switch (`examples/demos.py`) the page shows no panel of the demo
that ended, and a click on one never reaches its closed runner."""

from utama_core.dashboard.server import DashboardServer


def test_reset_drops_what_came_after_and_keeps_what_came_before():
    server = DashboardServer(0)
    server.add_route("/tournament/list", lambda: b"[]")
    views = server.views()
    server.add_route("/strategy/config", lambda: b"{}")
    server.add_command_handler("/strategy/run", lambda payload: None)
    server.reset_views(views)
    assert set(server._routes) == {"/tournament/list"} and server._command_handlers == {}

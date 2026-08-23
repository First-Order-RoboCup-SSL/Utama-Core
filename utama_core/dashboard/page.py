"""Assembles the dashboard's single-page shell by inlining `static/*` files.

Kept dependency-free and build-step-free on purpose (same constraint
`custom_referee/gui.py` operated under): one HTML string, served by the
stdlib HTTP server, no bundler, no separate static file route.
"""

from __future__ import annotations

from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"


def build_page() -> str:
    template = (_STATIC_DIR / "index.html").read_text()
    return (
        template.replace("__CSS__", (_STATIC_DIR / "app.css").read_text())
        .replace("__FIELD_CANVAS_JS__", (_STATIC_DIR / "field_canvas.js").read_text())
        .replace("__STATUS_PANEL_JS__", (_STATIC_DIR / "status_panel.js").read_text())
        .replace("__APP_JS__", (_STATIC_DIR / "app.js").read_text())
        .replace("__LIVE_JS__", (_STATIC_DIR / "live.js").read_text())
        .replace("__REPLAY_JS__", (_STATIC_DIR / "replay.js").read_text())
        .replace("__TOURNAMENT_JS__", (_STATIC_DIR / "tournament.js").read_text())
    )

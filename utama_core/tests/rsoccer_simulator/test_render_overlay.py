from unittest.mock import patch

from utama_core.rsoccer_simulator.src.Render.overlay import (
    OverlayObject,
    OverlayType,
    RenderOverlay,
)


def test_line_overlay_draws_from_first_to_last_point():
    """`draw_line`'s own docstring: "Draws a line as an overlay using the
    first and last point in a list of points." Intermediate points are
    accepted but not drawn as separate segments — matches every real caller
    in `ssl_gym_base.py`, none of which relies on multi-segment rendering."""
    overlay = RenderOverlay(
        [
            OverlayObject(
                type=OverlayType.LINE,
                color="RED",
                points=[(1, 2), (3, 4), (5, 6)],
                width=2,
            )
        ],
        scale=100,
    )

    with patch("pygame.draw.line") as draw_line:
        overlay.draw(screen=object())

    draw_line.assert_called_once()
    assert draw_line.call_args.args[2:4] == ((1, 2), (5, 6))


def test_point_overlay_draws_a_filled_circle():
    """`OverlayType.POINT` is the only marker type real callers use for a
    single-location dot — drawn as a filled circle (`width=0`)."""
    overlay = RenderOverlay(
        [
            OverlayObject(
                type=OverlayType.POINT,
                color="RED",
                points=[(10, 20)],
                width=3,
            )
        ],
        scale=100,
    )

    with patch("pygame.draw.circle") as draw_circle:
        overlay.draw(screen=object())

    draw_circle.assert_called_once()
    assert draw_circle.call_args.args[2] == (10, 20)
    assert draw_circle.call_args.args[3] == 3
    assert draw_circle.call_args.kwargs["width"] == 0

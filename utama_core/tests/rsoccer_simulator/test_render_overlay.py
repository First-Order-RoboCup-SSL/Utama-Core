from unittest.mock import patch

from utama_core.rsoccer_simulator.src.Render.overlay import (
    OverlayObject,
    OverlayType,
    RenderOverlay,
)


def test_line_overlay_draws_each_consecutive_segment():
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

    assert draw_line.call_count == 2
    assert draw_line.call_args_list[0].args[2:4] == ((1, 2), (3, 4))
    assert draw_line.call_args_list[1].args[2:4] == ((3, 4), (5, 6))


def test_circle_overlay_draws_circle_outline():
    overlay = RenderOverlay(
        [
            OverlayObject(
                type=OverlayType.CIRCLE,
                color="RED",
                points=[(10, 20)],
                width=3,
                radius=12,
            )
        ],
        scale=100,
    )

    with patch("pygame.draw.circle") as draw_circle:
        overlay.draw(screen=object())

    draw_circle.assert_called_once()
    assert draw_circle.call_args.args[2:5] == ((10, 20), 12, 3)

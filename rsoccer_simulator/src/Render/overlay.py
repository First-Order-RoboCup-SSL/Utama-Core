import pygame
from rsoccer_simulator.src.Render.utils import COLORS
from collections import namedtuple
from enum import Enum


class OverlayType(Enum):
    POINT = 0
    LINE = 1
    POLYGON = 2


OverlayObject = namedtuple("OverlayObject", ["type", "color", "points", "width"])


class RenderOverlay:
    def __init__(self, overlay_data: list[OverlayObject], scale) -> None:
        self.overlay_data = overlay_data
        self.scale = scale

    def draw(self, screen):
        for overlay_obj in self.overlay_data:
            color = overlay_obj.color.upper()
            if overlay_obj.type == OverlayType.POINT:
                pygame.draw.circle(
                    screen,
                    COLORS[color],
                    overlay_obj.points[0],
                    overlay_obj.width,
                    width=0,  # 0 means fill the circle
                )
            elif overlay_obj.type == OverlayType.LINE:
                pygame.draw.line(
                    screen,
                    COLORS[color],
                    overlay_obj.points[0],
                    overlay_obj.points[-1],
                    width=overlay_obj.width,
                )
            else:
                pygame.draw.polygon(
                    screen,
                    COLORS[color],
                    overlay_obj.points,
                    width=overlay_obj.width,
                )

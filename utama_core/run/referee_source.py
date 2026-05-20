"""Referee source discriminated union for StrategyRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from utama_core.custom_referee import CustomReferee


@dataclass(frozen=True)
class OfficialReferee:
    """Sentinel: consume referee commands from the official SSL game-controller over the network."""


RefereeSource = Union[None, OfficialReferee, "CustomReferee"]

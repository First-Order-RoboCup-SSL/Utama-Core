"""Kernel `Strategy` factories — `build_kernel_strategy(motion_controller) -> Strategy`
callables suitable for `AbstractStrategy`'s constructor argument of the same name.

See `utama_core.engine.abstract_strategy.AbstractStrategy` for the class that
consumes these and drives them under `StrategyRunner`.

Each factory lives in its own module, `utama_core/strategy/<name>.py`, next to the pickers only it
uses; helpers shared by more than one strategy are in `pickers.py`. This module only re-exports them, so
`from utama_core.strategy.kernel_strategy import build_X_kernel_strategy` and `dir(kernel_strategy)`
discovery keep working. The private pickers re-exported below are the ones tests import.
"""

# ruff: noqa: F401

from utama_core.strategy.clear_danger import (
    _clear_danger_picker,
    build_clear_danger_kernel_strategy,
)
from utama_core.strategy.clear_press_plus import (
    _clear_press_plus_picker,
    build_clear_press_plus_kernel_strategy,
)
from utama_core.strategy.counter_flow import (
    _counter_flow_picker,
    build_counter_flow_kernel_strategy,
)
from utama_core.strategy.counter_press import (
    _counter_press_picker,
    build_counter_press_kernel_strategy,
)
from utama_core.strategy.decoy_and_overload import (
    build_decoy_and_overload_kernel_strategy,
)
from utama_core.strategy.default import build_default_kernel_strategy
from utama_core.strategy.give_and_go_solo import build_give_and_go_solo_kernel_strategy
from utama_core.strategy.high_line_zone import build_high_line_zone_kernel_strategy
from utama_core.strategy.high_press import build_high_press_kernel_strategy
from utama_core.strategy.low_block import build_low_block_kernel_strategy
from utama_core.strategy.overload_flow import build_overload_flow_kernel_strategy
from utama_core.strategy.overload_press import build_overload_press_kernel_strategy
from utama_core.strategy.pickers import (
    carrier_first,
    fixed_ratio_picker,
    friendly_closer_to_ball,
)
from utama_core.strategy.press_and_pass import build_press_and_pass_kernel_strategy
from utama_core.strategy.press_trigger_flow import (
    build_press_trigger_flow_kernel_strategy,
)
from utama_core.strategy.score_aware_counter_flow import (
    build_score_aware_counter_flow_kernel_strategy,
)
from utama_core.strategy.score_aware_zone_flow import (
    build_score_aware_zone_flow_kernel_strategy,
)
from utama_core.strategy.shadow_switch import build_shadow_switch_kernel_strategy
from utama_core.strategy.split_shape import build_split_shape_kernel_strategy
from utama_core.strategy.switch_of_play import build_switch_of_play_kernel_strategy
from utama_core.strategy.three_slot import (
    _three_way_picker,
    build_three_slot_kernel_strategy,
)
from utama_core.strategy.tiki_taka import (
    _tiki_taka_picker,
    build_tiki_taka_kernel_strategy,
)
from utama_core.strategy.tiki_taka_plus import build_tiki_taka_plus_kernel_strategy
from utama_core.strategy.zone_fluid import (
    _zone_flow_picker,
    build_zone_fluid_kernel_strategy,
)

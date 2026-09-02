"""Common, simulator-independent contract for registered motion controllers."""

import inspect

import pytest

from utama_core.config.enums import Mode
from utama_core.motion_planning.src.common.control_schemes import (
    CONTROL_SCHEME_MAP,
    get_control_scheme,
)
from utama_core.motion_planning.src.common.motion_controller import MotionController

EXPECTED_CALCULATE_PARAMETERS = (
    "self",
    "game",
    "robot_id",
    "target_pos",
    "target_oren",
)
EXPECTED_RESET_PARAMETERS = ("self", "robot_id")


def _parameter_names(method) -> tuple[str, ...]:
    return tuple(inspect.signature(method).parameters)


@pytest.mark.parametrize(
    "scheme_name,controller_class",
    sorted(CONTROL_SCHEME_MAP.items()),
)
def test_registered_controller_implements_common_interface(scheme_name, controller_class):
    assert get_control_scheme(scheme_name) is controller_class
    assert get_control_scheme(scheme_name.upper()) is controller_class
    assert issubclass(controller_class, MotionController)
    assert not inspect.isabstract(controller_class)
    assert _parameter_names(controller_class.calculate) == EXPECTED_CALCULATE_PARAMETERS
    assert _parameter_names(controller_class.reset) == EXPECTED_RESET_PARAMETERS


@pytest.mark.parametrize("controller_class", CONTROL_SCHEME_MAP.values())
def test_registered_controller_supports_common_lifecycle(controller_class):
    controller = controller_class(Mode.RSIM, None)

    assert controller.mode is Mode.RSIM
    assert controller.rsim_env is None
    assert controller.reset(0) is None


def test_unknown_control_scheme_is_rejected():
    with pytest.raises(ValueError, match="Unknown control scheme: missing"):
        get_control_scheme("missing")

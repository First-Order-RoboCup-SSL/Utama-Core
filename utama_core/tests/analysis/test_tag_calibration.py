"""`fit_offset` recovers where a tag sits on a robot turning on the spot, from its track alone."""

import numpy as np
import pytest

from utama_core.analysis.tag_calibration import fit_offset


@pytest.mark.parametrize("forward, left", [(-0.011, -0.004), (0.02, 0.0), (0.0, 0.0)])
def test_a_spin_gives_the_tags_offset_from_the_centre(forward, left):
    """Blue 0's tag measured 1.1 cm behind and 0.4 cm right of its centre (2026-10-28); the
    centre drifts 1 cm/s, as a real spin does, and vision adds 1 mm of noise."""
    t = np.linspace(0.0, 4.0, 240)
    heading = 2.0 * t
    cx, cy = 0.3 + 0.01 * t, -0.2 + 0.005 * t
    rng = np.random.default_rng(0)
    x = cx + forward * np.cos(heading) - left * np.sin(heading) + rng.normal(0, 0.001, t.size)
    y = cy + forward * np.sin(heading) + left * np.cos(heading) + rng.normal(0, 0.001, t.size)

    got_forward, got_left, rms = fit_offset(t, x, y, heading)

    assert (got_forward, got_left) == pytest.approx((forward, left), abs=0.001)
    assert rms < 0.002

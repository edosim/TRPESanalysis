import numpy as np
import pytest

from conversions import (
    MASS_ELECTRON,
    LIGHTSPEED,
    tof_to_eke,
    pos_to_time,
)


# ---------- tof_to_eke ----------

def test_eke_at_light_speed_gives_half_rest_mass():
    # If v == c, the formula gives 0.5 * m c^2 (the non-relativistic result)
    s, t, t0, E0 = LIGHTSPEED, 1.0, 0.0, 0.0   # s/(t-t0) = c
    assert tof_to_eke(t, s, t0, E0) == pytest.approx(0.5 * MASS_ELECTRON)


def test_E0_is_a_pure_offset():
    base = tof_to_eke(100.0, 0.5, 10.0, 0.0)
    shifted = tof_to_eke(100.0, 0.5, 10.0, 2.5)
    assert shifted == pytest.approx(base - 2.5)


def test_t0_shifts_the_time_axis():
    # Only (t - t0) matters
    a = tof_to_eke(100.0, 0.5, 10.0, 0.0)
    b = tof_to_eke(150.0, 0.5, 60.0, 0.0)
    assert a == pytest.approx(b)


def test_doubling_flight_time_quarters_energy():
    e1 = tof_to_eke(50.0, 0.5, 0.0, 0.0)
    e2 = tof_to_eke(100.0, 0.5, 0.0, 0.0)
    assert e2 == pytest.approx(e1 / 4)


def test_energy_decreases_with_longer_tof():
    t = np.array([20.0, 40.0, 80.0])
    e = tof_to_eke(t, 0.5, 0.0, 0.0)
    assert np.all(np.diff(e) < 0)


def test_works_elementwise_on_arrays():
    t = np.array([20.0, 40.0, 80.0])
    result = tof_to_eke(t, 0.5, 0.0, 0.0)
    expected = [tof_to_eke(float(ti), 0.5, 0.0, 0.0) for ti in t]
    assert result == pytest.approx(expected)


def test_t_equals_t0_is_singular():
    # Decide what you WANT to happen here, then lock it in.
    # With plain floats this raises; with numpy it gives inf + a warning.
    with pytest.raises(ZeroDivisionError):
        tof_to_eke(10.0, 0.5, 10.0, 0.0)


# ---------- pos_to_time ----------

def test_zero_delay_at_pos0():
    assert pos_to_time(5.0, 5.0) == pytest.approx(0.0)


def test_one_mm_is_about_6_67_ps():
    # Round trip: 2 mm of path per 1 mm of stage travel
    assert pos_to_time(1.0, 0.0) == pytest.approx(6.6712819, rel=1e-6)


def test_antisymmetric_about_pos0():
    assert pos_to_time(3.0, 5.0) == pytest.approx(-pos_to_time(7.0, 5.0))


def test_linear_in_position():
    d = np.array([0.0, 1.0, 2.0, 3.0])
    t = pos_to_time(d, 0.0)
    assert np.allclose(np.diff(t), np.diff(t)[0])


def test_array_input_returns_array():
    out = pos_to_time(np.array([0.0, 1.0, 2.0]), 0.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
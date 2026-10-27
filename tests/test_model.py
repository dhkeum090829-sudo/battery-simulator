from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from battery_model import (
    Config,
    METHODS,
    PID,
    simulate,
    diffusion_step,
    field_transport,
    impedance_spectrum,
)


@pytest.fixture
def small():
    return Config(max_time_s=120.0, dt=0.5, sensor_noise_v=0.0)


@pytest.mark.parametrize("method", list(METHODS))
def test_charge_conservation_and_limits(small, method):
    r = simulate(small, method)
    d = r.data
    assert np.isfinite(d.to_numpy()).all()
    assert d.charge_residual_ah.abs().max() < 1e-10
    assert d.salt_residual_mol_m2.abs().max() < 1e-12
    assert d.voltage_v.max() <= small.max_voltage + 1e-6
    assert (d.current_a >= 0).all()
    assert np.all(np.diff(d.time_s) > 0)
    assert np.all(np.diff(d.plated_ah) >= -1e-15)
    assert (r.profiles > 0).all()
    assert d.soc.iloc[-1] <= small.target_soc + 1e-8


def test_zero_field_removes_field_effect(small):
    c = replace(small, field_t=0.0)
    pd.testing.assert_frame_equal(
        simulate(c, "combined").data, simulate(c, "adaptive").data
    )
    pd.testing.assert_frame_equal(simulate(c, "field").data, simulate(c, "cc").data)


def test_zero_gain_has_only_actuator_cost(small):
    c = replace(small, dispersion_gain=0.0, coil_heat_fraction=0.0)
    a, b = simulate(c, "cc").data, simulate(c, "field").data
    for key in ["soc", "plated_ah", "temperature_c", "surface_mol_m3", "current_a"]:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)
    assert b.coil_energy_wh.iloc[-1] > 0


def test_parallel_field_has_no_transport_increment(small):
    c = replace(small, angle_deg=0.0)
    f, u, d = field_transport(c, 15.0, 1.0, 25.0)
    assert f == u == 0.0
    assert d == pytest.approx(c.diffusion_m2_s)


def test_rest_relaxes_concentration(small):
    c = small
    profile = np.linspace(400.0, 990.0, c.nodes)
    after, residual = diffusion_step(profile, 20.0, c.diffusion_m2_s, 0.0, c)
    assert after[0] > profile[0]
    assert abs(residual) < 1e-12


def test_identical_inserted_charge_target():
    c = Config(target_soc=0.15, max_time_s=1800.0, sensor_noise_v=0.0)
    for m in METHODS:
        r = simulate(c, m)
        assert r.status == "목표 SOC 도달"
        assert r.data.soc.iloc[-1] == pytest.approx(0.15, abs=1e-9)
        assert r.data.input_ah.iloc[-1] - r.data.plated_ah.iloc[-1] == pytest.approx(
            0.25, abs=1e-9
        )


def test_timeout_not_claimed_as_completion(small):
    r = simulate(small)
    assert r.status == "시간 제한"
    assert not r.summary()["목표 도달"]


def test_pid_anti_windup(small):
    pid = PID(small)
    for _ in range(100):
        assert small.min_duty <= pid.update(100.0, 20.0) <= small.max_duty
    assert abs(pid.integral) < 1e-12
    assert pid.update(0.0, 20.0) > small.min_duty


def test_delay_is_causal_and_seed_reproducible(small):
    c = replace(small, sensor_delay_s=200.0, sensor_noise_v=0.001)
    a, b = simulate(c), simulate(c)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert (a.data.feedback_stress == 0).all()
    assert (a.data.dvdt_v_s == 0).all()


def test_synthetic_impedance_is_finite(small):
    z = impedance_spectrum(small, 0.5, 25.0, 500.0)
    assert np.isfinite(z.to_numpy()).all()
    assert (z.real_ohm > 0).all()


def test_diffusion_time_and_grid_convergence(small):
    a = simulate(small, "fixed").data.iloc[-1]
    b = simulate(replace(small, dt=0.25), "fixed").data.iloc[-1]
    c = simulate(replace(small, nodes=48), "fixed").data.iloc[-1]
    assert abs(a.surface_mol_m3 - b.surface_mol_m3) < 3.0
    assert abs(a.surface_mol_m3 - c.surface_mol_m3) < 10.0
    assert abs(a.temperature_c - b.temperature_c) < 0.05


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_soc": 0.9},
        {"dt": 10.0},
        {"nodes": 2},
        {"field_t": -1.0},
        {"c_rate": float("nan")},
        {"min_duty": 0.8},
        {"max_temp_c": 20.0},
    ],
)
def test_invalid_inputs_rejected(changes):
    with pytest.raises(ValueError):
        simulate(replace(Config(), **changes))


def test_sensor_clock_independent_of_solver_step(small):
    c = replace(small, sensor_noise_v=0.0005)
    a = simulate(c, "combined").data
    changes = a.dvdt_v_s.diff().fillna(0).abs() > 1e-12
    times = a.loc[changes, "time_s"].to_numpy()
    np.testing.assert_allclose(times, np.round(times), atol=1e-8)
    b = simulate(replace(c, dt=0.25), "combined").data
    assert abs(a.soc.iloc[-1] - b.soc.iloc[-1]) < 0.003

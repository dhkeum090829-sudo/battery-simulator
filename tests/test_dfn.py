import os

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import pytest

pytest.importorskip("pybamm")
from battery_model import Config
from dfn_validation import run_comparison


def test_dfn_same_host_soc_and_limits():
    c = Config(target_soc=0.8, max_time_s=4500.0)
    frames, table = run_comparison(c)
    assert table["목표 도달"].all()
    for frame in frames.values():
        assert frame.host_soc.iloc[-1] == pytest.approx(0.8, abs=2e-5)
        assert frame.voltage_v.max() <= c.max_voltage + 1e-4
        assert frame.current_a.max() <= c.c_rate * 5.1
        assert frame.cumulative_deposition_ah.iloc[-1] >= 0
        assert frame.plated_inventory_ah.max() >= 0

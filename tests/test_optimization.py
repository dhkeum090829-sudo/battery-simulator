import io
import json
import zipfile
from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from battery_model import Config
from optimization import (
    SearchSpace,
    Limits,
    SearchReport,
    generate_candidates,
    pareto_mask,
    assess,
    search,
    recommend,
    stress_scenarios,
    validate_candidate,
    export_search,
)


def baseline():
    return dict(
        reached=True,
        elapsed_min=10.0,
        plating_mah=0.2,
        energy_wh=10.0,
        max_temp_c=30.0,
        max_voltage_v=4.0,
        charge_error_ah=0.0,
        salt_error_mol_m2=0.0,
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"reached": False}, "미도달"),
        ({"elapsed_min": 21.0}, "시간"),
        ({"max_temp_c": 46.0}, "온도"),
        ({"max_voltage_v": 4.3}, "전압"),
        ({"plating_mah": 0.21}, "석출"),
        ({"energy_wh": 11.0}, "에너지"),
        ({"charge_error_ah": 1e-4}, "수지"),
        ({"plating_mah": float("nan")}, "유효"),
        ({"energy_wh": -1.0}, "유효"),
    ],
)
def test_constraints_reject_unusable_candidate(change, reason):
    row = {**baseline(), **change}
    ok, message = assess(row, baseline(), Limits())
    assert not ok and reason in message


def test_reference_incomplete_cannot_create_relative_recommendation():
    ok, message = assess(baseline(), {**baseline(), "reached": False}, Limits())
    assert not ok and "기준" in message


def test_zero_reference_plating_is_not_divided():
    ref = {**baseline(), "plating_mah": 0.0}
    assert assess(ref, ref, Limits())[0]
    assert not assess({**ref, "plating_mah": 0.001}, ref, Limits())[0]


def test_pareto_keeps_tradeoffs_and_ties_but_not_nonfinite_or_dominated():
    values = [[1, 3, 2], [2, 2, 1], [3, 4, 3], [1, 3, 2], [np.nan, 0, 0]]
    np.testing.assert_array_equal(pareto_mask(values), [True, True, False, True, False])
    assert pareto_mask(np.empty((0, 3))).size == 0


def test_pairing_reproducibility_and_requested_bounds():
    base = Config(field_t=0.8)
    space = SearchSpace(
        field_min=0.1,
        field_max=0.2,
        period_min=12.0,
        period_max=16.0,
        duty_min=0.4,
        duty_max=0.6,
        kp_max=0.2,
        pairs=4,
    )
    a = generate_candidates(base, space)
    assert a == generate_candidates(base, space)
    assert a != generate_candidates(base, replace(space, seed=999))
    for candidate in a:
        if candidate.candidate_id == "ref-cc":
            continue
        c = candidate.config
        assert 12 <= c.pulse_period_s <= 16 and 0.4 <= c.initial_duty <= 0.6
        assert c.kp <= 0.2
        if candidate.method in ("combined", "field"):
            assert 0.1 <= c.field_t <= 0.2
    for i in range(space.pairs):
        off, on = a[5 + 2 * i : 7 + 2 * i]
        x, y = asdict(off.config), asdict(on.config)
        assert x.pop("field_t") == 0
        y.pop("field_t")
        assert x == y


def test_invalid_space_and_limits_fail_early():
    with pytest.raises(ValueError):
        generate_candidates(Config(), SearchSpace(period_min=1.0))
    with pytest.raises(ValueError):
        generate_candidates(Config(), SearchSpace(field_min=1.0, field_max=0.5))
    with pytest.raises(ValueError):
        generate_candidates(Config(), SearchSpace(field_max=0.8))
    with pytest.raises(ValueError):
        generate_candidates(Config(), SearchSpace(seed=-1))
    with pytest.raises(ValueError):
        search(limits=Limits(deadline_min=0.0))


def test_priority_is_explicit_and_never_picks_infeasible_candidate():
    table = pd.DataFrame(
        [
            dict(
                candidate_id="A",
                feasible=True,
                pareto=True,
                plating_mah=0.1,
                elapsed_min=15,
                energy_wh=12,
            ),
            dict(
                candidate_id="B",
                feasible=True,
                pareto=True,
                plating_mah=0.2,
                elapsed_min=10,
                energy_wh=11,
            ),
            dict(
                candidate_id="C",
                feasible=True,
                pareto=True,
                plating_mah=0.3,
                elapsed_min=12,
                energy_wh=10,
            ),
            dict(
                candidate_id="bad",
                feasible=False,
                pareto=False,
                plating_mah=0,
                elapsed_min=1,
                energy_wh=1,
            ),
        ]
    )
    report = SearchReport(Config(), SearchSpace(), Limits(), table, {}, baseline())
    assert recommend(report, "plating") == "A"
    assert recommend(report, "time") == "B"
    assert recommend(report, "energy") == "C"
    report.table["feasible"] = False
    assert recommend(report) is None


@pytest.fixture
def real_report():
    return search(
        Config(target_soc=0.12, max_time_s=180.0, sensor_noise_v=0.0),
        SearchSpace(pairs=2),
        Limits(deadline_min=3.0),
    )


def test_real_search_target_and_export_are_consistent(real_report):
    report = real_report
    assert len(report.table) == 9
    assert report.table.loc[report.table.feasible, "reached"].all()
    chosen = recommend(report)
    assert chosen is not None
    assert report.table.set_index("candidate_id").loc[chosen, "pareto"]
    with zipfile.ZipFile(io.BytesIO(export_search(report))) as archive:
        assert archive.testzip() is None
        metadata = json.loads(archive.read("metadata.json"))
        selected = json.loads(archive.read("recommended_control.json"))
    assert metadata["recommended_candidate"] == chosen == selected["candidate_id"]
    assert selected["config"] == asdict(report.candidates[chosen].config)
    assert metadata["validated_candidate"] is None


def test_impossible_limits_and_partial_runs_do_not_get_recommended():
    report = search(
        Config(target_soc=0.12, max_time_s=180.0),
        SearchSpace(pairs=1),
        Limits(deadline_min=0.01),
    )
    assert not report.table.feasible.any() and recommend(report) is None
    report = search(Config(max_time_s=1.0), SearchSpace(pairs=1), Limits())
    assert not report.baseline["reached"] and recommend(report) is None


def test_stress_protocol_includes_zero_field_effect_and_resolution():
    cases = stress_scenarios(Config())
    assert len(cases) == 30 and len({name for name, c in cases}) == 30
    assert any(c.dispersion_gain == 0.0 for name, c in cases)
    assert any(c.ambient_c == 20.0 for name, c in cases)
    assert any(c.nodes == 48 for name, c in cases)
    assert any(c.dt == 0.25 for name, c in cases)


def test_stress_checks_pair_environment_and_preserve_chosen_controls(real_report):
    calls = []

    def fake(c, method):
        calls.append((c, method))
        d = pd.DataFrame(
            dict(
                soc=[c.initial_soc, c.target_soc],
                time_s=[0.0, 60.0],
                plated_ah=[0.0, 0.00001],
                cell_energy_wh=[0.0, 1.0],
                coil_energy_wh=[0.0, 0.0],
                temperature_c=[c.initial_c, c.initial_c + 1],
                voltage_v=[3.6, 4.0],
                charge_residual_ah=[0.0, 0.0],
                salt_residual_mol_m2=[0.0, 0.0],
                duty=[0.6, 0.6],
            )
        )
        return SimpleNamespace(data=d, method=method, config=c, status="목표 SOC 도달")

    candidate = "p001-on"
    env = replace(
        real_report.base,
        ambient_c=18.0,
        initial_c=18.0,
        dispersion_gain=0.0,
        sensor_noise_v=0.001,
        seed=123,
        sensor_delay_s=4.0,
    )
    checks = validate_candidate(
        real_report, candidate, simulator=fake, scenarios=[("test", env)]
    )
    assert checks.feasible.all() and len(calls) == 2
    first, second = calls[0][0], calls[1][0]
    for key in (
        "ambient_c",
        "initial_c",
        "dispersion_gain",
        "sensor_noise_v",
        "seed",
        "sensor_delay_s",
    ):
        assert getattr(first, key) == getattr(second, key) == getattr(env, key)
    assert second.kp == real_report.candidates[candidate].config.kp
    assert second.field_t == real_report.candidates[candidate].config.field_t
    bad = replace(env, initial_c=50.0)
    checks = validate_candidate(
        real_report, candidate, simulator=fake, scenarios=[("invalid", bad)]
    )
    assert not checks.feasible.any()

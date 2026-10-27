"""Finite candidate search and paired stress checks for the uncalibrated model."""

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
import io
import json
import zipfile
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from battery_model import Config, METHODS, simulate

OBJECTIVES = ("plating_mah", "elapsed_min", "energy_wh")
PRIORITIES = {
    "plating": ("plating_mah", "elapsed_min", "energy_wh"),
    "time": ("elapsed_min", "plating_mah", "energy_wh"),
    "energy": ("energy_wh", "plating_mah", "elapsed_min"),
}


@dataclass(frozen=True)
class SearchSpace:
    field_min: float = 0.0
    field_max: float = 0.5
    period_min: float = 10.0
    period_max: float = 40.0
    duty_min: float = 0.35
    duty_max: float = 0.9
    kp_min: float = 0.05
    kp_max: float = 0.7
    ki_min: float = 0.0
    ki_max: float = 0.01
    kd_min: float = 0.0
    kd_max: float = 0.5
    pairs: int = 16
    seed: int = 2103

    def validate(self, base):
        if not all(math.isfinite(float(x)) for x in asdict(self).values()):
            raise ValueError("탐색 범위는 유한한 수여야 합니다.")
        if not isinstance(self.pairs, int) or not 1 <= self.pairs <= 64:
            raise ValueError("탐색 표본 쌍은 1~64개여야 합니다.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("탐색 시드는 0 이상의 정수여야 합니다.")
        if self.field_max > base.field_t + 1e-9:
            raise ValueError(
                "탐색 자기장 상한은 공통 조건의 자기장 상한 이하여야 합니다."
            )
        for name in ("field", "period", "duty", "kp", "ki", "kd"):
            low, high = getattr(self, name + "_min"), getattr(self, name + "_max")
            if low < 0 or high < low:
                raise ValueError(
                    f"{name}: 최솟값 ≤ 최댓값, 음수 제외 조건이 필요합니다."
                )
        if self.period_min < 3 * base.dt / base.min_duty - 1e-9:
            raise ValueError("탐색 최소 주기를 늘리거나 공통 시간 간격을 줄여 주세요.")
        if not base.min_duty <= self.duty_min <= self.duty_max <= base.max_duty:
            raise ValueError("탐색 듀티비가 제어기의 듀티비 한계를 벗어났습니다.")


@dataclass(frozen=True)
class Limits:
    deadline_min: float = 20.0
    energy_ratio: float = 1.05
    plating_ratio: float = 1.0
    max_temp_c: float = 45.0
    max_voltage: float = 4.2

    def validate(self):
        if not all(math.isfinite(float(x)) for x in asdict(self).values()):
            raise ValueError("제약 조건은 유한한 수여야 합니다.")
        if self.deadline_min <= 0 or self.energy_ratio <= 0 or self.plating_ratio < 0:
            raise ValueError(
                "시간·에너지 제한은 양수, 석출 비율은 0 이상이어야 합니다."
            )
        if self.max_voltage <= 0 or self.max_temp_c <= -273.15:
            raise ValueError("전압·온도 제한을 확인해 주세요.")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    method: str
    config: Config


@dataclass
class SearchReport:
    base: Config
    space: SearchSpace
    limits: Limits
    table: pd.DataFrame
    candidates: dict
    baseline: dict

    def metadata(self):
        return {
            "base": asdict(self.base),
            "space": asdict(self.space),
            "limits": asdict(self.limits),
            "baseline": self.baseline,
            "algorithm": "seeded Latin-hypercube pairs plus five reference anchors",
            "objectives": list(OBJECTIVES),
            "candidates": {
                k: {"method": v.method, "config": asdict(v.config)}
                for k, v in self.candidates.items()
            },
            "scope": "sampled candidates only; not global optimum or real-cell safety validation",
        }


def generate_candidates(base, space):
    base.validate()
    space.validate(base)
    anchor = replace(
        base,
        field_t=float(np.clip(base.field_t, space.field_min, space.field_max)),
        pulse_period_s=float(
            np.clip(base.pulse_period_s, space.period_min, space.period_max)
        ),
        initial_duty=float(np.clip(base.initial_duty, space.duty_min, space.duty_max)),
        kp=float(np.clip(base.kp, space.kp_min, space.kp_max)),
        ki=float(np.clip(base.ki, space.ki_min, space.ki_max)),
        kd=float(np.clip(base.kd, space.kd_min, space.kd_max)),
    )
    candidates = [
        Candidate("ref-" + m, m, base if m == "cc" else anchor) for m in METHODS
    ]
    rng = np.random.default_rng(space.seed)
    names = ("field", "period", "duty", "kp", "ki", "kd")
    samples = {}
    for name in names:
        unit = (rng.permutation(space.pairs) + rng.random(space.pairs)) / space.pairs
        low, high = getattr(space, name + "_min"), getattr(space, name + "_max")
        samples[name] = low + (high - low) * unit
    for i in range(space.pairs):
        c = replace(
            base,
            field_t=float(samples["field"][i]),
            pulse_period_s=float(samples["period"][i]),
            initial_duty=float(samples["duty"][i]),
            kp=float(samples["kp"][i]),
            ki=float(samples["ki"][i]),
            kd=float(samples["kd"][i]),
        )
        c.validate()
        # Matched control parameters let the two field settings be compared.
        candidates.extend(
            [
                Candidate(f"p{i+1:03}-off", "adaptive", replace(c, field_t=0.0)),
                Candidate(f"p{i+1:03}-on", "combined", c),
            ]
        )
    return candidates


def metrics(result):
    d = result.data
    last = d.iloc[-1]
    return {
        "method": result.method,
        "method_label": METHODS[result.method],
        "status": result.status,
        "reached": bool(last.soc >= result.config.target_soc - 1e-8),
        "elapsed_min": float(last.time_s / 60),
        "final_soc_pct": float(last.soc * 100),
        "plating_mah": float(last.plated_ah * 1000),
        "max_temp_c": float(d.temperature_c.max()),
        "max_voltage_v": float(d.voltage_v.max()),
        "energy_wh": float(last.cell_energy_wh + last.coil_energy_wh),
        "coil_energy_wh": float(last.coil_energy_wh),
        "charge_error_ah": float(d.charge_residual_ah.abs().max()),
        "salt_error_mol_m2": float(d.salt_residual_mol_m2.abs().max()),
        "duty_total_variation": float(d.duty.diff().abs().sum()),
    }


def assess(row, baseline, limits):
    """All limits are research choices, never certified safety thresholds."""
    reasons = []
    values = [row.get(k, np.nan) for k in (*OBJECTIVES, "max_temp_c", "max_voltage_v")]
    if (
        not all(math.isfinite(float(v)) for v in values)
        or min(row.get(k, 0.0) for k in OBJECTIVES) < 0
    ):
        return False, "계산 실패 또는 유효하지 않은 결과"
    if not row["reached"]:
        reasons.append("목표 SOC 미도달")
    if row["elapsed_min"] > limits.deadline_min + 1e-7:
        reasons.append("충전 시간 초과")
    if row["max_temp_c"] > limits.max_temp_c + 1e-6:
        reasons.append("온도 제한 초과")
    if row["max_voltage_v"] > limits.max_voltage + 1e-6:
        reasons.append("전압 제한 초과")
    if row["charge_error_ah"] > 1e-8 or row["salt_error_mol_m2"] > 1e-9:
        reasons.append("수지 검증 실패")
    if not baseline["reached"] or not all(
        math.isfinite(baseline[k]) and baseline[k] >= 0 for k in OBJECTIVES
    ):
        reasons.append("기준 CC–CV 미도달: 상대 제한 비교 불가")
    else:
        # Zero reference plating uses an absolute numerical tolerance, no division.
        if row["plating_mah"] > baseline["plating_mah"] * limits.plating_ratio + 1e-9:
            reasons.append("석출량 제한 초과")
        if row["energy_wh"] > baseline["energy_wh"] * limits.energy_ratio + 1e-9:
            reasons.append("입력 에너지 제한 초과")
    return not reasons, " · ".join(reasons) if reasons else "모든 연구 제약 통과"


def pareto_mask(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("목적함수 행렬은 2차원이어야 합니다.")
    keep = np.zeros(len(values), dtype=bool)
    finite = np.isfinite(values).all(axis=1)
    for i in np.flatnonzero(finite):
        dominates = (
            finite
            & (values <= values[i]).all(axis=1)
            & (values < values[i]).any(axis=1)
        )
        keep[i] = not dominates.any()
    return keep


def search(
    base=Config(),
    space=SearchSpace(),
    limits=Limits(),
    progress=None,
    simulator=simulate,
):
    limits.validate()
    candidates = generate_candidates(base, space)
    baseline = metrics(simulator(base, "cc"))
    rows = []
    for i, candidate in enumerate(candidates):
        result_metrics = (
            baseline.copy()
            if candidate.candidate_id == "ref-cc"
            else metrics(simulator(candidate.config, candidate.method))
        )
        feasible, reason = assess(result_metrics, baseline, limits)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                **result_metrics,
                "field_limit_t": (
                    candidate.config.field_t
                    if candidate.method in ("field", "combined")
                    else 0.0
                ),
                "period_s": candidate.config.pulse_period_s,
                "initial_duty": candidate.config.initial_duty,
                "kp": candidate.config.kp,
                "ki": candidate.config.ki,
                "kd": candidate.config.kd,
                "feasible": feasible,
                "reason": reason,
                "pareto": False,
            }
        )
        if progress:
            progress(i + 1, len(candidates), candidate.candidate_id)
    table = pd.DataFrame(rows)
    eligible = table.index[table.feasible]
    table.loc[eligible, "pareto"] = pareto_mask(
        table.loc[eligible, list(OBJECTIVES)].to_numpy()
    )
    return SearchReport(
        base, space, limits, table, {c.candidate_id: c for c in candidates}, baseline
    )


def recommend(report, priority="plating"):
    if priority not in PRIORITIES:
        raise ValueError("선택 기준을 확인해 주세요.")
    eligible = report.table[report.table.feasible & report.table.pareto]
    if eligible.empty:
        return None
    return str(
        eligible.sort_values([*PRIORITIES[priority], "candidate_id"], kind="stable")
        .iloc[0]
        .candidate_id
    )


def stress_scenarios(base):
    """27 paired stress conditions + nominal and two numerical-resolution checks."""
    rows = [("nominal", base)]
    for offset, gain, seed in product((-5.0, 0.0, 5.0), (0.0, 1.0, 2.0), range(3)):
        conf = replace(
            base,
            initial_c=base.initial_c + offset,
            ambient_c=base.ambient_c + offset,
            dispersion_gain=base.dispersion_gain * gain,
            seed=base.seed + seed,
            sensor_noise_v=max(base.sensor_noise_v, 0.001),
            sensor_delay_s=base.sensor_delay_s + 2.0,
        )
        rows.append((f"T{offset:+g}_alpha{gain:g}_seed{seed}", conf))
    rows.extend(
        [
            ("dt_half", replace(base, dt=base.dt / 2)),
            ("nodes_double", replace(base, nodes=base.nodes * 2)),
        ]
    )
    return rows


def validate_candidate(
    report, candidate_id, progress=None, simulator=simulate, scenarios=None
):
    if candidate_id not in report.candidates:
        raise ValueError("존재하지 않는 후보입니다.")
    candidate = report.candidates[candidate_id]
    # Scenarios change environmental/numerical variables only, preserving all controls.
    scenarios = stress_scenarios(report.base) if scenarios is None else scenarios
    rows = []
    environment_keys = (
        "initial_c",
        "ambient_c",
        "dispersion_gain",
        "seed",
        "sensor_noise_v",
        "sensor_delay_s",
        "dt",
        "nodes",
    )
    for i, (label, reference_config) in enumerate(scenarios):
        overrides = {k: getattr(reference_config, k) for k in environment_keys}
        candidate_config = replace(candidate.config, **overrides)
        try:
            reference_config.validate()
            candidate_config.validate()
            baseline = metrics(simulator(reference_config, "cc"))
            chosen = metrics(simulator(candidate_config, candidate.method))
            feasible, reason = assess(chosen, baseline, report.limits)
            ratio = (
                chosen["plating_mah"] / baseline["plating_mah"]
                if baseline["reached"]
                and baseline["plating_mah"] > 1e-9
                and chosen["reached"]
                else np.nan
            )
            energy_ratio = (
                chosen["energy_wh"] / baseline["energy_wh"]
                if baseline["reached"]
                and baseline["energy_wh"] > 0
                and chosen["reached"]
                else np.nan
            )
            rows.append(
                {
                    "scenario": label,
                    "candidate_id": candidate_id,
                    **chosen,
                    "reference_reached": baseline["reached"],
                    "reference_plating_mah": baseline["plating_mah"],
                    "reference_energy_wh": baseline["energy_wh"],
                    "plating_ratio": ratio,
                    "energy_ratio": energy_ratio,
                    "feasible": feasible,
                    "reason": reason,
                    **{k: getattr(reference_config, k) for k in environment_keys},
                }
            )
        except (ValueError, ArithmeticError) as exc:
            rows.append(
                {
                    "scenario": label,
                    "candidate_id": candidate_id,
                    "reached": False,
                    "reference_reached": False,
                    "feasible": False,
                    "reason": str(exc),
                    **{k: getattr(reference_config, k) for k in environment_keys},
                }
            )
        if progress:
            progress(i + 1, len(scenarios), label)
    return pd.DataFrame(rows)


def export_search(
    report, priority="plating", validation=None, validated_candidate=None
):
    metadata = report.metadata()
    chosen = recommend(report, priority)
    metadata.update(
        {
            "priority": priority,
            "recommended_candidate": chosen,
            "validated_candidate": (
                validated_candidate if validation is not None else None
            ),
            "reference_energy_limit_wh": report.baseline["energy_wh"]
            * report.limits.energy_ratio,
            "reference_plating_limit_mah": report.baseline["plating_mah"]
            * report.limits.plating_ratio,
        }
    )
    root = Path(__file__).resolve().parent
    metadata["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("battery_model.py", "optimization.py", "optimization_ui.py")
    }
    packet = io.BytesIO()
    with zipfile.ZipFile(packet, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "search.csv", report.table.to_csv(index=False).encode("utf-8-sig")
        )
        archive.writestr(
            "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        if chosen is not None:
            candidate = report.candidates[chosen]
            archive.writestr(
                "recommended_control.json",
                json.dumps(
                    {
                        "candidate_id": chosen,
                        "method": candidate.method,
                        "config": asdict(candidate.config),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        if validation is not None:
            archive.writestr(
                "validation.csv", validation.to_csv(index=False).encode("utf-8-sig")
            )
        archive.writestr(
            "MODEL_NOTES.md", (root / "MODEL_NOTES.md").read_text(encoding="utf-8")
        )
    return packet.getvalue()

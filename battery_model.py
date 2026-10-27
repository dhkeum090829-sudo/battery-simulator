"""Uncalibrated reduced-order model. SI units; no dendrite/fire certification."""

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

F, R = 96485.33212, 8.314462618
METHODS = {
    "cc": "일반 CC–CV",
    "field": "자기장 + CC–CV",
    "fixed": "고정 펄스",
    "adaptive": "적응형 펄스",
    "combined": "자기장 + 적응형 펄스",
}


@dataclass(frozen=True)
class Config:
    capacity_ah: float = 5.0
    area_m2: float = 0.08
    initial_soc: float = 0.1
    target_soc: float = 0.8
    c_rate: float = 3.0
    ambient_c: float = 25.0
    initial_c: float = 25.0
    max_temp_c: float = 45.0
    max_voltage: float = 4.2
    max_time_s: float = 4500.0
    dt: float = 0.5
    nodes: int = 24
    length_m: float = 80e-6
    bulk_mol_m3: float = 1000.0
    diffusion_m2_s: float = 2.2e-10
    transference: float = 0.4
    field_t: float = 0.5
    angle_deg: float = 90.0
    permeability_m2: float = 1e-12
    viscosity_pa_s: float = 0.003
    dispersion_gain: float = 1.0
    coil_w_per_t2: float = 8.0
    coil_heat_fraction: float = 0.05
    pulse_period_s: float = 20.0
    initial_duty: float = 0.65
    min_duty: float = 0.15
    max_duty: float = 0.95
    kp: float = 0.3
    ki: float = 0.004
    kd: float = 0.1
    sensor_noise_v: float = 0.0005
    sensor_delay_s: float = 2.0
    seed: int = 2103

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("매개변수는 유한한 수여야 합니다.")
        if not 0 <= self.initial_soc < self.target_soc <= 0.95:
            raise ValueError("초기 SOC < 목표 SOC ≤ 95% 조건이 필요합니다.")
        if (
            min(
                self.capacity_ah,
                self.area_m2,
                self.c_rate,
                self.dt,
                self.length_m,
                self.bulk_mol_m3,
                self.diffusion_m2_s,
                self.viscosity_pa_s,
                self.pulse_period_s,
                self.max_time_s,
            )
            <= 0
        ):
            raise ValueError("용량·면적·시간·전류·확산계수는 양수여야 합니다.")
        if not isinstance(self.nodes, int) or not 4 <= self.nodes <= 200:
            raise ValueError("격자 수는 4~200의 정수여야 합니다.")
        if not 0 < self.min_duty <= self.initial_duty <= self.max_duty <= 1:
            raise ValueError("듀티비 범위를 확인해 주세요.")
        if self.dt > self.pulse_period_s * self.min_duty / 3:
            raise ValueError("시간 간격을 최소 ON 시간의 1/3 이하로 줄여 주세요.")
        if not 0 <= self.transference <= 1:
            raise ValueError("수송수는 0~1이어야 합니다.")
        if self.max_temp_c <= max(self.initial_c, self.ambient_c):
            raise ValueError("온도 제한은 초기·주위 온도보다 높아야 합니다.")
        if not 3.5 <= self.max_voltage <= 4.3:
            raise ValueError("전압 제한은 3.5~4.3 V여야 합니다.")
        if (
            min(
                self.field_t,
                self.dispersion_gain,
                self.permeability_m2,
                self.coil_w_per_t2,
                self.sensor_noise_v,
                self.sensor_delay_s,
                self.kp,
                self.ki,
                self.kd,
            )
            < 0
            or not 0 <= self.coil_heat_fraction <= 1
        ):
            raise ValueError(
                "자기장·효과 계수·PID 이득·센서 잡음은 음수일 수 없습니다."
            )


def field_transport(c, current, field, temp):
    j = abs(current) / c.area_m2
    force = j * field * abs(math.sin(math.radians(c.angle_deg)))
    speed = c.permeability_m2 * force / c.viscosity_pa_s
    base = c.diffusion_m2_s * math.exp(18000 / R * (1 / 298.15 - 1 / (temp + 273.15)))
    return force, speed, base + c.dispersion_gain * speed * c.length_m


def diffusion_step(profile, dt, diffusivity, sink, c):
    """Implicit finite volumes: flux at x=0, bulk Dirichlet at L. Open reservoir."""
    n, dx = len(profile), c.length_m / len(profile)
    a = diffusivity * dt / dx**2
    band = np.zeros((3, n))
    band[0, 1:] = -a
    band[2, :-1] = -a
    band[1, :] = 1 + 2 * a
    band[1, 0], band[1, -1] = 1 + a, 1 + 3 * a
    rhs = profile.copy()
    rhs[0] -= dt * sink / dx
    rhs[-1] += 2 * a * c.bulk_mol_m3
    result = solve_banded((1, 1), band, rhs, check_finite=False)
    influx = 2 * diffusivity * (c.bulk_mol_m3 - result[-1]) / dx
    residual = (result.sum() - profile.sum()) * dx - dt * (influx - sink)
    return result, residual


def electrode_state(c, soc, temp, conc, current):
    """Illustrative OCV + concentration/BV/solid polarization closures."""
    tk, ratio = temp + 273.15, max(conc / c.bulk_mol_m3, 1e-9)
    r0 = 0.012 * math.exp(16000 / R * (1 / tk - 1 / 298.15))
    j0 = 140 * math.sqrt(ratio) * math.exp(15000 / R * (1 / 298.15 - 1 / tk))
    ct = 2 * R * tk / F * math.asinh(current / (2 * c.area_m2 * j0))
    cp = R * tk / F * max(0.0, math.log(1 / ratio))
    solid = (
        0.007 * (current / c.capacity_ah) * math.exp(12000 / R * (1 / tk - 1 / 298.15))
    )
    eta = 0.12 - 0.09 * soc - ct - cp - solid
    ocv = 3.35 + 0.70 * soc + 0.08 * soc**8
    voltage = ocv + current * r0 + ct + cp + solid
    exponent = min(20.0, max(0.0, -eta * F / (2 * R * tk)))
    plated = min(current, c.area_m2 * 0.15 * math.expm1(exponent))
    return voltage, eta, plated, r0


def voltage_limited_current(c, soc, temp, conc, requested):
    if electrode_state(c, soc, temp, conc, requested)[0] <= c.max_voltage:
        return requested
    lo, hi = 0.0, requested
    for _ in range(25):
        mid = (lo + hi) / 2
        if electrode_state(c, soc, temp, conc, mid)[0] > c.max_voltage:
            hi = mid
        else:
            lo = mid
    return lo


class PID:
    def __init__(self, c):
        self.c, self.integral, self.previous = c, 0.0, None

    def update(self, stress, elapsed):
        c, e = self.c, 1 - stress
        derivative = 0.0 if self.previous is None else (e - self.previous) / elapsed
        trial = self.integral + e * elapsed
        raw = c.initial_duty + c.kp * e + c.ki * trial + c.kd * derivative
        if (
            c.min_duty <= raw <= c.max_duty
            or (raw > c.max_duty and e < 0)
            or (raw < c.min_duty and e > 0)
        ):
            self.integral = trial
        self.previous = e
        raw = c.initial_duty + c.kp * e + c.ki * self.integral + c.kd * derivative
        return float(np.clip(raw, c.min_duty, c.max_duty))


@dataclass
class Result:
    method: str
    config: Config
    data: pd.DataFrame
    profiles: np.ndarray
    status: str

    def summary(self):
        d, c = self.data, self.config
        last = d.iloc[-1]
        return {
            "방식": METHODS[self.method],
            "종료 사유": self.status,
            "목표 도달": bool(last.soc >= c.target_soc - 1e-8),
            "경과 시간 (분)": float(last.time_s / 60),
            "최종 SOC (%)": float(last.soc * 100),
            "누적 석출 (mAh)": float(last.plated_ah * 1000),
            "최고 온도 (°C)": float(d.temperature_c.max()),
            "입력 전하 (Ah)": float(last.input_ah),
            "셀 입력 에너지 (Wh)": float(last.cell_energy_wh),
            "전자석 에너지 (Wh)": float(last.coil_energy_wh),
            "총 입력 에너지 (Wh)": float(last.cell_energy_wh + last.coil_energy_wh),
        }


def simulate(c=Config(), method="combined"):
    c.validate()
    if method not in METHODS:
        raise ValueError("지원하지 않는 충전 방식입니다.")
    use_field, adaptive = method in ("field", "combined"), method in (
        "adaptive",
        "combined",
    )
    pulsed = method in ("fixed", "adaptive", "combined")
    rng, pid = np.random.default_rng(c.seed), PID(c)
    profile = np.full(c.nodes, c.bulk_mol_m3)
    soc, temp, duty = c.initial_soc, c.initial_c, c.initial_duty
    qin = qplate = cell_wh = coil_wh = t = cycle_end = cycle_start = stress = dvdt = 0.0
    r_hat = 0.02
    queue, last_sensor = [], None
    next_sensor_sample = 1.0  # Fixed 1 Hz sensor clock, independent of solver dt.
    rows, profiles = [], [profile.copy()]
    v0 = electrode_state(c, soc, temp, c.bulk_mol_m3, 0)[0]
    rows.append(
        dict(
            time_s=0.0,
            soc=soc,
            temperature_c=temp,
            voltage_v=v0,
            current_a=0.0,
            field_t=0.0,
            duty=duty if pulsed else 1.0,
            surface_mol_m3=c.bulk_mol_m3,
            plating_overpotential_v=0.12 - 0.09 * soc,
            plated_ah=0.0,
            input_ah=0.0,
            cell_energy_wh=0.0,
            coil_energy_wh=0.0,
            force_n_m3=0.0,
            speed_m_s=0.0,
            diffusivity_m2_s=c.diffusion_m2_s,
            resistance_est_ohm=r_hat,
            dvdt_v_s=0.0,
            feedback_stress=0.0,
            charge_residual_ah=0.0,
            salt_residual_mol_m2=0.0,
        )
    )
    status = "시간 제한"
    while t < c.max_time_s - 1e-9:
        if t >= cycle_end - 1e-9:
            cycle_start, cycle_end = t, t + c.pulse_period_s
            if adaptive and t > 0:
                duty = pid.update(stress, c.pulse_period_s)
        on_end = cycle_start + duty * c.pulse_period_s
        is_on = not pulsed or t < on_end - 1e-9
        h = min(c.dt, c.max_time_s - t, cycle_end - t, next_sensor_sample - t)
        if pulsed and is_on:
            h = min(h, on_end - t)
        requested = c.c_rate * c.capacity_ah if is_on else 0.0
        requested *= float(np.clip((c.max_temp_c - temp) / 3, 0.0, 1.0))
        current = voltage_limited_current(c, soc, temp, profile[0], requested)
        factor = (
            float(np.clip(0.35 + 0.65 * stress, 0.0, 1.0))
            if method == "combined"
            else 1.0
        )
        field = c.field_t * factor if use_field and current > 0 else 0.0
        for _ in range(40):
            force, speed, diffusivity = field_transport(c, current, field, temp)
            flux = (1 - c.transference) * current / (F * c.area_m2)
            next_profile, salt_residual = diffusion_step(
                profile, h, diffusivity, flux, c
            )
            if np.min(next_profile) <= 0:
                break
            v, eta, ip, r0 = electrode_state(c, soc, temp, next_profile[0], current)
            if v > c.max_voltage + 1e-7 and current > 0:
                current = voltage_limited_current(
                    c, soc, temp, next_profile[0], current
                )
                continue
            inserted = (current - ip) * h / 3600
            remaining = (c.target_soc - soc) * c.capacity_ah
            if inserted > remaining + 1e-12:
                h *= remaining / inserted
                continue
            break
        else:
            status = "수치 수렴 실패"
            break
        if np.min(next_profile) <= 0:
            status = "수송 모델 농도 고갈"
            break
        if not np.all(np.isfinite(next_profile)):
            status = "수치 오류"
            break
        qin += current * h / 3600
        qplate += ip * h / 3600
        soc += (current - ip) * h / 3600 / c.capacity_ah
        coil_w = c.coil_w_per_t2 * field**2
        cell_wh += v * current * h / 3600
        coil_wh += coil_w * h / 3600
        ocv = electrode_state(c, soc, temp, next_profile[0], 0)[0]
        heat = current * max(0.0, v - ocv) + c.coil_heat_fraction * coil_w
        temp += (heat - 0.65 * (temp - c.ambient_c)) * h / 75
        profile = next_profile
        t += h
        if t >= next_sensor_sample - 1e-9:
            queue.append((t, v + rng.normal(0, c.sensor_noise_v), current, temp))
            next_sensor_sample += 1.0
        while queue and queue[0][0] <= t - c.sensor_delay_s + 1e-9:
            sensor = queue.pop(0)
            if last_sensor is not None:
                elapsed = sensor[0] - last_sensor[0]
                di, dv = sensor[2] - last_sensor[2], sensor[1] - last_sensor[1]
                if abs(di) > max(0.5, c.capacity_ah * 0.05):
                    edge_r = dv / di
                    if 0 < edge_r < 0.5:
                        r_hat = 0.65 * r_hat + 0.35 * edge_r
                elif sensor[2] > 0.5:
                    dvdt = 0.8 * dvdt + 0.2 * dv / elapsed
            last_sensor = sensor
            sr = max(0.0, (r_hat / 0.02 - 1) / 0.8)
            sv = max(0.0, dvdt) / 0.001
            st = max(0.0, (sensor[3] - 35) / max(c.max_temp_c - 35, 1))
            sh = max(0.0, (sensor[1] - 4.0) / max(c.max_voltage - 4.0, 0.05))
            stress = max(sr, sv, st, sh)
        residual = qin - qplate - (soc - c.initial_soc) * c.capacity_ah
        rows.append(
            dict(
                time_s=t,
                soc=soc,
                temperature_c=temp,
                voltage_v=v,
                current_a=current,
                field_t=field,
                duty=duty if pulsed else 1.0,
                surface_mol_m3=float(profile[0]),
                plating_overpotential_v=eta,
                plated_ah=qplate,
                input_ah=qin,
                cell_energy_wh=cell_wh,
                coil_energy_wh=coil_wh,
                force_n_m3=force,
                speed_m_s=speed,
                diffusivity_m2_s=diffusivity,
                resistance_est_ohm=r_hat,
                dvdt_v_s=dvdt,
                feedback_stress=stress,
                charge_residual_ah=residual,
                salt_residual_mol_m2=salt_residual,
            )
        )
        profiles.append(profile.copy())
        if soc >= c.target_soc - 1e-9:
            status = "목표 SOC 도달"
            break
        if temp >= c.max_temp_c:
            status = "온도 제한"
            break
    return Result(method, c, pd.DataFrame(rows), np.asarray(profiles), status)


def impedance_spectrum(c, soc, temp, conc):
    """Synthetic Randles/Warburg response, not EIS measurements or a detector."""
    f = np.logspace(-2, 4, 120)
    w = 2 * np.pi * f
    r0 = electrode_state(c, soc, temp, conc, 0)[3]
    rct = 0.008 * math.sqrt(c.bulk_mol_m3 / max(conc, 1))
    z = r0 + rct / (1 + 1j * w * rct * 250) + 0.003 * (1 - 1j) / np.sqrt(w)
    return pd.DataFrame(
        {"frequency_hz": f, "real_ohm": z.real, "minus_imag_ohm": -z.imag}
    )


def equivalent_lithium_thickness_um(plated_ah, area_m2):
    """Faraday uniform-film equivalent, not dendrite tip length."""
    return plated_ah * 3600 / F * 0.00694 / (534 * area_m2) * 1e6

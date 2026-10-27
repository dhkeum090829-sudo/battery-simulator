"""Independent DFN benchmark, with no magnetic field or adaptive PID coupling."""

import os

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import numpy as np
import pandas as pd
import pybamm


def run_one(c, pulsed=False):
    c.validate()
    model = pybamm.lithium_ion.DFN(
        options={"lithium plating": "partially reversible", "thermal": "lumped"}
    )
    param = pybamm.ParameterValues("OKane2022")
    param.update(
        {
            "Initial temperature [K]": c.initial_c + 273.15,
            "Ambient temperature [K]": c.ambient_c + 273.15,
            "Upper voltage cut-off [V]": c.max_voltage,
            "Total heat transfer coefficient [W.m-2.K-1]": 100.0,
        }
    )
    x0, x100, _, _ = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    sto = (
        model.variables["Average negative particle concentration [mol.m-3]"]
        / param["Maximum concentration in negative electrode [mol.m-3]"]
    )
    host = (sto - x0) / (x100 - x0)
    model.variables["Research host SOC"] = host
    # Model-level events stop the entire experiment, including during rests.
    model.events.append(pybamm.Event("Research target host SOC", c.target_soc - host))
    model.events.append(
        pybamm.Event(
            "Research temperature limit",
            c.max_temp_c - model.variables["Volume-averaged cell temperature [C]"],
        )
    )
    if not pulsed:
        steps = [
            (
                pybamm.step.c_rate(
                    -c.c_rate, duration=c.max_time_s, termination=f"{c.max_voltage}V"
                ),
                pybamm.step.voltage(
                    c.max_voltage, duration=c.max_time_s, termination="C/100"
                ),
            )
        ]
    else:
        on = c.pulse_period_s * c.initial_duty
        off = c.pulse_period_s - on
        current_limit = c.c_rate * param["Nominal cell capacity [A.h]"]

        def pulse_cccv(variables):
            # Fischer-Burmeister complementarity: I >= -Imax, V <= Vmax;
            # either the current bound or the voltage bound is active.
            a = variables["Current [A]"] + current_limit
            b = current_limit * (c.max_voltage - variables["Voltage [V]"])
            return pybamm.sqrt(a * a + b * b + 0.01**2) - a - b

        steps = [
            (
                pybamm.step.CustomStepImplicit(
                    pulse_cccv, duration=on, direction="charge"
                ),
                pybamm.step.rest(duration=off),
            )
        ] * int(np.ceil(c.max_time_s / c.pulse_period_s) + 1)
    experiment = pybamm.Experiment(
        steps, period="2 seconds", termination=[f"{c.max_time_s} seconds"]
    )
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        experiment=experiment,
        solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8),
    )
    sol = sim.solve(initial_soc=c.initial_soc)

    def series(name):
        return np.asarray(sol[name].entries).reshape(-1)

    t = series("Time [s]")
    inventory = series("Loss of capacity to negative lithium plating [A.h]")
    j = series(
        "X-averaged negative electrode lithium plating volumetric interfacial current density [A.m-3]"
    )
    volume = (
        param["Negative electrode thickness [m]"]
        * param["Electrode width [m]"]
        * param["Electrode height [m]"]
        * param["Number of electrodes connected in parallel to make a cell"]
    )
    deposition = np.maximum(-j, 0) * volume
    cumulative = np.r_[
        0.0, np.cumsum((deposition[1:] + deposition[:-1]) / 2 * np.diff(t) / 3600)
    ]
    frame = pd.DataFrame(
        {
            "time_s": t,
            "host_soc": series("Research host SOC"),
            "voltage_v": series("Terminal voltage [V]"),
            "current_a": -series("Current [A]"),
            "temperature_c": series("Volume-averaged cell temperature [C]"),
            "plated_inventory_ah": inventory,
            "cumulative_deposition_ah": cumulative,
        }
    )
    if (
        t[-1] < c.max_time_s - 1e-3
        and frame.host_soc.iloc[-1] < c.target_soc - 1e-5
        and frame.temperature_c.iloc[-1] < c.max_temp_c - 1e-3
        and sol.termination == "final time"
    ):
        raise RuntimeError("DFN 해석이 목표·시간·온도 종료 조건 전에 중단되었습니다.")
    label = "고정 펄스 · CC–CV 상한" if pulsed else "CC–CV"
    summary = {
        "방식": label,
        "종료 사유": sol.termination,
        "목표 도달": bool(frame.host_soc.iloc[-1] >= c.target_soc - 1e-5),
        "경과 시간 (분)": t[-1] / 60,
        "최종 host SOC (%)": frame.host_soc.iloc[-1] * 100,
        "최대 리튬 재고 (mAh)": inventory.max() * 1000,
        "누적 석출 전하 (mAh)": cumulative[-1] * 1000,
        "최고 온도 (°C)": frame.temperature_c.max(),
    }
    return label, frame, summary


def run_comparison(c):
    frames, rows = {}, []
    for pulsed in (False, True):
        name, frame, row = run_one(c, pulsed)
        frames[name] = frame
        rows.append(row)
    return frames, pd.DataFrame(rows)

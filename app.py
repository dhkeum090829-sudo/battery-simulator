import io
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from battery_model import (
    Config,
    METHODS,
    simulate,
    impedance_spectrum,
    equivalent_lithium_thickness_um,
)

APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="배터리 충전 연구실", page_icon="🔋", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top:4rem;max-width:1500px}
[data-testid="stMetric"] {background:rgba(110,150,180,.07);padding:16px;border-radius:12px}
</style>""",
    unsafe_allow_html=True,
)
st.caption("BATTERY CHARGING LAB · 화학 × 물리 × 제어")
st.title("자기장과 적응형 펄스 충전")
st.write("같은 목표 충전량에서 충전 시간·리튬 석출·발열·전자석 에너지를 비교합니다.")
st.info(
    "연구용 가설 탐색 모델입니다. 자기장에 의한 혼합은 단순화한 가정이며, 실제 덴드라이트 형상·화재 확률·배터리 수명은 예측하지 않습니다."
)

with st.sidebar:
    st.header("공통 실험 조건")
    with st.form("config_form"):
        rate = st.slider("최대 충전율 (C)", 0.5, 6.0, 3.0, 0.5)
        start = st.slider("초기 SOC (%)", 0, 70, 10, 5)
        target = st.slider("목표 SOC (%)", 20, 95, 80, 5)
        ambient = st.slider("초기·주위 온도 (°C)", -5, 35, 25)
        duration = st.slider("계산 시간 상한 (분)", 10, 90, 75, 5)
        st.subheader("자기장")
        field = st.slider("자기장 상한 (T)", 0.0, 1.0, 0.5, 0.05)
        angle = st.slider("전류와 자기장 사이 각도 (°)", 0, 180, 90, 15)
        gain = st.slider("혼합 효과 계수 α (가정)", 0.0, 5.0, 1.0, 0.25)
        permeability = st.select_slider(
            "투과도 K (m², 가정)",
            options=["1e-14", "1e-13", "1e-12", "1e-11"],
            value="1e-12",
        )
        st.subheader("펄스와 PID")
        period = st.slider("펄스 주기 (초)", 10, 60, 20, 5)
        duty = st.slider("초기 통전 비율 (%)", 20, 90, 65, 5)
        kp = st.number_input("Kp", 0.0, 3.0, 0.3, 0.05)
        ki = st.number_input("Ki (1/s)", 0.0, 0.1, 0.004, 0.001, format="%.3f")
        kd = st.number_input("Kd (s)", 0.0, 10.0, 0.1, 0.1)
        with st.expander("센서·수치·에너지 설정"):
            noise = st.slider("전압 센서 잡음 표준편차 (mV)", 0.0, 3.0, 0.5, 0.1)
            delay = st.slider("센서 지연 (초)", 0.0, 10.0, 2.0, 0.5)
            dt = st.select_slider(
                "최대 시간 간격 (초)", options=[0.25, 0.5, 1.0], value=0.5
            )
            coil = st.number_input(
                "전자석 전력 계수 (W/T², 가정)", 0.0, 100.0, 8.0, 1.0
            )
        submitted = st.form_submit_button("조건 적용 · 5가지 방식 비교", type="primary")
    st.caption(
        "모든 방식에 동일한 최대 전류·4.2 V 상한·45°C 제한을 적용합니다. 휴지 시간도 충전 시간에 포함됩니다."
    )

config = Config(
    c_rate=rate,
    initial_soc=start / 100,
    target_soc=target / 100,
    ambient_c=float(ambient),
    initial_c=float(ambient),
    max_time_s=duration * 60,
    field_t=field,
    angle_deg=float(angle),
    dispersion_gain=gain,
    permeability_m2=float(permeability),
    pulse_period_s=float(period),
    initial_duty=duty / 100,
    kp=kp,
    ki=ki,
    kd=kd,
    sensor_noise_v=noise / 1000,
    sensor_delay_s=delay,
    dt=dt,
    coil_w_per_t2=coil,
)
try:
    config.validate()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_data(show_spinner=False, max_entries=12)
def calculate(c):
    return {m: simulate(c, m) for m in METHODS}


with st.spinner("확산·전극 반응·열·피드백 제어를 계산하고 있습니다…"):
    results = calculate(config)
summary = pd.DataFrame([r.summary() for r in results.values()])
colors = dict(zip(METHODS, ["#ef6c63", "#4ba3d3", "#bd93e8", "#e6aa46", "#23b59f"]))
tabs = st.tabs(
    [
        "비교 대시보드",
        "작동 과정",
        "가정·민감도·수치 검증",
        "PyBaMM 독립 비교",
        "최적 조건 탐색",
    ]
)


def chart(column, label, scale=1.0, x_soc=False):
    fig = go.Figure()
    for m, result in results.items():
        d = result.data
        x = d.soc * 100 if x_soc else d.time_s / 60
        fig.add_trace(
            go.Scatter(
                x=x,
                y=d[column] * scale,
                name=METHODS[m],
                mode="lines",
                line=dict(color=colors[m], width=2),
            )
        )
    fig.update_layout(
        height=310,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis_title="SOC (%)" if x_soc else "경과 시간 (분)",
        yaxis_title=label,
        legend=dict(orientation="h", y=-0.28),
        template="plotly_white",
    )
    return fig


def dual_chart(data, left_key, left_label, right_key, right_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.time_s / 60, y=data[left_key], name=left_label))
    fig.add_trace(
        go.Scatter(x=data.time_s / 60, y=data[right_key], name=right_label, yaxis="y2")
    )
    fig.update_layout(
        height=270,
        xaxis_title="경과 시간 (분)",
        yaxis_title=left_label,
        yaxis2=dict(title=right_label, overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.3),
        margin=dict(t=20),
    )
    return fig


with tabs[0]:
    baseline, combined = results["cc"], results["combined"]
    b, z = baseline.summary(), combined.summary()
    cols = st.columns(4)
    cols[0].metric("결합 제어 최종 SOC", f"{z['최종 SOC (%)']:.1f}%")
    cols[1].metric("결합 제어 경과 시간", f"{z['경과 시간 (분)']:.2f}분")
    cols[2].metric("결합 제어 누적 석출", f"{z['누적 석출 (mAh)']:.3f} mAh")
    cols[3].metric("결합 제어 최고 온도", f"{z['최고 온도 (°C)']:.2f}°C")
    if not summary["목표 도달"].all():
        st.warning(
            "목표 SOC에 도달하지 않은 방식이 있습니다. 해당 행은 완료 시간으로 해석할 수 없습니다. 종료 사유와 최종 SOC를 함께 확인하세요."
        )
    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "경과 시간 (분)": st.column_config.NumberColumn(format="%.2f"),
            "최종 SOC (%)": st.column_config.NumberColumn(format="%.2f"),
            "누적 석출 (mAh)": st.column_config.NumberColumn(format="%.4f"),
            "최고 온도 (°C)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "누적 석출은 단순화한 비가역 부반응의 누적 전하입니다. 가역 리튬 재고·실측 덴드라이트 길이와 다릅니다. 듀티비는 에너지 효율이 아닙니다."
    )
    x_soc = st.toggle("가로축을 SOC로 비교", value=False)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart("soc", "SOC (%)", 100, x_soc), width="stretch")
        st.plotly_chart(
            chart("plated_ah", "누적 석출 전하 (mAh)", 1000, x_soc), width="stretch"
        )
    with right:
        st.plotly_chart(
            chart("temperature_c", "전지 온도 (°C)", x_soc=x_soc), width="stretch"
        )
        st.plotly_chart(
            chart("voltage_v", "단자 전압 (V)", x_soc=x_soc), width="stretch"
        )
    st.subheader("실험 데이터 저장")
    csv = pd.concat(
        [r.data.assign(method=m) for m, r in results.items()], ignore_index=True
    )
    a, bcol = st.columns(2)
    a.download_button(
        "전체 시계열 CSV",
        csv.to_csv(index=False).encode("utf-8-sig"),
        "charging_timeseries.csv",
        "text/csv",
    )
    packet = io.BytesIO()
    with zipfile.ZipFile(packet, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "config.json", json.dumps(asdict(config), ensure_ascii=False, indent=2)
        )
        archive.writestr("summary.csv", summary.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("timeseries.csv", csv.to_csv(index=False))
        for m, r in results.items():
            archive.writestr(
                f"concentration_{m}.csv", pd.DataFrame(r.profiles).to_csv(index=False)
            )
        archive.writestr(
            "MODEL_NOTES.md", (APP_ROOT / "MODEL_NOTES.md").read_text(encoding="utf-8")
        )
    bcol.download_button(
        "설정·결과·농도 분포 ZIP",
        packet.getvalue(),
        "battery_experiment.zip",
        "application/zip",
    )

with tabs[1]:
    method = st.selectbox(
        "자세히 볼 충전 방식", list(METHODS), format_func=METHODS.get, index=4
    )
    result = results[method]
    d = result.data
    st.subheader("전류 · 통전 비율 · 자기장")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.time_s / 60, y=d.current_a, name="충전 전류 (A)"))
    fig.add_trace(
        go.Scatter(x=d.time_s / 60, y=d.duty * 100, name="통전 비율 (%)", yaxis="y2")
    )
    fig.update_layout(
        height=300,
        xaxis_title="분",
        yaxis_title="A",
        yaxis2=dict(title="%", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=20),
    )
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(
        dual_chart(d, "field_t", "자기장 (T)", "feedback_stress", "부담 지표 (무차원)"),
        width="stretch",
    )
    st.caption(
        "자기장 상위 제어는 이전 센서 신호의 부담 지표를 사용합니다. PID는 펄스 시작 때 다음 듀티비를 결정합니다."
    )
    idx = st.slider("상태를 살펴볼 기록 위치", 0, len(d) - 1, len(d) - 1)
    row = d.iloc[idx]
    cols = st.columns(4)
    cols[0].metric("기록 시각", f"{row.time_s:.1f} s")
    cols[1].metric("|j × B|", f"{row.force_n_m3:.3f} N/m³")
    cols[2].metric("가정한 Darcy 유속", f"{row.speed_m_s:.2e} m/s")
    cols[3].metric(
        "등가 균일 석출 두께",
        f"{equivalent_lithium_thickness_um(row.plated_ah,config.area_m2):.4f} μm",
    )
    left, right = st.columns(2)
    with left:
        x = (np.arange(config.nodes) + 0.5) * config.length_m / config.nodes * 1e6
        fig = go.Figure(go.Scatter(x=x, y=result.profiles[idx], mode="lines+markers"))
        fig.update_layout(
            title="전극측 → 벌크 전해질 농도",
            xaxis_title="거리 (μm)",
            yaxis_title="농도 (mol/m³)",
            height=300,
            margin=dict(t=45),
        )
        st.plotly_chart(fig, width="stretch")
        sample = np.unique(np.linspace(0, len(d) - 1, min(160, len(d))).astype(int))
        fig = go.Figure(
            go.Heatmap(
                z=result.profiles[sample],
                x=x,
                y=d.time_s.iloc[sample] / 60,
                colorscale="Viridis",
                colorbar=dict(title="mol/m³"),
            )
        )
        fig.update_layout(
            title="시간에 따른 농도 분포",
            xaxis_title="거리 (μm)",
            yaxis_title="분",
            height=300,
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        spectrum = impedance_spectrum(
            config, row.soc, row.temperature_c, row.surface_mol_m3
        )
        fig = go.Figure(
            go.Scatter(
                x=spectrum.real_ohm * 1000,
                y=spectrum.minus_imag_ohm * 1000,
                mode="lines",
            )
        )
        fig.update_layout(
            title="합성 임피던스 응답 · Nyquist",
            xaxis_title="Re(Z) (mΩ)",
            yaxis_title="−Im(Z) (mΩ)",
            height=300,
            margin=dict(t=45),
        )
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(
            dual_chart(
                d, "resistance_est_ohm", "펄스 저항 추정 (Ω)", "dvdt_v_s", "dV/dt (V/s)"
            ),
            width="stretch",
        )
        st.caption(
            "임피던스 곡선은 Randles–Warburg 등가회로 예시입니다. 제어기는 이 곡선을 보지 않고 펄스 경계의 ΔV/ΔI 저항 추정값과 필터링한 dV/dt를 사용합니다."
        )
    st.caption(
        "등가 두께는 Faraday 법칙으로 계산한 균일 막 두께입니다. 덴드라이트 끝 길이·분리막 관통 여부·성장 형상을 뜻하지 않습니다."
    )

with tabs[2]:
    st.subheader("어디까지 계산했는가")
    st.markdown((APP_ROOT / "MODEL_NOTES.md").read_text(encoding="utf-8"))
    st.subheader("보존식 확인")
    check = pd.DataFrame(
        [
            {
                "방식": METHODS[m],
                "전하 수지 최대 오차 (Ah)": r.data.charge_residual_ah.abs().max(),
                "확산층 수지 최대 오차 (mol/m²)": r.data.salt_residual_mol_m2.abs().max(),
            }
            for m, r in results.items()
        ]
    )
    st.dataframe(check, hide_index=True, width="stretch")
    st.caption(
        "확산층은 벌크 저장소와 물질을 교환하는 열린 영역입니다. 이 검사는 전지 전체 염 보존 검사가 아닙니다."
    )
    if st.button("혼합 효과 민감도 계산 · α = 0, 현재값, 2배"):
        with st.spinner("같은 결합 제어 조건에서 가정의 영향을 비교합니다…"):
            cases = []
            for value in sorted(
                set([0.0, config.dispersion_gain, 2 * config.dispersion_gain])
            ):
                r = simulate(replace(config, dispersion_gain=value), "combined")
                cases.append({"α": value, **r.summary()})
            st.session_state["sensitivity"] = (asdict(config), pd.DataFrame(cases))
    if "sensitivity" in st.session_state and st.session_state["sensitivity"][
        0
    ] == asdict(config):
        st.dataframe(
            st.session_state["sensitivity"][1], hide_index=True, width="stretch"
        )
    if st.button("시간·공간 격자 수렴 확인 · 결합 제어"):
        with st.spinner("dt/2 및 공간 격자 2배 결과를 비교합니다…"):
            checks = []
            for label, c in [
                ("현재", config),
                ("dt/2", replace(config, dt=config.dt / 2)),
                ("격자 2배", replace(config, nodes=config.nodes * 2)),
            ]:
                r = simulate(c, "combined")
                checks.append({"설정": label, **r.summary()})
            st.session_state["convergence"] = (asdict(config), pd.DataFrame(checks))
    if "convergence" in st.session_state and st.session_state["convergence"][
        0
    ] == asdict(config):
        st.dataframe(
            st.session_state["convergence"][1], hide_index=True, width="stretch"
        )
    st.caption(
        "수치 수렴과 물리적 타당성은 다른 검증입니다. 매개변수 보정과 독립 실험 비교가 추가로 필요합니다."
    )

with tabs[3]:
    st.subheader("DFN 전기화학 모델로 펄스 충전 비교")
    st.write(
        "기존 PyBaMM 기능을 유지하면서 동일한 전극 host SOC 목표·전류·전압·온도 제한을 사용하도록 수정했습니다."
    )
    st.caption(
        "고정 펄스는 각 통전 구간에 CC–CV 상한을 적용합니다. 이 탭의 PyBaMM 모델에는 자기장과 PID가 연결되지 않습니다. 위 단순화 모델의 자기장 효과를 검증한 결과로 해석할 수 없습니다."
    )
    if st.button("PyBaMM 비교 실행 · CC–CV와 고정 펄스"):
        try:
            from dfn_validation import run_comparison

            with st.spinner("DFN 해석 중입니다. 조건에 따라 수 분이 걸릴 수 있습니다…"):
                st.session_state["dfn"] = (asdict(config), run_comparison(config))
        except ImportError:
            st.error("PyBaMM 설치가 필요합니다: pip install -r requirements.txt")
        except Exception as exc:
            st.error(f"DFN 계산을 완료하지 못했습니다: {exc}")
    if "dfn" in st.session_state and st.session_state["dfn"][0] == asdict(config):
        frames, table = st.session_state["dfn"][1]
        st.dataframe(table, hide_index=True, width="stretch")
        fig = go.Figure()
        for name, frame in frames.items():
            fig.add_trace(
                go.Scatter(x=frame.time_s / 60, y=frame.host_soc * 100, name=name)
            )
        fig.update_layout(xaxis_title="분", yaxis_title="전극 host SOC (%)")
        st.plotly_chart(fig, width="stretch")
        left, right = st.columns(2)
        for col, key, label, scale in [
            (left, "voltage_v", "단자 전압 (V)", 1),
            (right, "plated_inventory_ah", "리튬 재고 (mAh)", 1000),
        ]:
            with col:
                fig = go.Figure()
                for name, frame in frames.items():
                    fig.add_trace(
                        go.Scatter(x=frame.time_s / 60, y=frame[key] * scale, name=name)
                    )
                fig.update_layout(
                    xaxis_title="경과 시간 (분)", yaxis_title=label, height=300
                )
                st.plotly_chart(fig, width="stretch")
        st.download_button(
            "DFN 시계열 CSV",
            pd.concat([v.assign(method=k) for k, v in frames.items()])
            .to_csv(index=False)
            .encode("utf-8-sig"),
            "dfn_comparison.csv",
        )
        st.caption(
            "host SOC는 음극 고체의 평균 화학량론을 신품 0~100% 구간으로 정규화한 값입니다. 단순화 모델의 SOC와 모델 정의가 다릅니다."
        )


with tabs[4]:
    from optimization_ui import render_optimizer

    render_optimizer(config)

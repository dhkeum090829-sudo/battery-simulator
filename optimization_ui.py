"""Streamlit view for constrained, finite candidate search."""

from dataclasses import asdict
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from battery_model import METHODS
from optimization import (
    SearchSpace,
    Limits,
    search,
    recommend,
    validate_candidate,
    export_search,
)

LABELS = {
    "candidate_id": "후보 ID",
    "method_label": "방식",
    "reached": "목표 도달",
    "feasible": "제약 통과",
    "pareto": "파레토 후보",
    "elapsed_min": "경과 시간 (분)",
    "plating_mah": "누적 석출 (mAh)",
    "energy_wh": "총 입력 에너지 (Wh)",
    "max_temp_c": "최고 온도 (°C)",
    "max_voltage_v": "최고 전압 (V)",
    "coil_energy_wh": "전자석 에너지 (Wh)",
    "final_soc_pct": "최종 SOC (%)",
    "reason": "판정 사유",
    "field_limit_t": "자기장 상한 (T)",
    "period_s": "펄스 주기 (초)",
    "initial_duty": "초기 듀티비",
    "scenario": "검사 조건",
    "reference_reached": "기준 충전 목표 도달",
    "plating_ratio": "기준 대비 석출 비율",
    "energy_ratio": "기준 대비 에너지 비율",
    "initial_c": "초기 온도 (°C)",
    "dispersion_gain": "혼합 계수 α",
    "sensor_noise_v": "센서 잡음 (V)",
    "sensor_delay_s": "센서 지연 (초)",
}


def view_table(table, columns):
    valid = [c for c in columns if c in table]
    return table[valid].rename(columns=LABELS)


def candidate_plots(report, chosen):
    completed = report.table[report.table.reached]
    left, right = st.columns(2)
    groups = [
        ("제약 미통과", completed[~completed.feasible], "#999999", "x"),
        (
            "제약 통과",
            completed[completed.feasible & ~completed.pareto],
            "#4ba3d3",
            "circle",
        ),
        ("파레토 후보", completed[completed.pareto], "#23b59f", "diamond"),
    ]
    for col, x, label in [
        (left, "elapsed_min", "경과 시간 (분)"),
        (right, "energy_wh", "총 입력 에너지 (Wh)"),
    ]:
        with col:
            fig = go.Figure()
            for name, frame, color, symbol in groups:
                if frame.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=frame[x],
                        y=frame.plating_mah,
                        mode="markers",
                        name=name,
                        marker=dict(color=color, symbol=symbol, size=9),
                        customdata=frame[
                            [
                                "candidate_id",
                                "method_label",
                                "energy_wh",
                                "max_temp_c",
                                "reason",
                            ]
                        ].to_numpy(),
                        hovertemplate="%{customdata[0]} · %{customdata[1]}<br>"
                        + label
                        + ": %{x:.4f}<br>석출: %{y:.5f} mAh<br>에너지: %{customdata[2]:.4f} Wh"
                        + "<br>최고 온도: %{customdata[3]:.2f}°C<br>%{customdata[4]}<extra></extra>",
                    )
                )
            if chosen is not None:
                row = report.table.set_index("candidate_id").loc[chosen]
                fig.add_trace(
                    go.Scatter(
                        x=[row[x]],
                        y=[row.plating_mah],
                        mode="markers",
                        name="선택 기준에 따른 추천",
                        marker=dict(symbol="star", size=17, color="#e6aa46"),
                    )
                )
            fig.update_layout(
                height=340,
                xaxis_title=label,
                yaxis_title="누적 석출 (mAh)",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=15),
                template="plotly_white",
            )
            st.plotly_chart(fig, width="stretch")
    st.caption(
        "그래프에는 목표 SOC에 도달한 조건만 표시합니다. 파레토 판정에는 시간·석출량·총 입력 에너지 세 값을 함께 사용합니다."
    )


def render_optimizer(base):
    st.subheader("제약 조건을 만족하는 충전 조건 탐색")
    st.write(
        "허용 범위를 먼저 정하고, 그 범위 안에서 석출량·시간·에너지의 우선순위로 후보를 선택합니다."
    )
    st.caption(
        "추천은 표본으로 계산한 후보 중의 선택입니다. 전역 최적점이나 실제 배터리의 안전 인증을 뜻하지 않습니다. 최대 충전 전류와 목표 SOC는 왼쪽 공통 조건으로 고정합니다."
    )
    with st.form("optimization_form"):
        a, b, c = st.columns(3)
        deadline = a.number_input(
            "허용 충전 시간 (분)", min_value=0.1, max_value=180.0, value=20.0, step=1.0
        )
        extra = b.number_input(
            "기준 대비 에너지 증가 허용 (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )
        plate = c.number_input(
            "기준 대비 석출량 상한 (%)",
            min_value=0.0,
            max_value=200.0,
            value=100.0,
            step=5.0,
        )
        temp = a.number_input(
            "최고 온도 상한 (°C)",
            min_value=-20.0,
            max_value=float(base.max_temp_c),
            value=float(base.max_temp_c),
            step=1.0,
        )
        pairs = b.select_slider("탐색 표본 쌍", options=[8, 16, 32, 64], value=16)
        seed = c.number_input(
            "탐색 재현용 시드", min_value=0, max_value=2147483647, value=2103, step=1
        )
        with st.expander("자기장·펄스·PID 탐색 범위"):
            a, b = st.columns(2)
            if base.field_t > 0:
                field_min = a.number_input(
                    "자기장 최소 (T)", 0.0, float(base.field_t), 0.0, 0.05
                )
                field_max = b.number_input(
                    "자기장 최대 (T)",
                    0.0,
                    float(base.field_t),
                    float(base.field_t),
                    0.05,
                )
            else:
                field_min = field_max = 0.0
                st.caption(
                    "공통 자기장 상한이 0 T이므로 자기장을 사용하지 않는 조건을 탐색합니다."
                )
            minimum = max(10.0, float(np.ceil(3 * base.dt / base.min_duty)))
            pmin = a.number_input("펄스 주기 최소 (초)", 1.0, 120.0, minimum, 1.0)
            pmax = b.number_input(
                "펄스 주기 최대 (초)", 1.0, 120.0, max(40.0, minimum), 1.0
            )
            dmin = a.number_input("초기 듀티비 최소 (%)", 15.0, 95.0, 35.0, 5.0)
            dmax = b.number_input("초기 듀티비 최대 (%)", 15.0, 95.0, 90.0, 5.0)
            kpmin = a.number_input("Kp 최소", 0.0, 3.0, 0.05, 0.05)
            kpmax = b.number_input("Kp 최대", 0.0, 3.0, 0.7, 0.05)
            kimin = a.number_input("Ki 최소", 0.0, 0.1, 0.0, 0.001, format="%.3f")
            kimax = b.number_input("Ki 최대", 0.0, 0.1, 0.01, 0.001, format="%.3f")
            kdmin = a.number_input("Kd 최소", 0.0, 10.0, 0.0, 0.1)
            kdmax = b.number_input("Kd 최대", 0.0, 10.0, 0.5, 0.1)
        run = st.form_submit_button("최적 조건 탐색 실행", type="primary")
    space = SearchSpace(
        field_min=field_min,
        field_max=field_max,
        period_min=pmin,
        period_max=pmax,
        duty_min=dmin / 100,
        duty_max=dmax / 100,
        kp_min=kpmin,
        kp_max=kpmax,
        ki_min=kimin,
        ki_max=kimax,
        kd_min=kdmin,
        kd_max=kdmax,
        pairs=pairs,
        seed=seed,
    )
    limits = Limits(
        deadline_min=deadline,
        energy_ratio=1 + extra / 100,
        plating_ratio=plate / 100,
        max_temp_c=temp,
        max_voltage=base.max_voltage,
    )
    signature = hashlib.sha256(
        json.dumps(
            {"base": asdict(base), "space": asdict(space), "limits": asdict(limits)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if run:
        try:
            space.validate(base)
            limits.validate()
            bar = st.progress(0.0, text="탐색 준비")
            report = search(
                base,
                space,
                limits,
                progress=lambda n, total, label: bar.progress(
                    n / total, text=f"{n}/{total} · {label}"
                ),
            )
            bar.empty()
            st.session_state["optimization_report"] = (signature, report)
            st.session_state.pop("optimization_validation", None)
            st.session_state.pop("optimization_candidate", None)
        except ValueError as exc:
            st.error(str(exc))
            return
    stored = st.session_state.get("optimization_report")
    if not stored or stored[0] != signature:
        st.info(
            "연구 제약과 탐색 범위를 정한 뒤 실행하세요. 기본값은 20분 이내, 에너지 증가 5% 이내, 석출량은 기준 충전 이하입니다."
        )
        return
    report = stored[1]
    table = report.table
    a, b, c = st.columns(3)
    a.metric("계산한 후보", str(len(table)))
    b.metric("제약 통과", str(int(table.feasible.sum())))
    c.metric("파레토 후보", str(int(table.pareto.sum())))
    if not report.baseline["reached"]:
        st.warning(
            "기준 CC–CV가 목표 SOC에 도달하지 못했습니다. 상대 에너지·석출량 제한을 평가할 수 없어 추천하지 않습니다. 공통 계산 시간이나 조건을 조정해 주세요."
        )
    if report.baseline["reached"]:
        st.caption(
            f"현재 제한값: 석출 ≤ {report.baseline['plating_mah']*limits.plating_ratio:.5f} mAh · "
            f"총 에너지 ≤ {report.baseline['energy_wh']*limits.energy_ratio:.4f} Wh · "
            f"시간 ≤ {limits.deadline_min:g}분 · 최고 온도 ≤ {limits.max_temp_c:g}°C"
        )
    priority = st.selectbox(
        "제약 안에서의 추천 기준",
        ["plating", "time", "energy"],
        format_func={
            "plating": "석출량 최소 → 시간 → 에너지",
            "time": "시간 최소 → 석출량 → 에너지",
            "energy": "에너지 최소 → 석출량 → 시간",
        }.get,
        key="optimization_priority",
    )
    chosen = recommend(report, priority)
    candidate_plots(report, chosen)
    validation = None
    selected = None
    if chosen is None:
        st.warning(
            "모든 제약을 만족하는 후보가 없습니다. 탐색 범위·표본 수를 늘리거나 연구 제약을 조정할 수 있습니다. 이 결과만으로 가능한 조건이 없다고 결론 내릴 수는 없습니다."
        )
    else:
        row = table.set_index("candidate_id").loc[chosen]
        st.subheader(f"현재 기준의 추천 후보 · {chosen}")
        a, b, c = st.columns(3)
        a.metric("충전 시간", f"{row.elapsed_min:.2f}분")
        b.metric("누적 석출", f"{row.plating_mah:.5f} mAh")
        c.metric("총 입력 에너지", f"{row.energy_wh:.4f} Wh")
        st.dataframe(
            view_table(
                table[table.candidate_id == chosen],
                [
                    "candidate_id",
                    "method_label",
                    "field_limit_t",
                    "period_s",
                    "initial_duty",
                    "kp",
                    "ki",
                    "kd",
                ],
            ),
            hide_index=True,
            width="stretch",
        )
        counterpart = None
        if chosen.startswith("p") and chosen.endswith("-on"):
            counterpart = chosen[:-3] + "-off"
        elif chosen.startswith("p") and chosen.endswith("-off"):
            counterpart = chosen[:-4] + "-on"
        elif chosen == "ref-combined":
            counterpart = "ref-adaptive"
        elif chosen == "ref-field":
            counterpart = "ref-cc"
        if counterpart in report.candidates:
            st.caption(
                "같은 제어 설정의 자기장 ON/OFF 비교 · 전체 개선과 자기장 추가 효과를 구분합니다."
            )
            st.dataframe(
                view_table(
                    table[table.candidate_id.isin([chosen, counterpart])],
                    [
                        "candidate_id",
                        "method_label",
                        "elapsed_min",
                        "plating_mah",
                        "energy_wh",
                        "coil_energy_wh",
                    ],
                ),
                hide_index=True,
                width="stretch",
            )
        feasible_ids = table.loc[table.feasible, "candidate_id"].tolist()
        selected = st.selectbox(
            "추가 검증할 후보",
            feasible_ids,
            index=feasible_ids.index(chosen),
            key="optimization_candidate",
            format_func=lambda x: f"{x} · {report.table.set_index('candidate_id').loc[x,'method_label']}",
        )
        st.caption(
            "추가 검증: 기준 조건 1개, 온도 ±5°C × 혼합 효과 0·1·2배 × 센서 시드 3개, 시간·공간 격자 검사 2개. 스트레스 조건은 잡음 최소 1 mV와 지연 2초 증가를 함께 적용합니다."
        )
        if st.button("선택 후보 추가 검증 · 30개 조건", key="optimization_validate"):
            bar = st.progress(0.0, text="후보와 기준 CC–CV를 같은 환경에서 비교")
            validation = validate_candidate(
                report,
                selected,
                progress=lambda n, total, label: bar.progress(
                    n / total, text=f"{n}/{total} · {label}"
                ),
            )
            bar.empty()
            st.session_state["optimization_validation"] = (
                signature,
                selected,
                validation,
            )
        saved = st.session_state.get("optimization_validation")
        if saved and saved[:2] == (signature, selected):
            validation = saved[2]
            passed = int(validation.feasible.sum())
            st.metric("추가 검사 조건 통과", f"{passed}/{len(validation)}")
            if passed == len(validation):
                st.success(
                    "검사한 모든 조건에서 연구 제약을 만족했습니다. 실제 배터리의 안전성을 보장하는 결과는 아닙니다."
                )
            else:
                st.warning(
                    "일부 검사 조건에서 제약을 만족하지 못했습니다. 명목 조건의 추천만으로 조건을 확정하지 마세요."
                )
            st.dataframe(
                view_table(
                    validation,
                    [
                        "scenario",
                        "reached",
                        "reference_reached",
                        "feasible",
                        "elapsed_min",
                        "plating_mah",
                        "energy_wh",
                        "max_temp_c",
                        "reason",
                    ],
                ),
                hide_index=True,
                width="stretch",
            )
    st.subheader("전체 탐색 기록")
    st.dataframe(
        view_table(
            table,
            [
                "candidate_id",
                "method_label",
                "reached",
                "feasible",
                "pareto",
                "elapsed_min",
                "plating_mah",
                "energy_wh",
                "max_temp_c",
                "reason",
            ],
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("모든 후보의 제어 설정"):
        st.dataframe(
            view_table(
                table,
                [
                    "candidate_id",
                    "method_label",
                    "field_limit_t",
                    "period_s",
                    "initial_duty",
                    "kp",
                    "ki",
                    "kd",
                ],
            ),
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        "탐색 설정·판정·검증 결과 ZIP",
        export_search(report, priority, validation, selected),
        "charging_optimization.zip",
        "application/zip",
        key="optimization_download",
    )

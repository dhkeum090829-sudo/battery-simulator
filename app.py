import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pybamm

# ==========================================
# 0. 웹사이트 기본 설정 & UI 헤더
# ==========================================
st.set_page_config(page_title="배터리 덴드라이트 통합 시뮬레이터", layout="wide")

st.markdown("""
    <div style='background-color:#1E1E1E; padding:20px; border-radius:10px;'>
        <h1 style='color:#00D4FF; text-align:center; font-family:sans-serif;'>초고속 충전 배터리 덴드라이트 통합 시뮬레이터</h1>
        <p style='color:#E0E0E0; text-align:center; font-size:16px;'>
        실시간 알고리즘 최적화 및 PyBaMM 정밀 물리/화학 검증 시스템
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["실시간 최적화 대시보드", "정밀 물리 시뮬레이션 (PyBaMM)"])

# 세션 상태 초기화 (전역 공유)
if 'charge_val' not in st.session_state:
    st.session_state.charge_val = 10
if 'rest_val' not in st.session_state:
    st.session_state.rest_val = 5

# =====================================================================
# 공통 함수: 최적값 탐색 알고리즘
# =====================================================================
def find_optimal_condition():
    best_c, best_r, best_efficiency = 10, 5, 0.0
    safety_limit = 0.003 
    for c in range(2, 61):
        for r in range(1, 31):
            growth_per_cycle = 0.000030 * (c / 10.0) * c
            recovery_per_cycle = 0.000025 * (r / 5.0) * r
            if growth_per_cycle <= recovery_per_cycle and growth_per_cycle < safety_limit:
                efficiency = c / (c + r)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_c, best_r = c, r
    st.session_state.charge_val = best_c
    st.session_state.rest_val = best_r

# =====================================================================
# [탭 1] 실시간 최적화 대시보드
# =====================================================================
with tab1:
    st.markdown("### 실시간 화재 위험 억제 시뮬레이터")
    st.write("수치를 직접 입력하거나 최적 효율 자동 찾기 버튼을 실행하여 시뮬레이션 결과를 확인하십시오.")
    
    @st.cache_data 
    def simulate_fast_plating(charge_time, rest_time, total_time=1500, dt=1):
        time_array = np.arange(0, total_time, dt)
        plating_amount = np.zeros_like(time_array, dtype=float)
        current_plating = 0.0
        cycle_time = charge_time + rest_time
        
        growth_rate = 0.000030 * (charge_time / 10.0)
        recovery_rate = 0.000025 * (rest_time / 5.0)
        
        for i, t in enumerate(time_array):
            time_in_cycle = t % cycle_time
            if time_in_cycle < charge_time:
                current_plating += growth_rate * dt
            else:
                current_plating -= recovery_rate * dt
                if current_plating < 0:
                    current_plating = 0.0 
            plating_amount[i] = current_plating
        return time_array, plating_amount

    t_baseline, plating_baseline = simulate_fast_plating(charge_time=1500, rest_time=0)

    col_ui, col_graph = st.columns([1, 3])
    
    with col_ui:
        st.markdown("#### 제어 패널")
        if st.button("최적 효율 자동 찾기 (탭1)", type="primary", use_container_width=True):
            find_optimal_condition()
            st.rerun()
            
        charge_time = st.number_input("충전 시간 (초)", min_value=2, max_value=60, value=int(st.session_state.charge_val), step=1, key="num_c1")
        rest_time = st.number_input("휴지 시간 (초)", min_value=1, max_value=30, value=int(st.session_state.rest_val), step=1, key="num_r1")
        
        st.session_state.charge_val = charge_time
        st.session_state.rest_val = rest_time

    with col_graph:
        t_proposed, plating_proposed = simulate_fast_plating(charge_time, rest_time)
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#FAFAFA')
        ax1.plot(t_baseline, plating_baseline, color="#FF4B4B", linewidth=2, alpha=0.8, label="Baseline (High Risk)")
        ax1.plot(t_proposed, plating_proposed, color="#00D4FF", linewidth=3, label=f"Proposed ({charge_time}s ON / {rest_time}s OFF)")
        ax1.axhline(y=0.003, color='#FF9F36', linestyle='--', linewidth=2, label='Safety Limit (0.003 Ah)')
        ax1.fill_between(t_proposed, 0, plating_proposed, color='#00D4FF', alpha=0.1)
        ax1.set_title(f"Lithium Plating Simulation: {charge_time}s Charge / {rest_time}s Rest", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Lithium Plating [Ah]")
        ax1.set_xlim(0, 1500)
        ax1.set_ylim(0, 0.04)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig1)

        efficiency = int((charge_time / (charge_time + rest_time)) * 100)
        max_val = np.max(plating_proposed)
        
        if max_val < 0.003:
            st.success(f"[상태: 안정 (Stable)] 덴드라이트 성장이 안전 기준치 이내로 억제되었습니다. (최대 석출량: {max_val:.5f} Ah / 충전 효율: {efficiency}%)")
        else:
            st.error(f"[상태: 불안정 (Unstable / Risk)] 리튬 석출량이 안전 한계선을 초과했습니다. 휴지 시간을 늘려주십시오. (최대 석출량: {max_val:.5f} Ah)")

# =====================================================================
# [탭 2] 정밀 물리 시뮬레이션
# =====================================================================
with tab2:
    st.markdown("### PyBaMM 물리/화학 정밀 시뮬레이션")
    st.write("실제 배터리 내부의 이온 확산 지배방정식을 연산하여 정확한 전압 및 석출량을 계산합니다. 수치를 직접 입력하거나 최적값을 불러와 검증하십시오.")
    
    col_ui2, col_desc2 = st.columns([1, 2])
    
    with col_ui2:
        st.markdown("#### 정밀 검증 제어 패널")
        if st.button("최적 효율 자동 찾기 (탭2)", type="primary", use_container_width=True):
            find_optimal_condition()
            st.rerun()
            
        current_c = st.number_input("정밀 검증 충전 시간 (초)", min_value=2, max_value=60, value=int(st.session_state.charge_val), step=1, key="num_c2")
        current_r = st.number_input("정밀 검증 휴지 시간 (초)", min_value=1, max_value=30, value=int(st.session_state.rest_val), step=1, key="num_r2")
        
        st.session_state.charge_val = current_c
        st.session_state.rest_val = current_r

    with col_desc2:
        st.info(f"설정된 검증 조건: [{current_c}초 충전 / {current_r}초 휴지]")
        st.write("하단의 버튼을 클릭하면 DFN 모델을 기반으로 한 정밀 물리 연산이 실행됩니다. (소요 시간: 약 30~60초)")
        
        run_sim = st.button("정밀 시뮬레이션 실행", type="primary")

    if run_sim:
        with st.spinner("배터리 내부 미분 방정식 연산 중입니다... 잠시만 기다려 주십시오."):
            try:
                options = {"lithium plating": "partially reversible", "thermal": "lumped"}
                model = pybamm.lithium_ion.DFN(options=options)
                param = pybamm.ParameterValues("OKane2022")

                exp_cc = pybamm.Experiment(["Charge at 3C until 4.1V", "Rest for 10 minutes"])
                sim_cc = pybamm.Simulation(model, parameter_values=param, experiment=exp_cc)
                sol_cc = sim_cc.solve(initial_soc=0.1)

                pulse_steps = []
                cycle_count = int(700 / (current_c + current_r)) + 1 
                for _ in range(cycle_count):
                    pulse_steps.append(f"Charge at 2.5C for {current_c} seconds")
                    pulse_steps.append(f"Rest for {current_r} seconds")
                pulse_steps.append("Rest for 10 minutes")
                
                exp_pulse = pybamm.Experiment(pulse_steps)
                sim_pulse = pybamm.Simulation(model, parameter_values=param, experiment=exp_pulse)
                sol_pulse = sim_pulse.solve(initial_soc=0.1)

                plate_var = "Loss of capacity to negative lithium plating [A.h]"
                plating_cc = sol_cc[plate_var].entries
                plating_pulse = sol_pulse[plate_var].entries
                t_cc = sol_cc["Time [s]"].entries
                v_cc = sol_cc["Terminal voltage [V]"].entries
                t_pulse = sol_pulse["Time [s]"].entries
                v_pulse = sol_pulse["Terminal voltage [V]"].entries

                fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].plot(t_cc, v_cc, label="Baseline (CC)", color="#d62728", linewidth=2)
                axes[0].plot(t_pulse, v_pulse, label=f"Proposed ({current_c}s/{current_r}s)", color="#1f77b4", linewidth=2)
                axes[0].set_title("Terminal Voltage Comparison", fontsize=14, fontweight='bold')
                axes[0].set_xlabel("Time [s]")
                axes[0].set_ylabel("Terminal Voltage [V]")
                axes[0].legend()
                axes[0].grid(True, linestyle='--')

                axes[1].plot(t_cc, plating_cc, label="Baseline (CC)", color="#d62728", linewidth=2)
                axes[1].plot(t_pulse, plating_pulse, label=f"Proposed ({current_c}s/{current_r}s)", color="#1f77b4", linewidth=2)
                axes[1].set_title("True Lithium Plating (DFN Model)", fontsize=14, fontweight='bold')
                axes[1].set_xlabel("Time [s]")
                axes[1].set_ylabel("Lithium Plating [Ah]")
                axes[1].legend()
                axes[1].grid(True, linestyle='--')

                st.pyplot(fig2)
                st.success("정밀 물리 시뮬레이션 연산이 완료되었습니다. 결과를 확인하십시오.")
                
            except Exception as e:
                st.error(f"시뮬레이션 연산 중 오류가 발생했습니다: {e}")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 세션 상태 초기화 (버튼으로 슬라이더를 움직이기 위해 필요)
# ==========================================
if 'charge_val' not in st.session_state:
    st.session_state.charge_val = 10
if 'rest_val' not in st.session_state:
    st.session_state.rest_val = 5

# ==========================================
# 1. 시뮬레이터 수학 모델 엔진
# ==========================================
def simulate_fast_plating(charge_time, rest_time, total_time=1500, dt=1):
    time_array = np.arange(0, total_time, dt)
    plating_amount = np.zeros_like(time_array, dtype=float)
    
    current_plating = 0.0
    cycle_time = charge_time + rest_time
    
    # 덴드라이트 생성 및 회복 가속도
    growth_rate = 0.000025 * (charge_time / 10.0)
    recovery_rate = 0.000015 * (rest_time / 5.0)
    
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

# 기준(Baseline) 데이터 미리 계산
t_baseline, plating_baseline = simulate_fast_plating(charge_time=1500, rest_time=0)

# ==========================================
# 2. 최적 효율(Golden Ratio) 자동 탐색 알고리즘
# ==========================================
def find_optimal_condition():
    best_c, best_r = 10, 5
    best_efficiency = 0.0
    
    # 가능한 모든 충전(2~60초)/휴지(1~30초) 경우의 수를 1초 만에 스캔
    for c in range(2, 61):
        for r in range(1, 31):
            growth_per_cycle = 0.000025 * (c / 10.0) * c
            recovery_per_cycle = 0.000015 * (r / 5.0) * r
            
            # 조건 1: 한 사이클 끝날 때 덴드라이트가 0으로 회복되어야 함 (장기 누적 방지)
            # 조건 2: 순간 치솟는 최대 석출량이 0.004 Ah를 넘지 않아야 함 (안전 한계선)
            if growth_per_cycle <= recovery_per_cycle and growth_per_cycle < 0.004:
                efficiency = c / (c + r)  # 충전 시간 비율 (높을수록 충전이 빨리 끝남)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_c = c
                    best_r = r
                    
    # 찾은 최적값을 세션 상태에 덮어씌움 -> 슬라이더가 자동으로 움직임
    st.session_state.charge_val = best_c
    st.session_state.rest_val = best_r

# ==========================================
# 3. 웹사이트 화면 구성
# ==========================================
st.set_page_config(page_title="배터리 덴드라이트 시뮬레이터", layout="wide")
st.title("⚡ 실시간 배터리 화재 위험 억제 시뮬레이터")
st.markdown("좌측의 슬라이더를 움직이거나 **최적 효율 찾기** 버튼을 눌러 가장 안전하고 빠른 충전 조건을 확인하세요!")

# 사이드바 UI
st.sidebar.header("⚙️ 시뮬레이션 제어 패널")

# 최적화 버튼
if st.sidebar.button("✨ 최적 효율 자동 찾기", use_container_width=True):
    find_optimal_condition()

# 슬라이더 (st.session_state 값과 연동)
charge_time = st.sidebar.slider("충전 시간 (초)", min_value=2, max_value=60, 
                                value=st.session_state.charge_val, step=1, key="charge_val")
rest_time = st.sidebar.slider("휴지 시간 (초)", min_value=1, max_value=30, 
                              value=st.session_state.rest_val, step=1, key="rest_val")

# 데이터 실시간 계산
t_proposed, plating_proposed = simulate_fast_plating(charge_time, rest_time)

# ==========================================
# 4. 그래프 출력
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_baseline, plating_baseline, color="#d62728", linewidth=2.5, label="Baseline (No Rest, High Risk)")
ax.plot(t_proposed, plating_proposed, color="#1f77b4", linewidth=2.5, 
        label=f"Proposed ({charge_time}s ON / {rest_time}s OFF)")

ax.set_title(f"Dendrite Risk Simulation | Current Setting: {charge_time}s Charge / {rest_time}s Rest", fontsize=15, fontweight='bold')
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Lithium Plating Amount [Ah]", fontsize=12)
ax.set_xlim(0, 1500)
ax.set_ylim(0, 0.04)
ax.axhline(y=0.004, color='orange', linestyle=':', linewidth=2, label='Safety Limit (0.004 Ah)') # 안전 한계선 표시
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.7)

st.pyplot(fig)

st.success(f"**현재 충전 효율:** 전체 시간의 **{int((charge_time/(charge_time+rest_time))*100)}%**를 충전에 사용 중입니다.")
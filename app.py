import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==========================================
# 0. 세션 상태 초기화 (자동 최적화 연동)
# ==========================================
if 'charge_val' not in st.session_state:
    st.session_state.charge_val = 10
if 'rest_val' not in st.session_state:
    st.session_state.rest_val = 5

# ==========================================
# 1. 시뮬레이터 수학 모델 엔진
# ==========================================
@st.cache_data # 연산 속도를 높이기 위한 캐싱
def simulate_fast_plating(charge_time, rest_time, total_time=1500, dt=1):
    time_array = np.arange(0, total_time, dt)
    plating_amount = np.zeros_like(time_array, dtype=float)
    
    current_plating = 0.0
    cycle_time = charge_time + rest_time
    
    # 덴드라이트 생성 및 회복 비율 (미세 조정)
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

# ==========================================
# 2. 정교한 최적 효율 탐색 알고리즘
# ==========================================
def find_optimal_condition():
    best_c, best_r = 10, 5
    best_efficiency = 0.0
    
    # 안전 마진(Safety Limit) 설정: 0.003 Ah를 절대 넘지 않게 엄격히 통제
    safety_limit = 0.003 
    
    for c in range(2, 61):
        for r in range(1, 31):
            # 한 사이클당 생성/회복량
            growth_per_cycle = 0.000030 * (c / 10.0) * c
            recovery_per_cycle = 0.000025 * (r / 5.0) * r
            
            # 조건: 1사이클 후 석출량이 0으로 리셋되며, 순간 최대치가 안전선 미만일 것
            if growth_per_cycle <= recovery_per_cycle and growth_per_cycle < safety_limit:
                efficiency = c / (c + r)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_c = c
                    best_r = r
                    
    st.session_state.charge_val = best_c
    st.session_state.rest_val = best_r

# ==========================================
# 3. 세련된 대시보드 UI (Streamlit 구성)
# ==========================================
st.set_page_config(page_title="배터리 덴드라이트 최적화 시뮬레이터", page_icon="🔋", layout="wide")

# 헤더 디자인
st.markdown("""
    <div style='background-color:#1E1E1E; padding:20px; border-radius:10px;'>
        <h1 style='color:#00D4FF; text-align:center; font-family:sans-serif;'>🔋 초고속 충전 배터리 덴드라이트 억제 시뮬레이터</h1>
        <p style='color:#E0E0E0; text-align:center; font-size:18px;'>적응형 펄스 제어(Adaptive Pulse Control)를 통한 화재 위험 실시간 모니터링</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# 사이드바 디자인
st.sidebar.markdown("## ⚙️ 제어 패널")
st.sidebar.markdown("충전과 휴지 비율을 조절하여 가장 안전한 조건을 찾아보세요.")

# 최적화 버튼
if st.sidebar.button("✨ 최적 효율 자동 찾기", type="primary", use_container_width=True):
    find_optimal_condition()

st.sidebar.markdown("---")

# 드래그 시 실시간 즉각 반응을 위해 on_change 인자 활용
def update_sliders():
    pass

charge_time = st.sidebar.slider("⚡ 충전 시간 (초) - 통전 구간", min_value=2, max_value=60, 
                                value=st.session_state.charge_val, step=1, key="charge_val", on_change=update_sliders)
rest_time = st.sidebar.slider("⏳ 휴지 시간 (초) - 이온 회복 구간", min_value=1, max_value=30, 
                              value=st.session_state.rest_val, step=1, key="rest_val", on_change=update_sliders)

# ==========================================
# 4. 실시간 그래프 시각화 (Matplotlib 디자인 향상)
# ==========================================
t_proposed, plating_proposed = simulate_fast_plating(charge_time, rest_time)

fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FAFAFA')
ax.set_facecolor('#FFFFFF')

# 기준선(위험)과 제안선(안전) 플로팅
ax.plot(t_baseline, plating_baseline, color="#FF4B4B", linewidth=2, alpha=0.8, label="Baseline (CC, No Rest - High Risk)")
ax.plot(t_proposed, plating_proposed, color="#00D4FF", linewidth=3, label=f"Proposed ({charge_time}s ON / {rest_time}s OFF)")

# 디자인 강화: 안전 한계선 및 영역 채우기
ax.axhline(y=0.003, color='#FF9F36', linestyle='--', linewidth=2, label='Safety Limit (0.003 Ah)')
ax.fill_between(t_proposed, 0, plating_proposed, color='#00D4FF', alpha=0.1)

# 라벨 및 눈금 디자인
ax.set_title(f"Lithium Plating Simulation: {charge_time}s Charge / {rest_time}s Rest", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("Time [s]", fontsize=13, fontweight='medium')
ax.set_ylabel("Lithium Plating Amount [Ah]", fontsize=13, fontweight='medium')
ax.set_xlim(0, 1500)
ax.set_ylim(0, 0.04)
ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True, facecolor='white')
ax.grid(True, linestyle=':', alpha=0.7, color='#CCCCCC')

# 테두리 디자인 제거
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 화면 중앙에 그래프 출력
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    st.pyplot(fig)
    
    # 하단 상태창 표시
    efficiency = int((charge_time / (charge_time + rest_time)) * 100)
    if np.max(plating_proposed) < 0.003:
        st.success(f"✅ **[안전]** 덴드라이트 성장이 완벽히 억제되고 있습니다. (현재 충전 효율: {efficiency}%)")
    else:
        else:
        st.error(f"⚠️ **[경고]** 리튬 석출량이 안전 한계선을 초과했습니다! 휴지 시간을 늘려주세요. (현재 충전 효율: {efficiency}%)")

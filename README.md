# 자기장·적응형 펄스 충전 연구 시뮬레이터

기존 Streamlit 앱에 자기장에 의한 물질 전달 가정과 센서 피드백 기반 PID 펄스 제어를 추가했습니다. 일반 CC–CV / 자기장 CC–CV / 고정 펄스 / 적응형 펄스 / 결합 제어를 같은 목표 삽입 SOC·최대 전류·전압·온도 제한에서 비교합니다.

## 실행
Python 3.11 이상 권장.
~~~powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
~~~
기존 Streamlit Community Cloud의 진입점 app.py를 그대로 사용합니다. 검토 후 이 브랜치를 main에 병합하면 연결된 배포가 새 코드를 반영할 수 있습니다.

## 사용
왼쪽에서 공통 조건을 변경하고 '조건 적용 · 5가지 방식 비교'를 누릅니다. 비교 화면에서 최종 SOC와 종료 사유를 먼저 확인합니다. 목표에 도달하지 않은 결과의 경과 시간을 완료 시간으로 해석하지 않습니다.

작동 과정 탭에서 전류·듀티비·자기장·농도 분포·가상 센서 응답을 확인합니다. 가정 탭에서 혼합 계수 민감도, 시간·공간 격자 수렴, 수지 오차를 검사할 수 있습니다. CSV와 ZIP에는 실행 설정과 원시 결과를 저장합니다.

PyBaMM 탭은 별도 DFN 비교입니다. 자기장/PID가 적용된 DFN이라고 표시하지 않습니다. 빠른 모델과 DFN의 출력 크기를 같은 전지의 일치성 검증으로 해석하지 않습니다.

## 검증
~~~powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~
전하·확산층 수지, 자기장 0/평행/혼합 계수 0, 휴지 중 농도 회복, 동일 목표 SOC, 전압·듀티 제한, 제어 재현성, 수치 수렴, Streamlit 기본 화면을 확인합니다.

## 범위
실제 셀에 보정되지 않은 연구 프로토타입입니다. 와류 CFD, 2D 덴드라이트 형태, 자기장 공간 분포, 장기 수명·화재 확률 모델은 포함하지 않습니다. 상세 식과 가정은 MODEL_NOTES.md를 참고하세요. 첨부 문서의 제출·발표 지시는 앱의 실행 명령이나 외부 제출 권한으로 사용하지 않았습니다.


원시 결과와 독립 HTML 보고서 저장: python export_experiment.py --output results --dfn

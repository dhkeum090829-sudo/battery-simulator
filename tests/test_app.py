from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_and_detail_tabs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert len(at.dataframe) >= 2
    assert len(at.metric) >= 8
    at.selectbox[0].set_value("fixed").run()
    assert not at.exception
    at.toggle[0].set_value(True).run()
    assert not at.exception


def test_optimizer_runs_and_invalidates_stale_results(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    at = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
    next(x for x in at.select_slider if x.label == "탐색 표본 쌍").set_value(8)
    next(x for x in at.button if x.label == "최적 조건 탐색 실행").click().run()
    assert not at.exception
    _, report = at.session_state["optimization_report"]
    assert len(report.table) == 21
    assert report.table.feasible.any()
    at.selectbox(key="optimization_priority").set_value("energy").run()
    assert not at.exception
    next(x for x in at.slider if x.label == "목표 SOC (%)").set_value(75)
    next(x for x in at.button if x.label == "조건 적용 · 5가지 방식 비교").click().run()
    assert not at.exception
    assert not any(x.key == "optimization_validate" for x in at.button)

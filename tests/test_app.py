from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_and_detail_tabs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.dataframe) >= 2
    assert len(at.metric) >= 8
    at.selectbox[0].set_value("fixed").run()
    assert not at.exception
    at.toggle[0].set_value(True).run()
    assert not at.exception

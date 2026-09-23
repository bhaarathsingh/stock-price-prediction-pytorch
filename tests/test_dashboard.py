from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_dashboard_opens_and_runs_synthetic_experiment():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    app.slider[1].set_value(5)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[2].value == "155"

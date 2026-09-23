from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_lstm.config import TrainConfig
from stock_lstm.data import download_prices, prepare_data, synthetic_prices, validate_prices


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, "not a price"])
def test_invalid_prices_are_rejected(bad):
    frame = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Close": [10, bad]})
    with pytest.raises(ValueError):
        validate_prices(frame)


def test_duplicate_dates_and_empty_dates_are_rejected():
    for dates in (["2024-01-01", "2024-01-01"], ["2024-01-01", None]):
        with pytest.raises(ValueError):
            validate_prices(pd.DataFrame({"Date": dates, "Close": [10, 11]}))


def test_input_sorted_without_mutating_original():
    frame = synthetic_prices(50).iloc[::-1]
    original = frame.copy()
    clean = validate_prices(frame)
    assert clean.Date.is_monotonic_increasing
    pd.testing.assert_frame_equal(frame, original)


def test_windows_stop_before_target_and_splits_are_chronological():
    frame = synthetic_prices(100)
    config = TrainConfig(lookback=5)
    data = prepare_data(frame, config)
    all_indices = []
    for split in (data.train, data.validation, data.test):
        for x, y, target in zip(split.x, split.y, split.target_indices, strict=True):
            np.testing.assert_allclose(
                data.scaler.inverse(x[:, 0]), frame.Close.iloc[target - 5 : target], rtol=1e-6
            )
            np.testing.assert_allclose(
                data.scaler.inverse(y), [frame.Close.iloc[target]], rtol=1e-6
            )
        all_indices.extend(split.target_indices)
    assert all_indices == list(range(5, 100))
    assert data.train.target_indices[-1] < data.validation.target_indices[0]
    assert data.validation.target_indices[-1] < data.test.target_indices[0]


def test_future_values_cannot_change_scaler_or_training_windows():
    frame = synthetic_prices(100)
    data = prepare_data(frame, TrainConfig(lookback=5))
    frame.loc[data.scaler_fit_end :, "Close"] *= 100
    changed = prepare_data(frame, TrainConfig(lookback=5))
    assert data.scaler == changed.scaler
    assert data.scaler.mean == pytest.approx(data.frame.Close.iloc[: data.scaler_fit_end].mean())
    np.testing.assert_array_equal(data.train.x, changed.train.x)
    np.testing.assert_array_equal(data.train.y, changed.train.y)


def test_constant_series_and_short_input():
    frame = synthetic_prices(100)
    frame["Close"] = 100.0
    data = prepare_data(frame, TrainConfig())
    assert np.isfinite(data.train.x).all()
    with pytest.raises(ValueError, match="Not enough"):
        prepare_data(frame.head(15), TrainConfig())


@pytest.mark.parametrize(
    "settings",
    [
        {"lookback": 0},
        {"epochs": 0},
        {"patience": -1},
        {"learning_rate": float("nan")},
        {"train_fraction": 0.9, "validation_fraction": 0.2},
        {"seed": -1},
        {"device": "bad"},
    ],
)
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        replace(TrainConfig(), **settings)


def test_provider_multilevel_output_and_adjustment_flag(monkeypatch, tmp_path):
    import sys

    calls = {}
    frame = pd.DataFrame(
        {("Close", "AAPL"): [100, 101]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )

    def download(*args, **kwargs):
        calls.update(kwargs)
        return frame

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=download, set_tz_cache_location=lambda _: None),
    )
    result = download_prices("AAPL", "2024-01-01", "2024-01-04", tmp_path)
    assert result.Close.tolist() == [100, 101]
    assert calls["auto_adjust"] is True
    assert calls["multi_level_index"] is False


def test_provider_empty_response_is_explicit(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(
            download=lambda *a, **kw: pd.DataFrame(), set_tz_cache_location=lambda _: None
        ),
    )
    with pytest.raises(ValueError, match="returned no data"):
        download_prices("AAPL", "2024-01-01", "2024-02-01", tmp_path)

import numpy as np
import pytest
import torch

from stock_lstm.cli import main
from stock_lstm.config import TrainConfig
from stock_lstm.data import synthetic_prices
from stock_lstm.experiment import run_experiment
from stock_lstm.metrics import regression_metrics
from stock_lstm.model import PriceLSTM
from stock_lstm.reporting import forecast_from_checkpoint, write_report


@pytest.fixture(scope="module")
def result():
    return run_experiment(
        synthetic_prices(100),
        TrainConfig(epochs=3, lookback=5, hidden_size=4, num_layers=1, batch_size=32),
        "test data",
    )


def test_metric_units_and_persistence_direction():
    metrics = regression_metrics([11, 13], [10, 11], [10, 11])
    assert metrics["mae"] == 1.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["direction_accuracy"] == 0
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1], [1, 2])


def test_network_shape():
    assert PriceLSTM(8, 2)(torch.zeros(4, 29, 1)).shape == (4, 1)


def test_reports_checkpoint_roundtrip_and_no_overwrite(result, tmp_path):
    out = tmp_path / "report"
    write_report(result, out)
    assert {"forecast.svg", "metrics.json", "predictions.csv", "history.csv", "model.pt"}.issubset(
        {p.name for p in out.iterdir()}
    )
    forecast = forecast_from_checkpoint(out / "model.pt", result.data.frame)
    assert forecast["lstm"] == pytest.approx(result.summary["next_observation"]["lstm"])
    with pytest.raises(ValueError, match="not empty"):
        write_report(result, out)


def test_heldout_values_do_not_change_selected_weights(result):
    changed = result.data.frame.copy()
    changed.loc[result.data.test.target_indices[0] :, "Close"] *= 3
    second = run_experiment(changed, result.config)
    assert result.summary["best_epoch"] == second.summary["best_epoch"]
    np.testing.assert_array_equal(result.history.to_numpy(), second.history.to_numpy())
    for key, value in result.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)


def test_baseline_uses_previous_observation_and_test_is_aligned(result):
    test = result.predictions.query("split == 'test'")
    indices = result.data.test.target_indices
    np.testing.assert_array_equal(test.persistence, result.data.frame.Close.iloc[indices - 1])
    np.testing.assert_array_equal(test.actual, result.data.frame.Close.iloc[indices])
    assert len(test) == result.summary["splits"]["test"]["targets"]


def test_cli_bad_input_returns_useful_error(tmp_path, capsys):
    assert main(["train", "--csv", str(tmp_path / "missing.csv")]) == 2
    assert "Error:" in capsys.readouterr().err


def test_cli_demo_writes_report(tmp_path):
    assert (
        main(
            [
                "demo",
                "--out",
                str(tmp_path / "demo"),
                "--rows",
                "80",
                "--epochs",
                "1",
                "--lookback",
                "5",
                "--hidden-size",
                "4",
                "--num-layers",
                "1",
            ]
        )
        == 0
    )
    assert (tmp_path / "demo" / "metrics.json").exists()

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stock_lstm.config import TrainConfig
from stock_lstm.data import Standardizer, validate_prices
from stock_lstm.experiment import Experiment, predict_scaled
from stock_lstm.model import PriceLSTM

matplotlib.use("Agg")


def forecast_figure(result: Experiment):
    holdout = result.predictions.query("split == 'test'")
    dates = pd.to_datetime(holdout["Date"])
    colors = {"actual": "#0f172a", "lstm": "#2563eb", "persistence": "#d97706"}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    for name in ("actual", "lstm", "persistence"):
        axes[0].plot(
            dates,
            holdout[name],
            label=name.title(),
            color=colors[name],
            linewidth=1.8,
            linestyle="--" if name == "persistence" else "-",
        )
    label = result.summary["label"]
    rmse = result.summary["metrics"]["test"]
    axes[0].set_title(f"{label} | Next-observation close prediction", loc="left", weight="bold")
    axes[0].set_ylabel("Close price (input units)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(
        dates,
        np.abs(holdout["lstm"] - holdout["actual"]),
        color="#2563eb",
        label="LSTM absolute error",
    )
    axes[1].axhline(rmse["lstm"]["rmse"], color="#2563eb", linestyle=":", label="LSTM RMSE")
    axes[1].axhline(
        rmse["persistence"]["rmse"], color="#d97706", linestyle="--", label="Persistence RMSE"
    )
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Held-out target date | Each estimate uses only earlier observed closes")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    return fig


def write_report(result: Experiment, destination: Path) -> None:
    """Use a new directory so a prior experiment cannot be silently overwritten."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --out path")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(result.summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    result.predictions.to_csv(destination / "predictions.csv", index=False)
    result.history.to_csv(destination / "history.csv", index=False)
    figure = forecast_figure(result)
    figure.savefig(destination / "forecast.svg", metadata={"Date": None})
    plt.close(figure)
    fig, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for name in ("train_mse", "validation_mse"):
        axis.plot(result.history["epoch"], result.history[name], label=name)
    axis.axvline(
        result.summary["best_epoch"], color="#94a3b8", linestyle=":", label="Selected checkpoint"
    )
    axis.set(xlabel="Epoch", ylabel="MSE (standardized units)", title="Training and validation")
    axis.legend(frameon=False)
    fig.savefig(destination / "loss.svg", metadata={"Date": None})
    plt.close(fig)
    torch.save(
        {
            "format_version": 1,
            "state_dict": result.model.state_dict(),
            "config": asdict(result.config),
            "scaler": asdict(result.data.scaler),
            "trained_through": result.summary["splits"]["train"]["last_target"],
            "label": result.summary["label"],
        },
        destination / "model.pt",
    )


def forecast_from_checkpoint(checkpoint: Path, frame: pd.DataFrame) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = TrainConfig(**payload["config"])
    frame = validate_prices(frame)
    if len(frame) < config.lookback:
        raise ValueError(f"At least {config.lookback} prices are required")
    scaler = Standardizer(**payload["scaler"])
    model = PriceLSTM(config.hidden_size, config.num_layers)
    model.load_state_dict(payload["state_dict"])
    tail = scaler.transform(frame["Close"].to_numpy()[-config.lookback :])
    x = tail.astype(np.float32)[None, :, None]
    return {
        "as_of": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "lstm": float(scaler.inverse(predict_scaled(model, x, "cpu"))[0]),
        "persistence": float(frame["Close"].iloc[-1]),
        "trained_through": payload["trained_through"],
        "target": "next observed session; exchange calendar is not inferred",
    }

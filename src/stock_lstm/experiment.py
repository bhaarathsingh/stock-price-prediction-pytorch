import copy
import hashlib
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stock_lstm.config import TrainConfig
from stock_lstm.data import PreparedData, prepare_data
from stock_lstm.metrics import regression_metrics
from stock_lstm.model import PriceLSTM


@dataclass
class Experiment:
    model: PriceLSTM
    data: PreparedData
    config: TrainConfig
    history: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict


def predict_scaled(model, x, device, batch_size=256):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    result = np.concatenate(outputs).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite predictions")
    return result


def run_experiment(frame, config: TrainConfig, label: str = "CSV") -> Experiment:
    """Select weights only on validation MSE; report the untouched test targets once."""
    data = prepare_data(frame, config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    model = PriceLSTM(config.hidden_size, config.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.train.x), torch.from_numpy(data.train.y)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_weights = None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Training diverged; reduce the learning rate")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        val_prediction = predict_scaled(model, data.validation.x, device)
        val_loss = float(np.mean((val_prediction - data.validation.y.reshape(-1)) ** 2))
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_loss / len(data.train.y),
                "validation_mse": val_loss,
            }
        )
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_weights is None:
        raise ValueError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_weights)
    frames, metrics, splits = [], {}, {}
    prices = data.frame["Close"].to_numpy()
    for split_name in ("train", "validation", "test"):
        split = getattr(data, split_name)
        indices = split.target_indices
        actual, previous = prices[indices], prices[indices - 1]
        predicted = data.scaler.inverse(predict_scaled(model, split.x, device))
        metrics[split_name] = {
            "lstm": regression_metrics(actual, predicted, previous),
            "persistence": regression_metrics(actual, previous, previous),
        }
        dates = data.frame["Date"].iloc[indices].dt.strftime("%Y-%m-%d")
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates.to_numpy(),
                    "split": split_name,
                    "actual": actual,
                    "lstm": predicted,
                    "persistence": previous,
                }
            )
        )
        splits[split_name] = {
            "targets": len(indices),
            "first_target": dates.iloc[0],
            "last_target": dates.iloc[-1],
        }
    tail = data.scaler.transform(prices[-config.lookback :]).astype(np.float32)[None, :, None]
    next_close = float(data.scaler.inverse(predict_scaled(model, tail, device))[0])
    csv = data.frame.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.10g")
    summary = {
        "label": label,
        "task": "one-step-ahead observed-history evaluation",
        "config": asdict(config),
        "device_used": device,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "data_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "rows": len(prices),
        "splits": splits,
        "scaler": {
            **asdict(data.scaler),
            "fit_rows": data.scaler_fit_end,
            "fit_through": splits["train"]["last_target"],
        },
        "metrics": metrics,
        "next_observation": {
            "as_of": data.frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "lstm": next_close,
            "persistence": float(prices[-1]),
        },
        "versions": {
            "python_torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return Experiment(
        model.cpu(),
        data,
        config,
        pd.DataFrame(history),
        pd.concat(frames, ignore_index=True),
        summary,
    )

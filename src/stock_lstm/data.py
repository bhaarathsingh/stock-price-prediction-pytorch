import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_lstm.config import TrainConfig


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Require one positive daily price per date; never impute future information."""
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError("CSV must contain Date and Close columns")
    if frame.empty:
        raise ValueError("No price rows were provided")
    result = frame[["Date", "Close"]].copy()
    try:
        result["Date"] = pd.to_datetime(result["Date"], errors="raise", utc=True).dt.normalize()
        result["Close"] = pd.to_numeric(result["Close"], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates and prices must be valid") from exc
    if result["Date"].isna().any() or result["Date"].duplicated().any():
        raise ValueError("Dates must be nonempty and unique at daily frequency")
    if not np.isfinite(result["Close"]).all() or (result["Close"] <= 0).any():
        raise ValueError("Close prices must be finite and greater than zero")
    result["Date"] = result["Date"].dt.tz_localize(None)
    return result.sort_values("Date").reset_index(drop=True)


def read_prices(path) -> pd.DataFrame:
    return validate_prices(pd.read_csv(path))


def synthetic_prices(rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Seeded geometric random walk, not observations of any actual security."""
    if rows < 2:
        raise ValueError("At least two synthetic observations are required")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, rows - 1)
    prices = 100 * np.exp(np.r_[0, np.cumsum(returns)])
    return pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=rows), "Close": prices})


def download_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch a single ticker; adjusted historical data can be revised by its provider."""
    if not re.fullmatch(r"[A-Za-z0-9.^=\-]{1,24}", ticker):
        raise ValueError("Provide one ticker symbol, such as AAPL")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must precede end (end is exclusive)")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError('Install market support: pip install -e ".[market]"') from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    frame = yf.download(
        ticker.upper(),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=20,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise ValueError("The market provider returned no data. Check dates/ticker or use a CSV.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("Expected exactly one ticker in the provider response")
        close = close.iloc[:, 0]
    return validate_prices(pd.DataFrame({"Date": close.index, "Close": close.to_numpy()}))


@dataclass(frozen=True)
class Standardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray):
        scale = float(np.std(values))
        return cls(float(np.mean(values)), scale if scale > 1e-12 else 1.0)

    def transform(self, values):
        return (np.asarray(values) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean


@dataclass
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray


@dataclass
class PreparedData:
    frame: pd.DataFrame
    scaler: Standardizer
    train: WindowSet
    validation: WindowSet
    test: WindowSet
    scaler_fit_end: int


def prepare_data(frame: pd.DataFrame, config: TrainConfig) -> PreparedData:
    frame = validate_prices(frame)
    targets = np.arange(config.lookback, len(frame))
    n_train = int(len(targets) * config.train_fraction)
    n_validation = int(len(targets) * config.validation_fraction)
    n_test = len(targets) - n_train - n_validation
    if min(n_train, n_validation, n_test) < 2:
        raise ValueError("Not enough observations for at least two targets in every split")
    prices = frame["Close"].to_numpy()
    fit_end = int(targets[n_train - 1]) + 1
    scaler = Standardizer.fit(prices[:fit_end])
    scaled = scaler.transform(prices).astype(np.float32)
    x = np.stack([scaled[t - config.lookback : t] for t in targets])[:, :, None]
    y = scaled[targets, None]

    def subset(start, stop):
        return WindowSet(x[start:stop], y[start:stop], targets[start:stop])

    return PreparedData(
        frame,
        scaler,
        subset(0, n_train),
        subset(n_train, n_train + n_validation),
        subset(n_train + n_validation, None),
        fit_end,
    )

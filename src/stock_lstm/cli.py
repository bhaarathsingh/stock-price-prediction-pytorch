import argparse
import json
import sys
from pathlib import Path

from stock_lstm.config import TrainConfig
from stock_lstm.data import download_prices, read_prices, synthetic_prices
from stock_lstm.experiment import run_experiment
from stock_lstm.reporting import forecast_from_checkpoint, write_report


def parser():
    root = argparse.ArgumentParser(description="Stock LSTM: reproducible one-step forecasting")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, default=Path("runs") / name)
        if name == "demo":
            command.add_argument("--rows", type=int, default=800)
        else:
            command.add_argument("--csv", type=Path, required=True)
            command.add_argument("--label", default="CSV prices")
        for flag in (
            "lookback",
            "hidden_size",
            "num_layers",
            "epochs",
            "batch_size",
            "patience",
            "seed",
        ):
            command.add_argument(
                "--" + flag.replace("_", "-"), type=int, default=getattr(TrainConfig(), flag)
            )
        command.add_argument("--learning-rate", type=float, default=0.001)
        command.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    download = commands.add_parser("download")
    download.add_argument("--ticker", required=True)
    download.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD")
    download.add_argument("--end", required=True, help="Exclusive YYYY-MM-DD")
    download.add_argument("--out", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--csv", type=Path, required=True)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "download":
            if arguments.out.exists():
                raise ValueError("Output CSV already exists; choose a new --out file")
            frame = download_prices(
                arguments.ticker,
                arguments.start,
                arguments.end,
                arguments.out.parent / ".yfinance-cache",
            )
            arguments.out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(arguments.out, index=False)
            print(f"Saved {len(frame)} adjusted daily closes to {arguments.out}")
            return 0
        if arguments.command == "predict":
            print(
                json.dumps(
                    forecast_from_checkpoint(arguments.checkpoint, read_prices(arguments.csv)),
                    indent=2,
                )
            )
            return 0
        if arguments.out.exists() and any(arguments.out.iterdir()):
            raise ValueError("Output directory is not empty; choose a new --out path")
        config_fields = set(TrainConfig.__dataclass_fields__)
        config = TrainConfig(
            **{key: value for key, value in vars(arguments).items() if key in config_fields}
        )
        if arguments.command == "demo":
            frame, label = synthetic_prices(arguments.rows, config.seed), "SYNTHETIC random walk"
        else:
            frame, label = read_prices(arguments.csv), arguments.label
        result = run_experiment(frame, config, label)
        write_report(result, arguments.out)
        test = result.summary["metrics"]["test"]
        print(f"{label} | {result.summary['splits']['test']['targets']} held-out targets")
        print(f"LSTM RMSE: {test['lstm']['rmse']:.4f}")
        print(f"Persistence RMSE: {test['persistence']['rmse']:.4f}")
        print(f"Best epoch: {result.summary['best_epoch']} / {result.summary['epochs_run']}")
        print(f"Report: {arguments.out / 'metrics.json'}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

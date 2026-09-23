# Validation record

Recorded on September 23, 2026 using Python 3.12.14 and CPU PyTorch 2.14.0.
Direct package versions are recorded in `constraints-tested.txt`.

## Completed checks

- **27 tests passed** locally, covering invalid data, target/window alignment,
  chronological splits, training-only normalization, holdout isolation, baseline
  alignment, metric units, checkpoint reloads, CLI output, and the Streamlit app.
- Ruff lint and formatting checks passed.
- `pip check` reported no broken requirements.
- All **six notebook code cells** executed sequentially through IPython, and the
  notebook passed `nbformat` schema validation.
- The standard synthetic CLI demo completed with the default configuration.
- The generated forecast chart was rendered and visually inspected.

The external Jupyter kernel launcher could not run because this workspace prohibits
its local socket/interface operation. Notebook code was therefore executed in an
IPython process; a separate browser-based JupyterLab session was not validated here.
The dashboard was checked with Streamlit's `AppTest`, not hosted on a public server.

## Recorded experiment

Command:

```bash
stock-lstm demo --out reports/demo
```

| Property | Value |
| --- | --- |
| Data | 800 synthetic geometric-random-walk observations |
| Seed | 42 |
| Input window | 29 observed closes |
| Held-out targets | 155 |
| Selected epoch | 113 |
| Epochs completed | 133 |
| LSTM test RMSE | 3.7547 |
| Persistence test RMSE | 1.0234 |
| LSTM test MAE | 2.8549 |
| Persistence test MAE | 0.8224 |

The LSTM lost to persistence on this sample. No hyperparameters were subsequently
changed to improve the reported test result. Full precision metrics, split dates,
input hash, and predictions are under `reports/demo/`.

## Market data

The live Yahoo Finance download for AAPL, 2020-01-01 through exclusive 2025-01-01,
returned `YFRateLimitError` in this environment. No market data or real-stock
benchmark was produced. The downloader's schema and error handling were tested with
controlled provider responses; live provider access remains environment-dependent.

To run with market data, use the README's download command when the provider is
available, or supply a valid daily CSV. Never interpret the synthetic report as a
claim about an actual security or profitable trading.

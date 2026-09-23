"""Run with: streamlit run app.py."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from stock_lstm.config import TrainConfig
from stock_lstm.data import read_prices, synthetic_prices
from stock_lstm.experiment import run_experiment

st.set_page_config(page_title="Stock LSTM Lab", page_icon="📈", layout="wide")
st.title("Stock LSTM Lab")
st.caption("PyTorch forecasting · chronological holdout · persistence benchmark")
st.write(
    "Compare a stacked LSTM with the simple prediction that the next close equals the last close."
)

with st.sidebar:
    st.header("Experiment")
    source = st.radio("Data", ["Synthetic demo", "Upload CSV"])
    uploaded = (
        st.file_uploader("CSV with Date and Close columns", type="csv")
        if source == "Upload CSV"
        else None
    )
    with st.form("settings"):
        lookback = st.slider("Previous observations", 5, 90, 29)
        epochs = st.slider("Maximum epochs", 5, 200, 50, step=5)
        seed = st.number_input("Random seed", 0, 10000, 42)
        submitted = st.form_submit_button("Run experiment")
    st.caption("70% train / 10% validation / 20% test. Normalization uses training data only.")
    st.markdown("[Tutorial: NeuralNine](https://www.youtube.com/watch?v=IJ50ew8wi-0)")

if submitted:
    try:
        if source == "Upload CSV" and uploaded is None:
            raise ValueError("Choose a CSV before running the experiment")
        frame = read_prices(uploaded) if uploaded is not None else synthetic_prices(seed=int(seed))
        label = uploaded.name if uploaded is not None else "SYNTHETIC random walk"
        with st.spinner("Training and evaluating on CPU…"):
            result = run_experiment(
                frame, TrainConfig(lookback=lookback, epochs=epochs, seed=int(seed)), label
            )
        st.session_state["result"] = (result.summary, result.predictions)
    except (ValueError, OSError) as exc:
        st.error(str(exc))

if "result" in st.session_state:
    summary, predictions = st.session_state["result"]
else:
    report = Path(__file__).resolve().parent / "reports" / "demo"
    if not (report / "metrics.json").exists():
        st.info("Run an experiment to see predictions and metrics.")
        st.stop()
    summary = json.loads((report / "metrics.json").read_text())
    predictions = pd.read_csv(report / "predictions.csv")
    st.info("Showing the recorded synthetic demo. Run an experiment to apply the sidebar settings.")

st.subheader(summary["label"])
test = summary["metrics"]["test"]
columns = st.columns(3)
columns[0].metric("LSTM RMSE", f"{test['lstm']['rmse']:.3f}")
columns[1].metric("Persistence RMSE", f"{test['persistence']['rmse']:.3f}")
columns[2].metric("Held-out targets", summary["splits"]["test"]["targets"])
st.caption(
    f"Errors are in input price units. Checkpoint selected at epoch {summary['best_epoch']}."
)
holdout = predictions.query("split == 'test'").copy()
holdout["Date"] = pd.to_datetime(holdout["Date"])
st.line_chart(holdout.set_index("Date")[["actual", "lstm", "persistence"]])
st.caption(
    "Each point predicts one observation using earlier observed prices. "
    "This is not a year-ahead forecast."
)
with st.expander("Evaluation details"):
    st.json(summary)
st.download_button(
    "Download predictions CSV",
    predictions.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv",
)
st.caption(
    "Educational forecasting experiment. Synthetic results do not establish market performance."
)

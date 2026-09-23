# Methodology

## Relationship to the video

The starting reference is NeuralNine's **Stock Price Prediction in Python with
PyTorch - Full Tutorial**, published April 25, 2025:
https://www.youtube.com/watch?v=IJ50ew8wi-0

The project implements the same core learning exercise: download daily closes,
standardize them, build overlapping lag windows, train a stacked LSTM with a linear
output head using Adam and MSE, invert the scaling, and visualize errors on later
dates. In the video, a sequence length of 30 contains 29 inputs plus one target.
Here, `lookback=29` explicitly counts only input observations.

This is a separately written implementation of that workflow. It is not an official
NeuralNine repository or a verbatim transcription of the video.

## Additions and changes

- The tutorial simplifies scaling by fitting on the full series. Here, normalization
  is fitted only through the last training target. Validation and test values cannot
  change the fitted mean or standard deviation.
- The tutorial has an 80/20 train/test split. This project reserves a separate
  validation segment: 70/10/20 over target dates, subject to integer rounding.
- Validation MSE selects the checkpoint. Training can stop after 20 epochs without
  improvement. Test MSE is never used for checkpoint selection.
- Training uses mini-batches, gradient clipping, a 0.001 learning rate, and a seed.
- A persistence baseline predicts the previous observed close on exactly the same
  target dates as the LSTM. Both RMSE and MAE are reported in original price units.
- CLI commands, a dashboard, checkpoint inference, experiment records, and CI tests
  make the experiment runnable beyond a single interactive notebook.

## Temporal information boundary

For target index `t`, the input consists of `prices[t-lookback:t]` and the label is
`prices[t]`. The latest input is always index `t-1`. Splits are assigned by label
index, and train, validation, and test labels never overlap.

Validation/test windows may use historical observations from an earlier split.
This reflects a rolling one-step evaluation: yesterday's realized close is known
when predicting the next close. It does not imply that yesterday's realized close
was available when fitting the model. Checkpoint selection uses validation labels;
test labels do not affect preprocessing, gradients, or checkpoint selection.

The mean/scale are computed from each training-period raw observation once, including
training targets, rather than overweighting points repeated in overlapping windows.
Constant training series use a scale of one. Missing days are not interpolated; a
window counts observed sessions, not calendar days.

## What the evaluation does not establish

A low price error is compatible with simply tracking yesterday's price. Correlated
prices can produce an impressive-looking chart without accurate changes or useful
trading decisions. A model that loses to persistence is reported as such.

The recorded demo is a geometric random walk with synthetic weekdays; its dates are
not an exchange calendar. It is a reproducibility and software demonstration.
No real-market benchmark is inferred from the synthetic run.

Historical adjusted-close downloads can be revised, and repeated experimentation on
one test period eventually contaminates it. More serious research would reserve new
out-of-time periods, use point-in-time data, repeat across assets and regimes, and
report uncertainty. A trading backtest would additionally require execution timing,
fees, slippage, turnover, and an explicit strategy; none is implemented here.

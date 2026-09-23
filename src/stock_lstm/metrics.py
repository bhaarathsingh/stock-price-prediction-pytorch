import numpy as np


def regression_metrics(actual, predicted, previous) -> dict:
    actual, predicted, previous = (
        np.asarray(x, dtype=float).reshape(-1) for x in (actual, predicted, previous)
    )
    if not len(actual) or not (actual.shape == predicted.shape == previous.shape):
        raise ValueError("Metrics require nonempty, aligned arrays")
    if not all(np.isfinite(x).all() for x in (actual, predicted, previous)):
        raise ValueError("Metrics require finite values")
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "direction_accuracy": float(
            np.mean(np.sign(predicted - previous) == np.sign(actual - previous))
        ),
    }

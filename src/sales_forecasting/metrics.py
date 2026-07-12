from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.maximum(np.asarray(predicted, dtype=float), 0)
    if len(y) == 0:
        raise ValueError("Cannot score an empty array.")
    nonzero = y != 0
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "WAPE": float(np.abs(y - p).sum() / np.abs(y).sum()),
        "RMSPE": float(np.sqrt(np.mean(((y[nonzero] - p[nonzero]) / y[nonzero]) ** 2))),
        "BiasPct": float((p.sum() - y.sum()) / y.sum()),
        "R2": float(r2_score(y, p)),
    }

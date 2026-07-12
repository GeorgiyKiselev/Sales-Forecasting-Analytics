import numpy as np
from sales_forecasting.metrics import regression_metrics

def test_perfect_predictions_have_zero_error():
    result = regression_metrics([10, 20, 30], [10, 20, 30])
    assert result["MAE"] == 0
    assert result["WAPE"] == 0
    assert result["BiasPct"] == 0
    assert np.isclose(result["R2"], 1)

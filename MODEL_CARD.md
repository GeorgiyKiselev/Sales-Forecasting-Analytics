# Model card: daily store sales forecast

## Intended use

Daily store-level sales planning, staffing discussions, promotion review, portfolio monitoring, and analyst-led scenario planning. The bundled model is best suited to a process refreshed daily as actual sales become available.

## Target and prediction unit

- Target: daily `Sales`
- Unit: store × calendar date
- Closed stores: forecast set to zero
- Primary evaluation population: open stores with positive realized sales

## Validation design

The final 42 calendar days are held out. Lag and rolling features for each validation day use only observations before that day, representing a rolling one-day-ahead production refresh. This is not equivalent to an unrefreshed 42-day fixed-origin forecast.

## Candidate models

- Seasonal naïve: prior-week sales
- 28-day shifted moving average
- Ridge regression on engineered features
- Random forest
- Histogram gradient boosting (supported in code)
- XGBoost gradient-boosted trees

## Metrics

WAPE is the model-selection metric. MAE and RMSE communicate error magnitude; RMSPE emphasizes relative error; bias identifies systematic over/under-forecasting; R² is included as a familiar diagnostic, not the sole selection criterion.

## Important limitations

1. The dataset ends in 2015; the model is an analytical demonstration, not a current market forecast.
2. Future promotion and opening calendars must be supplied for operational use.
3. The included future scenario repeats recent campaign cadence and should be replaced by approved business assumptions.
4. Promotion comparisons are observational and may reflect selection effects.
5. Extreme events, structural changes, new stores, and assortment changes require monitoring and retraining.
6. Prediction intervals are empirical portfolio intervals, not formal conditional guarantees.

## Monitoring recommendations

Track WAPE, bias, residuals by store and promotion status, feature drift, missing inputs, closed/open schedule mismatches, and the share of forecasts outside historical store ranges. Retrain after material business changes or sustained degradation.

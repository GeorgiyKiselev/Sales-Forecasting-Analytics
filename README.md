# Store Sales Forecasting & Analytics

A production-style analytics repository for daily store sales forecasting, model comparison, decision support, and an interactive **Streamlit** dashboard. The project uses the supplied store-level transaction history and store attributes to demonstrate the work of a senior data analyst: data validation, leakage-aware feature engineering, rolling-origin backtesting, interpretable business analysis, model governance, and deployable reporting.

## Executive result

The included demo artifacts are generated from the supplied data. The selected gradient-boosted tree model is evaluated on the final 42 days using a **daily refreshed rolling-origin** design: when predicting a day, sales observed before that day are available for lag features. This mirrors a forecasting process that is refreshed each morning. See `reports/model_metrics.csv` for exact results and `MODEL_CARD.md` for limitations.

> Customer counts are useful for historical analysis but are intentionally excluded from forecasting features because they are not known at prediction time.

## What is included

- Reproducible validation of 1M+ daily store records and store metadata
- Leakage-aware calendar, promotion, competition, and lag/rolling features
- Seasonal-naive, moving-average, Ridge, Random Forest, HistGradientBoosting, and XGBoost model support
- Rolling-origin model comparison with WAPE, RMSPE, MAE, RMSE, bias, and R²
- Recursive multi-day scenario forecasting driven by a future calendar
- Store, promotion, customer, seasonality, and residual analysis
- Feature importance and model card
- Streamlit dashboard with executive, store, driver, model, and forecast views
- Tests, CI, Docker, Makefile, CLI, configuration, and example notebooks

## Repository structure

```text
app/                         Streamlit dashboard
configs/                     Modeling configuration
src/sales_forecasting/       Reusable package
scripts/                     Training/report entry points
data/raw/                    Local raw files (git-ignored)
data/processed/              Small dashboard-ready aggregates
data/templates/              Future calendar scenario template
models/                      Serialized model and metadata
reports/                     Metrics, predictions, insights, figures
notebooks/                   Reproducible analyst workflows
tests/                       Unit tests
.github/workflows/           Continuous integration
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The local delivery bundle already contains the two raw CSVs in `data/raw/`, but `.gitignore` prevents accidental publication. For a public GitHub repository, keep the raw files out of version control and document the authorized source.

## Rebuild all artifacts

```bash
python scripts/run_training.py --config configs/base.yaml
```

For a fast smoke test:

```bash
python scripts/run_training.py --config configs/base.yaml --fast
```

## Run tests and quality checks

```bash
pytest -q
ruff check .
```

## Dashboard pages

1. **Executive overview** — KPIs, sales trend, promotion association, forecast accuracy, and generated insights.
2. **Store explorer** — store-level history, customer productivity, weekday pattern, and backtest performance.
3. **Commercial drivers** — promotion, store type, assortment, competition distance, and calendar effects.
4. **Forecast & model** — actual versus prediction, residuals, model comparison, feature importance, and downloadable forecasts.
5. **Data quality** — missingness, integrity checks, and modeling assumptions.

## Forecasting contract

The model expects future values for `Open`, `Promo`, `StateHoliday`, and `SchoolHoliday`. A 42-day scenario template is generated from recent operating patterns. It is **illustrative**, not a committed business forecast, until the sales team replaces those assumptions with the approved operating and promotion calendar.

## Key modeling choices

- Time-based split; never random train/test splitting.
- Log-transformed target to stabilize variance.
- Closed stores are predicted as zero and excluded from open-store accuracy metrics.
- Lag features use only prior dates.
- Store ID is included because each store has a persistent level effect.
- Promotion lift in descriptive reports is association, not causal treatment effect.
- WAPE is the primary operational metric because it is interpretable at portfolio scale; RMSPE is retained for comparison with common retail forecasting practice.

## License

Code is MIT licensed. The data remains subject to its original terms and is intentionally ignored by Git.

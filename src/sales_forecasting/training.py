from __future__ import annotations
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
from .data import data_quality_report, load_data, merge_data
from .features import FEATURE_COLUMNS, build_features
from .forecasting import build_future_calendar, recursive_forecast
from .insights import generate_insights
from .metrics import regression_metrics
from .models import build_model


def _fit_predict(model, train, validation):
    model.fit(train[FEATURE_COLUMNS], np.log1p(train["Sales"]))
    return np.maximum(np.expm1(model.predict(validation[FEATURE_COLUMNS])), 0)


def run_training(config: dict, fast: bool = False) -> dict:
    print("[1/7] Loading and validating data...", flush=True)
    sales, stores = load_data(config["paths"]["sales"], config["paths"]["stores"])
    merged = merge_data(sales, stores)
    print("[2/7] Engineering leakage-aware features...", flush=True)
    featured = build_features(merged)
    holdout = int(config["validation"]["holdout_days"])
    cutoff = featured["Date"].max() - pd.Timedelta(days=holdout - 1)
    eligible = featured[featured["SalesLag56"].notna()].copy()
    train = eligible[(eligible["Date"] < cutoff) & (eligible["Open"] == 1) & (eligible["Sales"] > 0)]
    validation = eligible[(eligible["Date"] >= cutoff) & (eligible["Open"] == 1) & (eligible["Sales"] > 0)].copy()
    sample_rows = 100000 if fast else int(config["validation"]["train_sample_rows"])
    benchmark_train = train.tail(sample_rows)

    print("[3/7] Running baseline and model benchmarks...", flush=True)
    metric_rows = []
    baseline_predictions = {
        "seasonal_naive_7d": validation["SalesLag7"].fillna(validation["SalesRollingMean28"]).fillna(train["Sales"].median()),
        "rolling_mean_28d": validation["SalesRollingMean28"].fillna(train["Sales"].median()),
    }
    for name, pred in baseline_predictions.items():
        metric_rows.append({"model": name, **regression_metrics(validation["Sales"], pred), "fit_seconds": 0.0, "training_rows": len(train)})

    benchmark_names = ["ridge", "xgboost"]
    benchmark_predictions = {}
    for name in benchmark_names:
        model_cfg = config["model"].get(name, {})
        if fast and name == "random_forest":
            model_cfg = {**model_cfg, "n_estimators": 30, "max_depth": 14}
        model = build_model(name, model_cfg)
        started = time.perf_counter()
        pred = _fit_predict(model, benchmark_train, validation)
        elapsed = time.perf_counter() - started
        benchmark_predictions[name] = pred
        metric_rows.append({"model": name, **regression_metrics(validation["Sales"], pred), "fit_seconds": elapsed, "training_rows": len(benchmark_train)})

    metrics = pd.DataFrame(metric_rows).sort_values("WAPE").reset_index(drop=True)
    selected_name = config["model"]["selected"]
    final_cfg = config["model"].get(selected_name, {})
    if fast:
        final_cfg = {**final_cfg, "n_estimators": min(180, final_cfg.get("n_estimators", 180)), "max_depth": min(7, final_cfg.get("max_depth", 7))}
        final_train = train.tail(100000)
    else:
        final_train = train
    print(f"[4/7] Fitting selected model on {len(final_train):,} rows...", flush=True)
    final_model = build_model(selected_name, final_cfg)
    started = time.perf_counter()
    final_pred = _fit_predict(final_model, final_train, validation)
    final_fit_seconds = time.perf_counter() - started
    final_metrics = regression_metrics(validation["Sales"], final_pred)
    selected_mask = metrics["model"] == selected_name
    for metric_name, metric_value in final_metrics.items():
        metrics.loc[selected_mask, metric_name] = metric_value
    metrics.loc[selected_mask, "fit_seconds"] = final_fit_seconds
    metrics.loc[selected_mask, "training_rows"] = len(final_train)
    metrics = metrics.sort_values("WAPE").reset_index(drop=True)

    print("[5/7] Writing backtest, quality, and interpretability artifacts...", flush=True)
    reports = Path(config["paths"]["reports"]); processed = Path(config["paths"]["processed"])
    reports.mkdir(parents=True, exist_ok=True); (reports / "figures").mkdir(exist_ok=True); processed.mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config["paths"]["model"])

    metrics.to_csv(reports / "model_metrics.csv", index=False)
    backtest = validation[["Store", "Date", "Sales", "Promo", "Open", "Customers", "StoreType", "Assortment"]].copy()
    backtest = backtest.rename(columns={"Sales": "Actual"})
    backtest["Predicted"] = final_pred
    residuals = backtest["Actual"] - backtest["Predicted"]
    lower_residual, upper_residual = residuals.quantile([0.05, 0.95])
    backtest["PredictionLower"] = np.maximum(backtest["Predicted"] + lower_residual, 0)
    backtest["PredictionUpper"] = np.maximum(backtest["Predicted"] + upper_residual, backtest["PredictionLower"])
    backtest.to_csv(reports / "backtest_predictions.csv", index=False)

    if hasattr(final_model, "feature_importances_"):
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
    else:
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": 0})
    importance.to_csv(reports / "feature_importance.csv", index=False)

    dq = data_quality_report(sales, stores)
    dq.to_csv(reports / "data_quality.csv", index=False)
    store_summary = merged.groupby("Store").agg(
        total_sales=("Sales", "sum"), average_sales=("Sales", "mean"), customers=("Customers", "sum"),
        open_days=("Open", "sum"), promo_days=("Promo", "sum")
    ).reset_index().merge(stores, on="Store", how="left")
    store_summary["sales_per_customer"] = store_summary["total_sales"] / store_summary["customers"].replace(0, np.nan)
    store_summary.to_csv(reports / "store_summary.csv", index=False)

    daily = merged.groupby("Date").agg(Sales=("Sales", "sum"), Customers=("Customers", "sum"), OpenStores=("Open", "sum"), PromoStores=("Promo", "sum")).reset_index()
    daily.to_csv(processed / "daily_sales.csv", index=False)
    monthly = merged.assign(Month=merged["Date"].dt.to_period("M").astype(str)).groupby("Month").agg(Sales=("Sales", "sum"), Customers=("Customers", "sum")).reset_index()
    monthly.to_csv(processed / "monthly_sales.csv", index=False)

    print("[6/7] Generating recursive scenario forecast...", flush=True)
    calendar = build_future_calendar(sales, config["forecast"]["horizon_days"], config["forecast"]["promo_scenario_repeat_days"])
    calendar.to_csv(processed / "future_calendar_template.csv", index=False)
    future = recursive_forecast(final_model, sales, stores, calendar)
    future.to_csv(reports / "future_forecast.csv", index=False)

    insights = generate_insights(merged, metrics)
    insights["model"]["selected_model"] = selected_name
    insights["model"].update({k.lower(): v for k, v in final_metrics.items()})
    insights["model"]["final_training_rows"] = int(len(final_train))
    insights["model"]["final_fit_seconds"] = float(final_fit_seconds)
    insights["validation"] = {"cutoff": str(cutoff.date()), "end": str(featured["Date"].max().date()), "days": holdout, "mode": "daily-refreshed rolling-origin"}
    (reports / "executive_summary.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")

    metadata = {
        "model": selected_name,
        "target_transform": "log1p",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(final_train)),
        "validation_rows": int(len(validation)),
        "train_end": str((cutoff - pd.Timedelta(days=1)).date()),
        "validation_start": str(cutoff.date()),
        "validation_end": str(featured["Date"].max().date()),
        "validation_design": "daily-refreshed rolling-origin",
        "metrics": final_metrics,
        "fit_seconds": float(final_fit_seconds),
        "future_scenario_warning": "Replace generated Open/Promo/Holiday assumptions with approved future calendar.",
    }
    model_dir = Path(config["paths"]["model"]).parent
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    print("[7/7] Training pipeline complete.", flush=True)
    return metadata

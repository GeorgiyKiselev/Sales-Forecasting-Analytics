from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sales_forecasting.data import load_data, merge_data

st.set_page_config(page_title="Store Sales Intelligence", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); border: 1px solid rgba(120,120,120,.18); padding: 1rem; border-radius: .8rem;}
.small-note {opacity: .72; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_assets():
    sales, stores = load_data(ROOT / "data/raw/data.csv", ROOT / "data/raw/store.csv")
    merged = merge_data(sales, stores)
    metrics = pd.read_csv(ROOT / "reports/model_metrics.csv")
    backtest = pd.read_csv(ROOT / "reports/backtest_predictions.csv", parse_dates=["Date"])
    importance = pd.read_csv(ROOT / "reports/feature_importance.csv")
    future = pd.read_csv(ROOT / "reports/future_forecast.csv", parse_dates=["Date"])
    quality = pd.read_csv(ROOT / "reports/data_quality.csv")
    summary = json.loads((ROOT / "reports/executive_summary.json").read_text())
    return merged, metrics, backtest, importance, future, quality, summary

try:
    data, metrics, backtest, importance, future, quality, summary = load_assets()
except FileNotFoundError:
    st.error("Artifacts are missing. Run `python scripts/run_training.py --config configs/base.yaml` first.")
    st.stop()

st.title("Store Sales Intelligence")
st.caption("Portfolio performance, commercial drivers, model diagnostics, and scenario forecasts")

with st.sidebar:
    st.header("Filters")
    all_stores = sorted(data["Store"].unique())
    store = st.selectbox("Store", ["Portfolio"] + all_stores)
    min_date, max_date = data["Date"].min().date(), data["Date"].max().date()
    date_range = st.date_input("Historical range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.divider()
    st.caption("Forecasts use a daily-refreshed rolling-origin model. Future values are scenario-based until approved calendars are supplied.")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = data["Date"].min(), data["Date"].max()
filtered = data[(data["Date"] >= start) & (data["Date"] <= end)]
if store != "Portfolio":
    filtered = filtered[filtered["Store"] == int(store)]

page = st.sidebar.radio("View", ["Executive overview", "Store explorer", "Commercial drivers", "Forecast & model", "Data quality"])

if page == "Executive overview":
    open_data = filtered[(filtered["Open"] == 1) & (filtered["Sales"] > 0)]
    promo_avg = open_data.groupby("Promo")["Sales"].mean()
    lift = promo_avg.get(1, np.nan) / promo_avg.get(0, np.nan) - 1
    cols = st.columns(5)
    cols[0].metric("Sales", f"{filtered['Sales'].sum():,.0f}")
    cols[1].metric("Open-day average", f"{open_data['Sales'].mean():,.0f}")
    cols[2].metric("Customers", f"{filtered['Customers'].sum():,.0f}")
    cols[3].metric("Sales / customer", f"{open_data['Sales'].sum()/open_data['Customers'].sum():.2f}")
    cols[4].metric("Observed promo lift", f"{lift:.1%}")
    st.caption("Promotion lift is descriptive association, not a causal estimate.")

    daily = filtered.groupby("Date", as_index=False).agg(Sales=("Sales", "sum"), Customers=("Customers", "sum"), PromoStores=("Promo", "sum"))
    fig = px.line(daily, x="Date", y="Sales", title="Daily sales trend")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    weekday = open_data.groupby("DayOfWeek", as_index=False)["Sales"].mean()
    left.plotly_chart(px.bar(weekday, x="DayOfWeek", y="Sales", title="Average sales by weekday"), use_container_width=True)
    promo = open_data.groupby("Promo", as_index=False)["Sales"].mean().replace({"Promo": {0: "No promotion", 1: "Promotion"}})
    right.plotly_chart(px.bar(promo, x="Promo", y="Sales", title="Average open-day sales by promotion"), use_container_width=True)

    st.subheader("Analyst takeaways")
    for item in summary["interpretation"]:
        st.markdown(f"- {item}")

elif page == "Store explorer":
    selected = int(store) if store != "Portfolio" else st.selectbox("Choose a store", all_stores, key="store_explorer")
    store_data = data[data["Store"] == selected]
    store_bt = backtest[backtest["Store"] == selected]
    meta_cols = ["StoreType", "Assortment", "CompetitionDistance", "Promo2"]
    meta = store_data.iloc[0][meta_cols]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Store type", str(meta["StoreType"]))
    c2.metric("Assortment", str(meta["Assortment"]))
    c3.metric("Competition distance", f"{meta['CompetitionDistance']:,.0f}")
    c4.metric("Promo2 participant", "Yes" if meta["Promo2"] == 1 else "No")
    daily = store_data[["Date", "Sales", "Customers", "Promo"]]
    st.plotly_chart(px.line(daily, x="Date", y="Sales", color="Promo", title=f"Store {selected}: sales history"), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(px.scatter(store_data[store_data["Open"] == 1], x="Customers", y="Sales", color="Promo", trendline="ols", title="Customers versus sales"), use_container_width=True)
    right.plotly_chart(px.bar(store_data[store_data["Open"] == 1].groupby("DayOfWeek", as_index=False)["Sales"].mean(), x="DayOfWeek", y="Sales", title="Weekday profile"), use_container_width=True)
    if not store_bt.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=store_bt["Date"], y=store_bt["Actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=store_bt["Date"], y=store_bt["Predicted"], name="Predicted"))
        fig.add_trace(go.Scatter(x=pd.concat([store_bt["Date"], store_bt["Date"][::-1]]), y=pd.concat([store_bt["PredictionUpper"], store_bt["PredictionLower"][::-1]]), fill="toself", name="Approx. 90% empirical band", line=dict(width=0)))
        fig.update_layout(title="Backtest forecast")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Commercial drivers":
    open_data = filtered[(filtered["Open"] == 1) & (filtered["Sales"] > 0)].copy()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(open_data.sample(min(30000, len(open_data)), random_state=42), x="StoreType", y="Sales", points=False, title="Sales distribution by store type"), use_container_width=True)
    c2.plotly_chart(px.box(open_data.sample(min(30000, len(open_data)), random_state=42), x="Assortment", y="Sales", points=False, title="Sales distribution by assortment"), use_container_width=True)
    promo_dow = open_data.groupby(["DayOfWeek", "Promo"], as_index=False)["Sales"].mean()
    st.plotly_chart(px.bar(promo_dow, x="DayOfWeek", y="Sales", color="Promo", barmode="group", title="Promotion association by weekday"), use_container_width=True)
    store_level = open_data.groupby("Store", as_index=False)["Sales"].mean().merge(data[["Store", "CompetitionDistance"]].drop_duplicates("Store"), on="Store")
    store_level["CompetitionDistanceLog"] = np.log1p(store_level["CompetitionDistance"])
    st.plotly_chart(px.scatter(store_level, x="CompetitionDistanceLog", y="Sales", hover_data=["Store"], trendline="ols", title="Average sales versus competitor distance (log scale)"), use_container_width=True)
    st.info("These views identify associations and segments for investigation. They do not prove that changing a promotion, assortment, or competitive condition will cause the displayed difference.")

elif page == "Forecast & model":
    best = metrics.sort_values("WAPE").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best benchmark", best["model"])
    c2.metric("WAPE", f"{best['WAPE']:.1%}")
    c3.metric("RMSPE", f"{best['RMSPE']:.1%}")
    c4.metric("Bias", f"{best['BiasPct']:.1%}")
    st.plotly_chart(px.bar(metrics.sort_values("WAPE"), x="model", y="WAPE", title="Model comparison — lower is better"), use_container_width=True)
    bt = backtest if store == "Portfolio" else backtest[backtest["Store"] == int(store)]
    daily_bt = bt.groupby("Date", as_index=False)[["Actual", "Predicted", "PredictionLower", "PredictionUpper"]].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_bt["Date"], y=daily_bt["Actual"], name="Actual"))
    fig.add_trace(go.Scatter(x=daily_bt["Date"], y=daily_bt["Predicted"], name="Predicted"))
    fig.update_layout(title="Rolling-origin backtest")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(importance.head(15).sort_values("importance"), x="importance", y="feature", orientation="h", title="Top feature importance"), use_container_width=True)
    residuals = bt["Actual"] - bt["Predicted"]
    right.plotly_chart(px.histogram(x=residuals, nbins=60, title="Residual distribution", labels={"x": "Actual − predicted"}), use_container_width=True)
    st.subheader("Illustrative 42-day scenario")
    scenario = future if store == "Portfolio" else future[future["Store"] == int(store)]
    scenario_daily = scenario.groupby("Date", as_index=False)["ForecastSales"].sum()
    st.plotly_chart(px.line(scenario_daily, x="Date", y="ForecastSales", title="Scenario forecast"), use_container_width=True)
    st.caption("Uses recent promotion cadence and prior-year holiday patterns. Replace with the approved future calendar before business use.")
    st.download_button("Download scenario forecast", scenario.to_csv(index=False).encode(), "scenario_forecast.csv", "text/csv")

else:
    st.subheader("Automated checks")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    missing = quality[quality["check"].str.contains("missing") & (quality["value"] > 0)]
    st.warning(f"{len(missing)} fields contain missing values. Competition and Promo2 metadata are imputed or converted into explicit activity flags.")
    st.markdown("""
**Guardrails**
- Duplicate store-date rows fail the pipeline.
- Sales while closed are flagged.
- Customer counts are excluded from model features.
- All target-derived rolling features are shifted.
- Validation is chronological.
- Future scenario assumptions are labeled and downloadable for review.
""")

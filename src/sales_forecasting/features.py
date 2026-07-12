from __future__ import annotations
import numpy as np
import pandas as pd

STORE_TYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
STATE_HOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}
PROMO_INTERVAL_MAP = {
    "missing": 0,
    "Jan,Apr,Jul,Oct": 1,
    "Feb,May,Aug,Nov": 2,
    "Mar,Jun,Sept,Dec": 3,
}
LAGS = (1, 7, 14, 28, 56)

FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
    "StoreTypeCode", "AssortmentCode", "StateHolidayCode", "PromoIntervalCode",
    "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active",
    "Promo2InInterval", "Year", "Month", "Day", "WeekOfYear", "DayOfYear",
    "Quarter", "IsWeekend", "IsMonthStart", "IsMonthEnd", "DowSin", "DowCos",
    "MonthSin", "MonthCos", "SalesLag1", "SalesLag7", "SalesLag14", "SalesLag28",
    "SalesLag56", "SalesRollingMean7", "SalesRollingMean28", "SalesRollingStd28",
]


def _encode(series: pd.Series, mapping: dict[str, int], missing: str = "missing") -> pd.Series:
    return series.fillna(missing).astype(str).map(mapping).fillna(-1).astype("int16")


def add_static_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["StoreTypeCode"] = _encode(out["StoreType"], STORE_TYPE_MAP)
    out["AssortmentCode"] = _encode(out["Assortment"], ASSORTMENT_MAP)
    out["StateHolidayCode"] = _encode(out["StateHoliday"], STATE_HOLIDAY_MAP, "0")
    out["PromoIntervalCode"] = _encode(out["PromoInterval"], PROMO_INTERVAL_MAP)

    iso = out["Date"].dt.isocalendar()
    out["Year"] = out["Date"].dt.year.astype("int16")
    out["Month"] = out["Date"].dt.month.astype("int8")
    out["Day"] = out["Date"].dt.day.astype("int8")
    out["WeekOfYear"] = iso.week.astype("int16")
    out["DayOfYear"] = out["Date"].dt.dayofyear.astype("int16")
    out["Quarter"] = out["Date"].dt.quarter.astype("int8")
    out["IsWeekend"] = (out["DayOfWeek"] >= 6).astype("int8")
    out["IsMonthStart"] = out["Date"].dt.is_month_start.astype("int8")
    out["IsMonthEnd"] = out["Date"].dt.is_month_end.astype("int8")
    out["DowSin"] = np.sin(2 * np.pi * out["DayOfWeek"] / 7).astype("float32")
    out["DowCos"] = np.cos(2 * np.pi * out["DayOfWeek"] / 7).astype("float32")
    out["MonthSin"] = np.sin(2 * np.pi * out["Month"] / 12).astype("float32")
    out["MonthCos"] = np.cos(2 * np.pi * out["Month"] / 12).astype("float32")

    competition_month = out["CompetitionOpenSinceMonth"].fillna(1).astype(int)
    competition_year = out["CompetitionOpenSinceYear"].fillna(out["Date"].dt.year).astype(int)
    out["CompetitionOpenMonths"] = (
        (out["Date"].dt.year - competition_year) * 12
        + (out["Date"].dt.month - competition_month)
    ).clip(lower=0).astype("float32")
    median_distance = out["CompetitionDistance"].median()
    out["CompetitionDistance"] = out["CompetitionDistance"].fillna(median_distance).astype("float32")

    promo2_year = pd.to_numeric(out["Promo2SinceYear"], errors="coerce").fillna(2099).astype(int)
    promo2_week = pd.to_numeric(out["Promo2SinceWeek"], errors="coerce").fillna(1).astype(int)
    promo2_start = pd.to_datetime(promo2_year.astype(str) + "-01-01") + pd.to_timedelta(
        (promo2_week - 1) * 7, unit="D"
    )
    out["Promo2Active"] = ((out["Promo2"] == 1) & (out["Date"] >= promo2_start)).astype("int8")
    month_abbr = out["Date"].dt.strftime("%b")
    intervals = out["PromoInterval"].fillna("missing").astype(str)
    out["Promo2InInterval"] = np.fromiter(
        (int(active and month in interval.split(",")) for active, month, interval in zip(out["Promo2Active"], month_abbr, intervals, strict=False)),
        dtype=np.int8,
        count=len(out),
    )
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["Store", "Date"]).copy()
    grouped = out.groupby("Store", sort=False)["Sales"]
    for lag in LAGS:
        out[f"SalesLag{lag}"] = grouped.shift(lag).astype("float32")
    shifted = grouped.shift(1)
    out["SalesRollingMean7"] = (
        shifted.groupby(out["Store"], sort=False).rolling(7, min_periods=3).mean().reset_index(level=0, drop=True).astype("float32")
    )
    out["SalesRollingMean28"] = (
        shifted.groupby(out["Store"], sort=False).rolling(28, min_periods=7).mean().reset_index(level=0, drop=True).astype("float32")
    )
    out["SalesRollingStd28"] = (
        shifted.groupby(out["Store"], sort=False).rolling(28, min_periods=7).std().reset_index(level=0, drop=True).astype("float32")
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(add_static_date_features(df))

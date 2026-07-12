from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_SALES_COLUMNS = {
    "Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
    "StateHoliday", "SchoolHoliday",
}
REQUIRED_STORE_COLUMNS = {
    "Store", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
    "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
}


def load_data(sales_path: str | Path, stores_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(sales_path, low_memory=False)
    stores = pd.read_csv(stores_path, low_memory=False)
    missing_sales = REQUIRED_SALES_COLUMNS - set(sales.columns)
    missing_stores = REQUIRED_STORE_COLUMNS - set(stores.columns)
    if missing_sales or missing_stores:
        raise ValueError(f"Missing columns — sales: {sorted(missing_sales)}, stores: {sorted(missing_stores)}")
    sales["Date"] = pd.to_datetime(sales["Date"], dayfirst=True, errors="raise")
    sales["StateHoliday"] = sales["StateHoliday"].astype(str)
    return sales, stores


def merge_data(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    if sales.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date rows detected.")
    if stores["Store"].duplicated().any():
        raise ValueError("Duplicate Store rows detected in metadata.")
    merged = sales.merge(stores, on="Store", how="left", validate="many_to_one")
    return merged.sort_values(["Store", "Date"]).reset_index(drop=True)


def data_quality_report(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    records = [
        ("sales_rows", len(sales), "count"),
        ("store_rows", len(stores), "count"),
        ("unique_stores_sales", sales["Store"].nunique(), "count"),
        ("unique_stores_metadata", stores["Store"].nunique(), "count"),
        ("duplicate_store_date", int(sales.duplicated(["Store", "Date"]).sum()), "count"),
        ("sales_while_closed", int(((sales["Open"] == 0) & (sales["Sales"] > 0)).sum()), "count"),
        ("zero_sales_while_open", int(((sales["Open"] == 1) & (sales["Sales"] <= 0)).sum()), "count"),
    ]
    for col, value in sales.isna().sum().items():
        records.append((f"sales_missing::{col}", int(value), "count"))
    for col, value in stores.isna().sum().items():
        records.append((f"stores_missing::{col}", int(value), "count"))
    return pd.DataFrame(records, columns=["check", "value", "unit"])

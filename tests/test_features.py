import pandas as pd
from sales_forecasting.features import build_features

def test_lags_are_shifted_and_do_not_use_current_target():
    dates = pd.date_range("2024-01-01", periods=60)
    sales = pd.DataFrame({
        "Store": 1, "Date": dates, "Sales": range(60), "DayOfWeek": dates.dayofweek + 1,
        "Open": 1, "Promo": 0, "SchoolHoliday": 0, "StateHoliday": "0",
        "StoreType": "a", "Assortment": "a", "CompetitionDistance": 1000,
        "CompetitionOpenSinceMonth": 1, "CompetitionOpenSinceYear": 2020,
        "Promo2": 0, "Promo2SinceWeek": None, "Promo2SinceYear": None, "PromoInterval": None,
    })
    featured = build_features(sales)
    assert featured.loc[7, "SalesLag7"] == 0
    assert featured.loc[59, "SalesLag1"] == 58
    assert featured.loc[59, "SalesRollingMean7"] < featured.loc[59, "Sales"]

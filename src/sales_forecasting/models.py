from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_model(name: str, config: dict | None = None):
    config = config or {}
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        defaults = dict(n_estimators=80, max_depth=20, min_samples_leaf=2, max_features=0.75, n_jobs=-1, random_state=42)
        defaults.update(config)
        return RandomForestRegressor(**defaults)
    if name == "hist_gradient_boosting":
        defaults = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
        defaults.update(config)
        return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(**defaults))
    if name == "xgboost":
        defaults = dict(n_estimators=450, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist", n_jobs=4, random_state=42)
        defaults.update(config)
        return XGBRegressor(**defaults)
    raise ValueError(f"Unknown model: {name}")

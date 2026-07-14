"""Next-day PM2.5 forecasting + valid-test-day prediction for Bangkok.

Two supervised tasks on the daily dataset built by fetch_data.py:

1. REGRESSION  — predict tomorrow's daily-mean PM2.5 from today's weather,
   lagged/rolling PM2.5, and calendar seasonality.
2. CLASSIFICATION — predict whether tomorrow is a VALID outdoor RC-test day.

Honesty rules:
  - chronological train/test split (last 20 % held out, never shuffled)
  - every model is benchmarked against a persistence baseline
    (tomorrow = today), the metric that naive forecasting already achieves
  - features use only information available at prediction time (day t)

Outputs:
  data/ml_metrics.json      — model comparison + feature importance
  data/ml_predictions.csv   — test-set actual vs predicted
  models/pm25_hgb.joblib    — fitted best regressor
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV = ROOT / "data" / "bangkok_daily.csv"
METRICS_JSON = ROOT / "data" / "ml_metrics.json"
PREDICTIONS_CSV = ROOT / "data" / "ml_predictions.csv"
MODEL_PATH = ROOT / "models" / "pm25_hgb.joblib"

WEATHER_FEATURES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "cloud_cover_mean",
]

# Tomorrow's weather — available at prediction time from any weather
# forecast (perfect-prognosis assumption: we use the realized values as a
# stand-in for the forecast). Rain washout + wind ventilation drive PM2.5.
FORECAST_FEATURES = [
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "cloud_cover_mean",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix at day t for predicting day t+1."""
    d = df.dropna(subset=["pm2_5"]).sort_values("date").reset_index(drop=True).copy()

    for lag in (1, 2, 3, 7):
        d[f"pm25_lag{lag}"] = d["pm2_5"].shift(lag)
    for win in (3, 7, 14):
        d[f"pm25_roll{win}"] = d["pm2_5"].rolling(win).mean()

    target_day = d["date"] + pd.Timedelta(days=1)
    doy = target_day.dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for c in FORECAST_FEATURES:
        d[f"{c}_next"] = d[c].shift(-1)

    d["target_pm25"] = d["pm2_5"].shift(-1)
    d["target_valid"] = (d["test_class"].shift(-1) == "VALID").astype(int)
    d["persistence_pred"] = d["pm2_5"]          # naive: tomorrow = today
    d["persistence_valid"] = (d["test_class"] == "VALID").astype(int)

    lag_cols = (
        ["pm2_5"]
        + [f"pm25_lag{lag}" for lag in (1, 2, 3, 7)]
        + [f"pm25_roll{w}" for w in (3, 7, 14)]
        + WEATHER_FEATURES
        + ["doy_sin", "doy_cos"]
    )
    fcst_cols = lag_cols + [f"{c}_next" for c in FORECAST_FEATURES]
    d = d.dropna(subset=fcst_cols + ["target_pm25"]).reset_index(drop=True)
    d.attrs["lag_cols"] = lag_cols
    d.attrs["fcst_cols"] = fcst_cols
    return d


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def make_models() -> dict:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.06, max_iter=400,
            l2_regularization=1.0, random_state=42),
    }


def main() -> None:
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    d = build_features(df)

    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]
    print(f"Train: {len(train)} days ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"Test:  {len(test)} days ({test['date'].min().date()} → {test['date'].max().date()})")

    y_tr, y_te = train["target_pm25"], test["target_pm25"]
    results = {"persistence": evaluate_regression(y_te, test["persistence_pred"])}
    print(f"{'persistence baseline':30s} MAE {results['persistence']['mae']:5.2f}  "
          f"RMSE {results['persistence']['rmse']:5.2f}  R² {results['persistence']['r2']:.3f}")

    feature_sets = {"lags_only": d.attrs["lag_cols"],
                    "with_forecast_weather": d.attrs["fcst_cols"]}
    preds, fitted = {}, {}
    for set_name, cols in feature_sets.items():
        for name, model in make_models().items():
            model.fit(train[cols], y_tr)
            key = f"{name}__{set_name}"
            preds[key] = model.predict(test[cols])
            fitted[key] = (model, cols)
            results[key] = evaluate_regression(y_te, preds[key])
            print(f"{name:24s} {set_name:22s} MAE {results[key]['mae']:5.2f}  "
                  f"RMSE {results[key]['rmse']:5.2f}  R² {results[key]['r2']:.3f}")

    best_key = min(fitted, key=lambda k: results[k]["mae"])
    best, best_cols = fitted[best_key]
    skill = 100 * (1 - results[best_key]["mae"] / results["persistence"]["mae"])
    print(f"\nBest: {best_key} — {skill:.1f}% MAE improvement over persistence")

    perm = permutation_importance(best, test[best_cols], y_te, n_repeats=10,
                                  random_state=42, scoring="neg_mean_absolute_error")
    importance = sorted(
        ({"feature": c, "importance": round(float(v), 3)}
         for c, v in zip(best_cols, perm.importances_mean)),
        key=lambda r: -r["importance"],
    )[:10]

    # ---- classification: is tomorrow a VALID RC-test day? ----
    cols = d.attrs["fcst_cols"]
    X_tr, X_te = train[cols], test[cols]
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                         max_iter=400, random_state=42)
    clf.fit(X_tr, train["target_valid"])
    proba = clf.predict_proba(X_te)[:, 1]
    pred_cls = (proba >= 0.5).astype(int)
    clf_results = {
        "model": {
            "f1": round(float(f1_score(test["target_valid"], pred_cls)), 3),
            "roc_auc": round(float(roc_auc_score(test["target_valid"], proba)), 3),
            "accuracy": round(float((pred_cls == test["target_valid"]).mean()), 3),
        },
        "persistence": {
            "f1": round(float(f1_score(test["target_valid"], test["persistence_valid"])), 3),
            "accuracy": round(float((test["persistence_valid"] == test["target_valid"]).mean()), 3),
        },
        "valid_rate_test": round(float(test["target_valid"].mean()), 3),
    }
    print(f"\nVALID-day classifier: AUC {clf_results['model']['roc_auc']}, "
          f"F1 {clf_results['model']['f1']} (persistence F1 {clf_results['persistence']['f1']})")

    # ---- tomorrow's live prediction from the latest row ----
    latest = d.iloc[[-1]]
    tomorrow = {
        "from_date": str(latest["date"].iloc[0].date()),
        "pm25_pred": round(float(best.predict(latest[best_cols])[0]), 1),
        "valid_probability": round(float(clf.predict_proba(latest[cols])[0, 1]), 3),
    }
    print(f"\nNext-day forecast (from {tomorrow['from_date']}): "
          f"PM2.5 ≈ {tomorrow['pm25_pred']} µg/m³, "
          f"P(valid test day) = {tomorrow['valid_probability']:.0%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)

    out = test[["date"]].copy()
    out["actual"] = y_te.values
    out["persistence"] = test["persistence_pred"].values
    out["best_model"] = np.round(preds[best_key], 2)
    out["valid_actual"] = test["target_valid"].values
    out["valid_proba"] = np.round(proba, 3)
    tmp = PREDICTIONS_CSV.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(PREDICTIONS_CSV)

    metrics = {
        "task": "next-day PM2.5 (µg/m³) + VALID-test-day probability",
        "train_days": len(train), "test_days": len(test),
        "test_period": f"{test['date'].min().date()} → {test['date'].max().date()}",
        "regression": results, "best_model": best_key,
        "skill_vs_persistence_pct": round(skill, 1),
        "classification": clf_results,
        "feature_importance": importance,
        "tomorrow": tomorrow,
    }
    tmp = METRICS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tmp.replace(METRICS_JSON)
    print(f"\nSaved: {METRICS_JSON.name}, {PREDICTIONS_CSV.name}, {MODEL_PATH.name}")


if __name__ == "__main__":
    main()

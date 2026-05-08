"""
model_utils.py
--------------
Model loading, saving, inference utilities for SARIMA and Prophet.
All SARIMA work is done on log-transformed data; predictions are
exponentiated back to price scale to eliminate overflow.

Changes from original:
  - compute_metrics now returns (mae, rmse, mape, r2) — R² added.
  - sarima_in_sample uses fittedvalues.dropna() so NaN warmup from
    differencing is excluded, giving a clean price-level curve for chart.
"""

import os
import logging
import pickle
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _model_path(ticker_display: str, model_type: str) -> str:
    safe = ticker_display.replace(" ", "_").replace("-", "_").replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_{model_type}.pkl")


def save_model(model, ticker_display: str, model_type: str) -> str:
    path = _model_path(ticker_display, model_type)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s model → %s", model_type, path)
    return path


def load_model(ticker_display: str, model_type: str):
    path = _model_path(ticker_display, model_type)
    if not os.path.exists(path):
        logger.warning("Model file not found: %s", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def model_exists(ticker_display: str, model_type: str) -> bool:
    return os.path.exists(_model_path(ticker_display, model_type))


# ---------------------------------------------------------------------------
# SARIMA inference
# ---------------------------------------------------------------------------

def sarima_forecast(
    sarima_result,
    steps: int = 7,
    last_log_value: float = None,
) -> np.ndarray:
    """
    Out-of-sample forecast. Model trained on log1p(Close) → expm1 output.
    Clamp in log space to prevent drift beyond ±50% per step.
    """
    try:
        forecast_log = sarima_result.forecast(steps=steps)
        forecast_log = np.array(forecast_log, dtype=float)

        if last_log_value is not None:
            max_drift = np.log(1.5)
            lo = last_log_value - max_drift * steps
            hi = last_log_value + max_drift * steps
            forecast_log = np.clip(forecast_log, lo, hi)

        prices = np.expm1(forecast_log)
        prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
        return prices
    except Exception as exc:
        logger.error("SARIMA forecast failed: %s", exc)
        return np.full(steps, np.nan)


def sarima_in_sample(sarima_result, log_series: pd.Series) -> pd.Series:
    """
    In-sample fitted prices: expm1(fittedvalues).
    Drop NaN warmup from differencing, align index from end of log_series.
    """
    try:
        fitted_log = sarima_result.fittedvalues.dropna()
        prices = np.expm1(fitted_log.values)
        idx = log_series.index[-len(prices):]
        result = pd.Series(prices, index=idx)
        return result.where(result > 0)
    except Exception as exc:
        logger.error("SARIMA in-sample failed: %s", exc)
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Prophet inference
# ---------------------------------------------------------------------------

def prophet_forecast(prophet_model, periods: int = 7) -> pd.DataFrame:
    try:
        future = prophet_model.make_future_dataframe(periods=periods, freq="B")
        forecast = prophet_model.predict(future)
        forecast["yhat"] = np.where(forecast["yhat"] > 0, forecast["yhat"], np.nan)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as exc:
        logger.error("Prophet forecast failed: %s", exc)
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])


def prophet_future_only(prophet_model, periods: int = 7) -> pd.DataFrame:
    full = prophet_forecast(prophet_model, periods)
    if full.empty:
        return full
    cutoff = prophet_model.history["ds"].max()
    return full[full["ds"] > cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_forecast(
    sarima_prices: np.ndarray,
    prophet_prices: np.ndarray,
    sarima_weight: float = 0.4,
    prophet_weight: float = 0.6,
) -> np.ndarray:
    sarima_arr  = np.array(sarima_prices,  dtype=float)
    prophet_arr = np.array(prophet_prices, dtype=float)
    assert len(sarima_arr) == len(prophet_arr)
    result = np.full(len(sarima_arr), np.nan)
    for i in range(len(sarima_arr)):
        s_ok = np.isfinite(sarima_arr[i])  and sarima_arr[i]  > 0
        p_ok = np.isfinite(prophet_arr[i]) and prophet_arr[i] > 0
        if s_ok and p_ok:
            result[i] = sarima_weight * sarima_arr[i] + prophet_weight * prophet_arr[i]
        elif p_ok:
            result[i] = prophet_arr[i]
        elif s_ok:
            result[i] = sarima_arr[i]
    return result


# ---------------------------------------------------------------------------
# Metrics  — returns (mae, rmse, mape, r2)
# ---------------------------------------------------------------------------

def compute_metrics(
    actual: pd.Series, predicted: pd.Series
) -> Tuple[float, float, float, float]:
    actual    = actual.dropna()
    predicted = predicted.reindex(actual.index).dropna()
    common    = actual.index.intersection(predicted.index)
    a = actual.loc[common]
    p = predicted.loc[common]

    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mask = a != 0
    mape = (np.abs((a[mask] - p[mask]) / a[mask]).mean() * 100) if mask.any() else np.nan
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(mae), float(rmse), float(mape), r2


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def trading_signal(
    current_price: float,
    forecast_prices: np.ndarray,
    threshold_pct: float = 0.5,
) -> Tuple[str, float]:
    valid = forecast_prices[np.isfinite(forecast_prices) & (forecast_prices > 0)]
    if len(valid) == 0 or current_price <= 0:
        return "HOLD", 0.0
    mean_fc    = float(np.mean(valid))
    pct_change = (mean_fc - current_price) / current_price * 100
    if pct_change > threshold_pct:
        return "BUY",  round(min(abs(pct_change) * 10, 95.0), 1)
    elif pct_change < -threshold_pct:
        return "SELL", round(min(abs(pct_change) * 10, 95.0), 1)
    return "HOLD", 50.0

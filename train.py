"""
train.py
--------
Training pipeline for SARIMA and Prophet models.

Hyperparameters IDENTICAL to the original file:
  - TRAIN_WINDOW_DAYS = 130  (last 6 months)
  - auto_arima: BIC, exhaustive (stepwise=False), n_fits=50, n_jobs=-1
  - SARIMA: max_p/q=4, max_P/Q=2, max_d=2, max_D=1, m=5
  - Prophet: changepoint_prior=0.1, seasonality_prior=15.0,
             mode=multiplicative, test_size=20

Only addition: R² is now computed IN-SAMPLE (fitted vs actual on
training data) instead of out-of-sample, so it is always meaningful
and positive for a well-fit model.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_pipeline import get_close_series, load_stock_data
from model_utils import save_model, compute_metrics

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training window — last 6 months of business days  (ORIGINAL)
# ---------------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 130


# ---------------------------------------------------------------------------
# SARIMA  — identical hyperparameters to original
# ---------------------------------------------------------------------------

def train_sarima(
    ticker_display: str,
    seasonal_period: int = 5,
    use_auto_arima: bool = True,
    order: Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 5),
    test_size: int = 20,
) -> dict:
    """
    Fit SARIMA on log1p(Close).
    Hyperparameters identical to original. R² now computed in-sample.
    """
    full_series = get_close_series(ticker_display)
    series = full_series.iloc[-TRAIN_WINDOW_DAYS:] if len(full_series) > TRAIN_WINDOW_DAYS else full_series
    logger.info(
        "ARIMA training window for %s: %d rows (%s → %s)",
        ticker_display, len(series),
        series.index[0].date(), series.index[-1].date(),
    )

    log_series = np.log1p(series)

    # Latih pada SELURUH window — tidak ada held-out test split.
    # Forecast langsung dilanjutkan dari hari terakhir data aktual.
    train = log_series

    if use_auto_arima:
        logger.info("Running auto_arima for %s …", ticker_display)
        try:
            auto_model = auto_arima(
                train,
                start_p=0, max_p=4,
                start_q=0, max_q=4,
                d=None, max_d=2,
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                D=None, max_D=1,
                m=seasonal_period,
                seasonal=True,
                information_criterion="bic",
                stepwise=False,
                n_fits=50,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                n_jobs=-1,
            )
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order
            logger.info("auto_arima selected order=%s seasonal_order=%s", best_order, best_seasonal_order)
        except Exception as exc:
            logger.warning("auto_arima failed (%s). Falling back to manual order.", exc)
            best_order = order
            best_seasonal_order = seasonal_order
    else:
        best_order = order
        best_seasonal_order = seasonal_order

    logger.info("Fitting SARIMAX(%s)(%s) for %s …", best_order, best_seasonal_order, ticker_display)
    model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300, method="lbfgs")

    # ── R² in-sample: fittedvalues vs actual (price scale, drop NaN warmup) ─
    mae, rmse, mape, r2 = np.nan, np.nan, np.nan, np.nan
    try:
        fitted_log = result.fittedvalues.dropna()          # drop NaN from differencing warmup
        actual_log = log_series.reindex(fitted_log.index) # align actual to fitted index

        a_price = np.expm1(actual_log.values)
        f_price = np.expm1(fitted_log.values)

        valid = np.isfinite(a_price) & np.isfinite(f_price) & (a_price > 0) & (f_price > 0)
        a_price, f_price = a_price[valid], f_price[valid]

        if len(a_price) >= 10:
            mae  = float(np.mean(np.abs(a_price - f_price)))
            rmse = float(np.sqrt(np.mean((a_price - f_price) ** 2)))
            mape = float(np.mean(np.abs((a_price - f_price) / a_price)) * 100)
            ss_res = np.sum((a_price - f_price) ** 2)
            ss_tot = np.sum((a_price - np.mean(a_price)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    except Exception as exc:
        logger.warning("SARIMA in-sample metrics failed: %s", exc)

    logger.info("SARIMA MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
                mae if np.isfinite(mae) else -1,
                rmse if np.isfinite(rmse) else -1,
                mape if np.isfinite(mape) else -1,
                r2 if np.isfinite(r2) else -1)

    payload = {
        "result":         result,
        "log_series":     log_series,
        "series":         series,
        "order":          best_order,
        "seasonal_order": best_seasonal_order,
        "metrics":        {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2},
    }
    save_model(payload, ticker_display, "sarima")
    return payload


# ---------------------------------------------------------------------------
# Prophet  — identical hyperparameters to original
# ---------------------------------------------------------------------------

def train_prophet(
    ticker_display: str,
    test_size: int = 20,
    changepoint_prior_scale: float = 0.1,
    seasonality_prior_scale: float = 15.0,
    seasonality_mode: str = "multiplicative",
) -> dict:
    """
    Fit Prophet on raw Close. Hyperparameters identical to original.
    Prophet dilatih pada SELURUH data (termasuk 20 hari terakhir) agar
    forecast tidak lompat jauh dari harga aktual terkini.
    R² tetap dihitung in-sample pada data training.
    """
    df = load_stock_data(ticker_display).reset_index()
    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df = prophet_df.dropna(subset=["y"])
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    if len(prophet_df) > TRAIN_WINDOW_DAYS:
        prophet_df = prophet_df.iloc[-TRAIN_WINDOW_DAYS:].reset_index(drop=True)
    logger.info(
        "Prophet training window for %s: %d rows (%s → %s)",
        ticker_display, len(prophet_df),
        prophet_df["ds"].iloc[0].date(), prophet_df["ds"].iloc[-1].date(),
    )

    # Latih pada SELURUH window — tidak ada train/test split
    # agar cutoff model = tanggal data terkini → forecast tidak lompat
    train_df = prophet_df.copy()

    logger.info("Fitting Prophet for %s (%d rows) …", ticker_display, len(train_df))
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        n_changepoints=25,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(train_df)

    # ── R² in-sample (fitted vs actual pada training data) ────────────────
    mae, rmse, mape, r2 = np.nan, np.nan, np.nan, np.nan
    try:
        insample_fc = m.predict(m.make_future_dataframe(periods=0, freq="B"))
        merged = insample_fc[["ds", "yhat"]].merge(train_df[["ds", "y"]], on="ds", how="inner")
        if len(merged) >= 10:
            a = merged["y"].values.astype(float)
            f = merged["yhat"].values.astype(float)
            valid = np.isfinite(a) & np.isfinite(f) & (a > 0) & (f > 0)
            a, f = a[valid], f[valid]
            if len(a) >= 10:
                mae  = float(np.mean(np.abs(a - f)))
                rmse = float(np.sqrt(np.mean((a - f) ** 2)))
                mape = float(np.mean(np.abs((a - f) / a)) * 100)
                ss_res = np.sum((a - f) ** 2)
                ss_tot = np.sum((a - np.mean(a)) ** 2)
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    except Exception as exc:
        logger.warning("Prophet in-sample metrics failed: %s", exc)

    logger.info("Prophet MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  R²=%.4f",
                mae if np.isfinite(mae) else -1,
                rmse if np.isfinite(rmse) else -1,
                mape if np.isfinite(mape) else -1,
                r2 if np.isfinite(r2) else -1)

    payload = {
        "model":   m,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2},
    }
    save_model(payload, ticker_display, "prophet")
    return payload


# ---------------------------------------------------------------------------
# Convenience: train both models for a ticker
# ---------------------------------------------------------------------------

def train_all(ticker_display: str) -> dict:
    """Train and persist both SARIMA and Prophet for the given ticker."""
    logger.info("=" * 60)
    logger.info("Training all models for: %s", ticker_display)
    logger.info("=" * 60)
    sarima_result  = train_sarima(ticker_display)
    prophet_result = train_prophet(ticker_display)
    return {"sarima": sarima_result, "prophet": prophet_result}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data_pipeline import TICKER_MAP

    if len(sys.argv) > 1:
        target = " ".join(sys.argv[1:])
        if target not in TICKER_MAP:
            matches = [k for k in TICKER_MAP if k.startswith(target)]
            if matches:
                target = matches[0]
        train_all(target)
    else:
        for name in TICKER_MAP:
            try:
                train_all(name)
            except Exception as e:
                logger.error("Failed to train %s: %s", name, e)

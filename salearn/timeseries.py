"""salearn.timeseries — forecasting for analysts (numpy-only, sklearn-style).

Estimators: NaiveForecaster, MovingAverageForecaster, ExponentialSmoothingForecaster,
ARForecaster (least-squares lag model), SeasonalNaiveForecaster.
Utils: temporal_split, rolling_origin_cv, mae/mase/smape, seasonal_decompose.
"""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, RegressorMixin


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    return y


class NaiveForecaster(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, y, X=None):
        self.last_ = float(_as_1d(y)[-1])
        self._is_fitted = True
        return self

    def predict(self, h: int | list = 1, X=None):
        self._check_fitted()
        h = len(h) if hasattr(h, "__len__") else int(h)
        return np.full(h, self.last_)


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, season=7):
        self.season = season

    def fit(self, y, X=None):
        y = _as_1d(y)
        self.hist_ = y[-self.season:].copy()
        self._is_fitted = True
        return self

    def predict(self, h=1, X=None):
        self._check_fitted()
        h = len(h) if hasattr(h, "__len__") else int(h)
        return np.array([self.hist_[i % self.season] for i in range(h)], dtype=np.float64)


class MovingAverageForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, window=7):
        self.window = window

    def fit(self, y, X=None):
        y = _as_1d(y)
        self.avg_ = float(y[-self.window:].mean())
        self._is_fitted = True
        return self

    def predict(self, h=1, X=None):
        self._check_fitted()
        h = len(h) if hasattr(h, "__len__") else int(h)
        return np.full(h, self.avg_)


class ExponentialSmoothingForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, y, X=None):
        y = _as_1d(y)
        s = float(y[0])
        for v in y[1:]:
            s = self.alpha * float(v) + (1 - self.alpha) * s
        self.level_ = s
        self._is_fitted = True
        return self

    def predict(self, h=1, X=None):
        self._check_fitted()
        h = len(h) if hasattr(h, "__len__") else int(h)
        return np.full(h, self.level_)


class ARForecaster(BaseEstimator, RegressorMixin):
    """AR(p) via least squares on lags. predict(h) iterates recursively."""

    def __init__(self, lags=7):
        self.lags = lags

    def fit(self, y, X=None):
        y = _as_1d(y)
        p = min(self.lags, len(y) - 1)
        self.lags_ = p
        Y = y[p:]
        L = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)])
        A = np.hstack([L, np.ones((len(L), 1))])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = float(coef[-1])
        self.hist_ = y[-p:].tolist()
        self._is_fitted = True
        return self

    def predict(self, h=1, X=None):
        self._check_fitted()
        h = len(h) if hasattr(h, "__len__") else int(h)
        hist = list(self.hist_)
        out = []
        for _ in range(h):
            window = hist[-self.lags_:]
            if len(window) < self.lags_:
                window = [window[0]] * (self.lags_ - len(window)) + window if window else [0.0] * self.lags_
            # coef_[0]*y[t-1] + coef_[1]*y[t-2] + ...
            v = float(sum(c * w for c, w in zip(self.coef_, list(reversed(window)))) + self.intercept_)
            out.append(v)
            hist.append(v)
        return np.array(out)


def temporal_split(y, test_size=0.2):
    y = _as_1d(y)
    n = len(y)
    k = int(n * test_size) if isinstance(test_size, float) else int(test_size)
    return y[: n - k], y[n - k:]


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def smape(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(100 * np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-9)))


def mase(train, pred, actual, season=1) -> float:
    train = _as_1d(train)
    scale = np.mean(np.abs(train[season:] - train[:-season])) or 1.0
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))) / scale)


def rolling_origin_cv(y, forecaster, horizon=7, folds=3, metric=mae):
    """Walk-forward validation: returns list of scores (lower=better)."""
    from .base import clone
    y = _as_1d(y)
    n = len(y)
    step = max(horizon, (n - horizon) // (folds + 1))
    scores = []
    for f in range(folds):
        end = n - horizon * (folds - f)
        if end <= horizon:
            continue
        m = clone(forecaster).fit(y[:end])
        p = m.predict(horizon)
        scores.append(metric(y[end: end + horizon], p))
    return scores


def seasonal_decompose(y, season=7):
    """Return dict(trend, seasonal, resid) via moving averages."""
    y = _as_1d(y)
    s = np.convolve(y, np.ones(season) / season, mode="same")
    detr = y - s
    seas = np.array([np.nanmean([detr[i] for i in range(len(detr)) if i % season == k]) for k in range(season)])
    seas_full = np.array([seas[i % season] for i in range(len(y))])
    return {"trend": s, "seasonal": seas_full, "resid": y - s - seas_full}


__all__ = ["NaiveForecaster", "SeasonalNaiveForecaster", "MovingAverageForecaster",
           "ExponentialSmoothingForecaster", "ARForecaster", "temporal_split",
           "mae", "smape", "mase", "rolling_origin_cv", "seasonal_decompose"]

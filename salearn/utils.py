"""salearn.utils — validation helpers (sklearn-compatible subset)."""
from __future__ import annotations

import numpy as np


def check_array(X, ensure_2d=True, dtype=np.float64, ensure_min_samples=1,
                ensure_min_features=1, force_all_finite=True, copy=False, order="K"):
    if isinstance(X, (list, tuple)):
        X = np.asarray(X, dtype=dtype, order=order)
    elif hasattr(X, "to_numpy"):  # pandas
        X = X.to_numpy(dtype=dtype)
    elif hasattr(X, "values"):
        try:
            X = np.asarray(X.values, dtype=dtype)
        except Exception:
            X = np.asarray(X, dtype=dtype)
    else:
        X = np.asarray(X, dtype=dtype, order=order) if not isinstance(X, np.ndarray) else X
        if X.dtype != dtype and dtype is not None:
            X = X.astype(dtype, copy=False)
    if copy:
        X = X.copy()
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    if ensure_2d and X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if force_all_finite and X.size and np.issubdtype(X.dtype, np.number):
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN or infinity. Use SimpleImputer first.")
    if ensure_min_samples and X.shape[0] < ensure_min_samples:
        raise ValueError("Not enough samples")
    return X


def check_X_y(X, y, dtype=np.float64, ensure_2d=True, force_all_finite=True):
    X = check_array(X, ensure_2d=ensure_2d, dtype=dtype, force_all_finite=force_all_finite)
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X ({X.shape[0]}) and y ({y.shape[0]}) have inconsistent lengths")
    return X, y


def check_is_fitted(est):
    if not getattr(est, "_is_fitted", False):
        from .base import NotFittedError
        raise NotFittedError(f"{type(est).__name__} is not fitted yet.")


def column_or_1d(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    if y.ndim > 2 or (y.ndim == 2 and y.shape[1] > 1):
        raise ValueError("y should be 1d")
    return y


def shuffle(*arrays, random_state=None):
    rng = np.random.RandomState(random_state)
    n = len(arrays[0])
    idx = rng.permutation(n)
    return [np.asarray(a)[idx] for a in arrays]


def safe_indexing(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return np.asarray(X)[indices]

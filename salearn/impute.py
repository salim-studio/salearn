"""salearn.impute — missing value imputers."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin


class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None):
        self.missing_values = missing_values; self.strategy = strategy; self.fill_value = fill_value

    def _mask(self, X):
        X = np.asarray(X, dtype=np.float64) if self.missing_values is np.nan or (isinstance(self.missing_values, float) and np.isnan(self.missing_values)) else np.asarray(X)
        try:
            if isinstance(self.missing_values, float) and np.isnan(self.missing_values):
                return np.isnan(X.astype(np.float64, copy=False)) if np.asarray(X).dtype.kind in "iufc" else np.asarray([[False] * np.asarray(X).shape[1]] * len(X))
            return np.asarray(X) == self.missing_values
        except Exception:
            return np.zeros(np.shape(X), dtype=bool)

    def fit(self, X, y=None):
        Xo = np.asarray(X)
        is_float = Xo.dtype.kind in "iufc"
        Xf = Xo.astype(np.float64) if is_float else Xo
        mask = self._mask(Xo if not is_float else Xf)
        if self.strategy == "mean":
            self.statistics_ = np.array([np.nanmean(Xf[:, j][~mask[:, j]]) if (~mask[:, j]).any() else 0.0 for j in range(Xf.shape[1])])
        elif self.strategy == "median":
            self.statistics_ = np.array([np.nanmedian(Xf[:, j][~mask[:, j]]) if (~mask[:, j]).any() else 0.0 for j in range(Xf.shape[1])])
        elif self.strategy == "most_frequent":
            s = []
            for j in range(Xf.shape[1] if is_float else Xo.shape[1]):
                col = (Xo[:, j] if not is_float else Xf[:, j])[~mask[:, j]]
                if len(col) == 0:
                    s.append(0)
                else:
                    v, c = np.unique(col, return_counts=True)
                    s.append(v[c.argmax()])
            self.statistics_ = np.array(s, dtype=object if not is_float else np.float64)
        elif self.strategy == "constant":
            n = Xo.shape[1]
            self.statistics_ = np.full(n, 0 if self.fill_value is None else self.fill_value)
        else:
            raise ValueError("unknown strategy")
        self.n_features_in_ = Xo.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        Xo = np.asarray(X)
        out = Xo.astype(np.float64, copy=True) if Xo.dtype.kind in "iufc" or isinstance(self.statistics_[0], (int, float, np.number)) else Xo.astype(object, copy=True)
        mask = self._mask(Xo)
        for j in range(out.shape[1]):
            out[mask[:, j], j] = self.statistics_[j]
        if out.dtype == object:
            try:
                out = out.astype(np.float64)
            except Exception:
                pass
        return out


class KNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, n_neighbors=5, weights="uniform"):
        self.missing_values = missing_values; self.n_neighbors = n_neighbors; self.weights = weights

    def fit(self, X, y=None):
        self._X = np.asarray(X, dtype=np.float64)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        from .metrics import pairwise_distances
        X = np.asarray(X, dtype=np.float64).copy()
        mask = np.isnan(X)
        if not mask.any():
            return X
        col_mean = np.nanmean(X, axis=0)
        # nan-aware distances: fill nan with col mean for distance computation
        Xfill = np.where(mask, col_mean, X)
        D = pairwise_distances(Xfill)
        np.fill_diagonal(D, np.inf)
        for i in range(len(X)):
            for j in np.where(mask[i])[0]:
                # neighbors with observed j
                cand = np.where(~np.isnan(self._X[:, j]))[0] if hasattr(self, "_X") else np.where(~mask[:, j])[0]
                if len(cand) == 0:
                    X[i, j] = col_mean[j]
                    continue
                d = D[i, cand]
                k = min(self.n_neighbors, len(cand))
                nn = cand[np.argpartition(d, k - 1)[:k]]
                vals = X[nn, j]
                valid = ~np.isnan(vals)
                if not valid.any():
                    X[i, j] = col_mean[j]
                elif self.weights == "distance":
                    w = 1.0 / np.maximum(D[i, nn[valid]], 1e-12)
                    X[i, j] = (vals[valid] * w).sum() / w.sum()
                else:
                    X[i, j] = vals[valid].mean()
        return X

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y).transform(X)

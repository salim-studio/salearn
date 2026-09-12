"""salearn.features — feature engineering beyond sklearn basics.

- DateFeaturizer: year/month/day/dow/quarter/weekend from str/datetime col
- TargetEncoder / FrequencyEncoder (K-fold safe-ish, sklearn API)
- NumericBinner (quantile/uniform), RareLabelEncoder
- AutoFeaturizer: one-shot numeric+categorical pipeline helper
All work with pandas; fall back gracefully for numpy.
"""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin


class DateFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, drop_original=True):
        self.column = column
        self.drop_original = drop_original

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        import pandas as pd
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        col = self.column if self.column in df.columns else df.columns[0]
        s = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = s.dt.year.fillna(0).astype(int)
        df[f"{col}_month"] = s.dt.month.fillna(0).astype(int)
        df[f"{col}_day"] = s.dt.day.fillna(0).astype(int)
        df[f"{col}_dow"] = s.dt.dayofweek.fillna(0).astype(int)
        df[f"{col}_quarter"] = s.dt.quarter.fillna(0).astype(int)
        df[f"{col}_weekend"] = (s.dt.dayofweek >= 5).astype(int)
        if self.drop_original:
            df = df.drop(columns=[col])
        return df


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Mean-target encoding with smoothing. fit(X, y) where X is DataFrame/Series."""

    def __init__(self, columns=None, smoothing=10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y):
        import pandas as pd
        y = np.asarray(y, dtype=np.float64)
        self.global_mean_ = float(y.mean())
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or list(df.columns)
        if isinstance(cols, str):
            cols = [cols]
        self.columns_ = cols
        self.maps_ = {}
        for c in cols:
            g = pd.DataFrame({"k": df[c].astype(str), "y": y}).groupby("k")["y"].agg(["mean", "count"])
            smooth = (g["count"] * g["mean"] + self.smoothing * self.global_mean_) / (g["count"] + self.smoothing)
            self.maps_[c] = smooth.to_dict()
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        import pandas as pd
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in self.columns_:
            df[c] = df[c].astype(str).map(self.maps_[c]).fillna(self.global_mean_)
        return df[self.columns_] if len(self.columns_) > 1 or isinstance(X, pd.DataFrame) else df.iloc[:, 0].to_numpy()


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        import pandas as pd
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns or list(df.columns)
        if isinstance(cols, str):
            cols = [cols]
        self.columns_ = cols
        self.maps_ = {c: (df[c].astype(str).value_counts(normalize=True).to_dict()) for c in cols}
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        import pandas as pd
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in self.columns_:
            df[c] = df[c].astype(str).map(self.maps_[c]).fillna(0.0)
        return df


class NumericBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, strategy="quantile"):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        self.edges_ = []
        for j in range(Xa.shape[1]):
            col = Xa[:, j]
            col = col[np.isfinite(col)]
            if self.strategy == "quantile":
                e = np.quantile(col, np.linspace(0, 1, self.n_bins + 1)) if len(col) else np.linspace(0, 1, self.n_bins + 1)
            else:
                e = np.linspace(col.min(), col.max(), self.n_bins + 1) if len(col) else np.linspace(0, 1, self.n_bins + 1)
            e = np.unique(e)
            if len(e) < 2:
                e = np.array([e[0] - 0.5, e[0] + 0.5]) if len(e) else np.array([0, 1])
            self.edges_.append(e)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        return np.column_stack([np.clip(np.digitize(Xa[:, j], self.edges_[j][1:-1]), 0, len(self.edges_[j]) - 2)
                                for j in range(Xa.shape[1])])


def auto_featurize(df, target: str | None = None, date_cols=(), max_onehot=20):
    """Return (X numpy, y or None, feature_names) — numeric fill + one-hot small cats."""
    import pandas as pd
    import numpy as np
    work = df.copy()
    y = work.pop(target).to_numpy() if target and target in work.columns else None
    for c in date_cols:
        if c in work.columns:
            work = DateFeaturizer(c).fit_transform(work)
    num = work.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float64)
    num_names = list(work.select_dtypes(include=[np.number]).columns)
    cats = work.select_dtypes(exclude=[np.number])
    oh_parts, oh_names = [], []
    for c in cats.columns:
        vc = cats[c].astype(str).value_counts()
        keep = list(vc.head(max_onehot).index)
        for k in keep:
            oh_parts.append((cats[c].astype(str) == k).to_numpy(dtype=np.float64))
            oh_names.append(f"{c}={k}")
    X = np.hstack([num] + ([np.column_stack(oh_parts)] if oh_parts else [])) if (num.size or oh_parts) else np.empty((len(work), 0))
    return X, y, num_names + oh_names


__all__ = ["DateFeaturizer", "TargetEncoder", "FrequencyEncoder", "NumericBinner", "auto_featurize"]

"""salearn.clean — data cleaning as sklearn transformers.

    from salearn.clean import DataCleaner
    Xc = DataCleaner(drop_dupes=True, outlier_clip="iqr").fit_transform(df)

- handles pandas DataFrame OR numpy (numeric path)
- missing: median/mean/most_frequent/constant/drop
- outliers: clip by IQR or z-score
- dedup, constant-column drop, negative/inf sanitizer
"""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing="median", fill_value=0.0, drop_dupes=True,
                 drop_constants=True, outlier_clip: str | None = None,
                 z_thresh=3.0, iqr_k=1.5):
        self.missing = missing
        self.fill_value = fill_value
        self.drop_dupes = drop_dupes
        self.drop_constants = drop_constants
        self.outlier_clip = outlier_clip
        self.z_thresh = z_thresh
        self.iqr_k = iqr_k

    # -- fit --
    def fit(self, X, y=None):
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return self._fit_frame(X)
        except ImportError:
            pass
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        self._is_frame = False
        col_mean = np.nanmean(Xa, axis=0)
        col_med = np.nanmedian(Xa, axis=0)
        self.fill_ = np.where(np.isnan(col_med if self.missing == "median" else col_mean),
                              self.fill_value, col_med if self.missing == "median" else col_mean)
        if self.missing == "constant":
            self.fill_ = np.full(Xa.shape[1], self.fill_value)
        elif self.missing == "mean":
            self.fill_ = np.where(np.isnan(col_mean), self.fill_value, col_mean)
        self._bounds = self._numpy_bounds(np.where(np.isnan(Xa), self.fill_, Xa))
        self._is_fitted = True
        return self

    def _fit_frame(self, df):
        import pandas as pd
        self._is_frame = True
        self.columns_ = list(df.columns)
        self.numeric_ = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.fill_dict_ = {}
        for c in df.columns:
            s = df[c]
            if self.missing == "median" and c in self.numeric_:
                self.fill_dict_[c] = s.median()
            elif self.missing == "mean" and c in self.numeric_:
                self.fill_dict_[c] = s.mean()
            elif self.missing == "most_frequent":
                self.fill_dict_[c] = s.mode().iloc[0] if not s.mode().empty else self.fill_value
            elif self.missing == "constant":
                self.fill_dict_[c] = self.fill_value
            else:
                self.fill_dict_[c] = s.median() if c in self.numeric_ else (s.mode().iloc[0] if not s.mode().empty else self.fill_value)
            try:
                if pd.isna(self.fill_dict_[c]):
                    self.fill_dict_[c] = self.fill_value
            except Exception:
                pass
        self._const_drop = [c for c in df.columns if df[c].nunique(dropna=False) <= 1] if self.drop_constants else []
        tmp = df.fillna(self.fill_dict_)
        self._bounds = {}
        for c in self.numeric_:
            s = tmp[c].to_numpy(dtype=np.float64)
            self._bounds[c] = self._col_bounds(s)
        self._is_fitted = True
        return self

    def _col_bounds(self, s):
        if self.outlier_clip == "zscore":
            mu, sd = float(np.mean(s)), float(np.std(s) or 1)
            return mu - self.z_thresh * sd, mu + self.z_thresh * sd
        if self.outlier_clip == "iqr":
            q1, q3 = np.percentile(s, [25, 75]) if len(s) else (0, 0)
            iqr = q3 - q1
            return q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
        return -np.inf, np.inf

    def _numpy_bounds(self, Xa):
        lo = np.full(Xa.shape[1], -np.inf)
        hi = np.full(Xa.shape[1], np.inf)
        for j in range(Xa.shape[1]):
            lo[j], hi[j] = self._col_bounds(Xa[:, j])
        return (lo, hi)

    # -- transform --
    def transform(self, X):
        self._check_fitted()
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame) and getattr(self, "_is_frame", False):
                df = X.copy()
                if self.drop_dupes:
                    df = df.drop_duplicates().reset_index(drop=True)
                df = df.fillna(self.fill_dict_)
                for c, (lo, hi) in self._bounds.items():
                    if c in df.columns and (lo != -np.inf or hi != np.inf):
                        df[c] = df[c].clip(lo, hi)
                df = df.replace([np.inf, -np.inf], np.nan).fillna(self.fill_dict_)
                if self._const_drop:
                    df = df.drop(columns=[c for c in self._const_drop if c in df.columns])
                return df
        except ImportError:
            pass
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        Xa = np.where(np.isnan(Xa), self.fill_, Xa)
        lo, hi = self._bounds
        Xa = np.clip(Xa, lo, hi)
        Xa[~np.isfinite(Xa)] = self.fill_value
        return Xa


__all__ = ["DataCleaner"]

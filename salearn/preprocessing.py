"""salearn.preprocessing — scalers / encoders / transformers (sklearn-compatible)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin
from .utils import check_array


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean; self.with_std = with_std

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite=False)
        self.mean_ = X.mean(0) if self.with_mean else np.zeros(X.shape[1])
        s = X.std(0) if self.with_std else np.ones(X.shape[1])
        self.scale_ = np.where(s == 0, 1.0, s)
        self.var_ = (X.var(0) if self.with_std else np.ones(X.shape[1]))
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X, force_all_finite=False).astype(np.float64, copy=True)
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64).copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite=False)
        self.data_min_ = X.min(0); self.data_max_ = X.max(0)
        rng = self.data_max_ - self.data_min_
        self.scale_ = np.where(rng == 0, 1.0, (self.feature_range[1] - self.feature_range[0]) / np.where(rng == 0, 1, rng))
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return check_array(X, force_all_finite=False) * self.scale_ + self.min_

    def inverse_transform(self, X):
        self._check_fitted()
        return (np.asarray(X, dtype=np.float64) - self.min_) / np.where(self.scale_ == 0, 1, self.scale_)


class MaxAbsScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X, force_all_finite=False)
        self.max_abs_ = np.maximum(np.abs(X).max(0), 1e-12)
        self.scale_ = self.max_abs_
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return check_array(X, force_all_finite=False) / self.scale_

    def inverse_transform(self, X):
        self._check_fitted()
        return np.asarray(X, dtype=np.float64) * self.scale_


class RobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering; self.with_scaling = with_scaling
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite=False)
        q0, q1 = np.percentile(X, self.quantile_range, axis=0)
        self.center_ = np.median(X, axis=0) if self.with_centering else np.zeros(X.shape[1])
        iqr = (q1 - q0) if self.with_scaling else np.ones(X.shape[1])
        self.scale_ = np.where(iqr == 0, 1.0, iqr)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X, force_all_finite=False).astype(np.float64, copy=True)
        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64).copy()
        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        return X


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X, y=None):
        self.n_features_in_ = check_array(X, force_all_finite=False).shape[1]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X, force_all_finite=False).astype(np.float64, copy=True)
        if self.norm == "l2":
            n = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == "l1":
            n = np.abs(X).sum(1, keepdims=True)
        elif self.norm == "max":
            n = np.abs(X).max(1, keepdims=True)
        else:
            raise ValueError("norm must be l1/l2/max")
        return X / np.maximum(n, 1e-12)


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return (check_array(X, force_all_finite=False) > self.threshold).astype(np.float64)


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree; self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        from itertools import combinations, combinations_with_replacement
        X = check_array(X, force_all_finite=False)
        n = X.shape[1]
        gen = combinations if self.interaction_only else combinations_with_replacement
        self.powers_ = []
        for d in range(1 if not self.include_bias else 0, self.degree + 1):
            if d == 0:
                self.powers_.append(())
                continue
            for c in gen(range(n), d):
                self.powers_.append(c)
        if self.include_bias and () not in self.powers_:
            self.powers_ = [()] + self.powers_
        self.n_output_features_ = len(self.powers_)
        self.n_features_in_ = n
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X, force_all_finite=False)
        cols = []
        for c in self.powers_:
            if len(c) == 0:
                cols.append(np.ones(len(X)))
            else:
                cols.append(np.prod(X[:, list(c)], axis=1))
        return np.column_stack(cols) if cols else np.ones((len(X), 0))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False, handle_unknown="error", drop=None):
        self.sparse_output = sparse_output; self.handle_unknown = handle_unknown; self.drop = drop

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = []
        for j in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, j]))
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        import scipy.sparse as sp
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        parts = []
        for j, cats in enumerate(self.categories_):
            idx = {c: i for i, c in enumerate(cats)}
            rows, cols, data = [], [], []
            for i, v in enumerate(X[:, j]):
                if v in idx:
                    rows.append(i); cols.append(idx[v]); data.append(1.0)
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category {v!r}")
            m = sp.csr_matrix((data, (rows, cols)), shape=(len(X), len(cats)))
            parts.append(m)
        import scipy.sparse as sp
        M = sp.hstack(parts).tocsr()
        if self.sparse_output:
            return M
        return M.toarray()

    def get_feature_names_out(self, names=None):
        out = []
        for j, cats in enumerate(self.categories_):
            pre = names[j] if names is not None else f"x{j}"
            out += [f"{pre}_{c}" for c in cats]
        return np.array(out)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.empty(X.shape, dtype=np.float64)
        for j, cats in enumerate(self.categories_):
            idx = {c: i for i, c in enumerate(cats)}
            out[:, j] = [idx.get(v, -1) for v in X[:, j]]
        return out


class LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = np.unique(np.asarray(y))
        self._is_fitted = True
        return self

    def transform(self, y):
        self._check_fitted()
        idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([idx[v] for v in np.asarray(y)])

    def fit_transform(self, y, **kw):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        self._check_fitted()
        return self.classes_[np.asarray(y)]


class LabelBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = np.unique(np.asarray(y))
        self._is_fitted = True
        return self

    def transform(self, y):
        self._check_fitted()
        y = np.asarray(y)
        if len(self.classes_) == 2:
            return (y == self.classes_[1]).astype(int).reshape(-1, 1)
        out = np.zeros((len(y), len(self.classes_)), dtype=int)
        idx = {c: i for i, c in enumerate(self.classes_)}
        for i, v in enumerate(y):
            out[i, idx[v]] = 1
        return out

    def fit_transform(self, y, **kw):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y):
        Y = np.asarray(Y)
        if len(self.classes_) == 2:
            return np.where(Y.ravel() > 0, self.classes_[1], self.classes_[0])
        return self.classes_[Y.argmax(1)]


def add_dummy_feature(X, value=1.0):
    X = check_array(X, force_all_finite=False)
    return np.hstack([X, np.full((len(X), 1), value)])


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    return StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(np.asarray(X, dtype=np.float64))


def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
    return MinMaxScaler(feature_range).fit_transform(np.asarray(X, dtype=np.float64))


def robust_scale(X, **kw):
    return RobustScaler(**kw).fit_transform(np.asarray(X, dtype=np.float64))


def normalize(X, norm="l2", axis=1, copy=True):
    return Normalizer(norm=norm).fit_transform(np.asarray(X, dtype=np.float64))


def binarize(X, threshold=0.0, copy=True):
    return (np.asarray(X, dtype=np.float64) > threshold).astype(np.float64)

"""salearn.feature_selection — selectors (sklearn-compatible)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin
from .utils import check_X_y, check_array


def chi2(X, y):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    classes = np.unique(y)
    obs = np.array([X[y == c].sum(0) for c in classes])
    feat_sum = obs.sum(0, keepdims=True)
    class_sum = obs.sum(1, keepdims=True)
    tot = obs.sum()
    exp = class_sum @ feat_sum / max(tot, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = ((obs - exp) ** 2 / np.maximum(exp, 1e-12)).sum(0)
    from scipy.stats import chi2 as _c
    p = _c.sf(chi, len(classes) - 1)
    return chi, p


def f_classif(X, y):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    classes = np.unique(y)
    grand = X.mean(0)
    ss_b = sum((X[y == c].mean(0) - grand) ** 2 * (y == c).sum() for c in classes) / (len(classes) - 1)
    ss_w = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in classes) / max(len(X) - len(classes), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ss_b / np.maximum(ss_w, 1e-12)
    from scipy.stats import f as _f
    p = _f.sf(F, len(classes) - 1, len(X) - len(classes))
    return F, p


def f_regression(X, y):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    corr = (X - X.mean(0)).T @ y / np.maximum(np.sqrt(((X - X.mean(0)) ** 2).sum(0) * (y ** 2).sum()), 1e-12)
    corr = np.clip(corr, -1, 1)
    dof = len(y) - 2
    with np.errstate(divide="ignore"):
        F = corr ** 2 / (1 - corr ** 2) * dof
    from scipy.stats import f as _f
    p = _f.sf(F, 1, dof)
    return F, p


def mutual_info_classif(X, y, random_state=None):
    try:
        from sklearn.feature_selection import mutual_info_classif as _m
        return _m(X, y, random_state=random_state)
    except ImportError:
        F, _ = f_classif(X, y)
        return F


class SelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func; self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=False)
        s, p = self.score_func(X, y)
        self.scores_ = np.asarray(s, dtype=np.float64); self.pvalues_ = np.asarray(p)
        k = X.shape[1] if self.k == "all" else min(self.k, X.shape[1])
        self._idx = np.argsort(-np.nan_to_num(self.scores_))[:k]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return check_array(X, force_all_finite=False)[:, self._idx]

    def get_support(self):
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self._idx] = True
        return mask


class SelectPercentile(SelectKBest):
    def __init__(self, score_func=f_classif, percentile=10):
        super().__init__(score_func=score_func, k=10)
        self.percentile = percentile

    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=False)
        k = max(1, int(X.shape[1] * self.percentile / 100))
        self.k = k
        return super().fit(X, y)


class VarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.variances_ = X.var(0)
        self._idx = np.where(self.variances_ > self.threshold)[0]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return np.asarray(X)[:, self._idx]


class SelectFromModel(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, threshold="mean", prefit=False):
        self.estimator = estimator; self.threshold = threshold; self.prefit = prefit

    def fit(self, X, y=None):
        from .base import clone
        self.estimator_ = self.estimator if self.prefit else clone(self.estimator).fit(X, y)
        if hasattr(self.estimator_, "feature_importances_"):
            imp = self.estimator_.feature_importances_
        elif hasattr(self.estimator_, "coef_"):
            imp = np.abs(np.asarray(self.estimator_.coef_)).ravel() if np.asarray(self.estimator_.coef_).ndim <= 2 else np.abs(np.asarray(self.estimator_.coef_)).mean(0)
        else:
            raise ValueError("estimator has no importances/coef_")
        thr = imp.mean() if self.threshold == "mean" else (np.median(imp) if self.threshold == "median" else float(self.threshold))
        self.threshold_ = thr
        self._idx = np.where(imp >= thr)[0]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return np.asarray(X)[:, self._idx]


class RFE(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator = estimator; self.n_features_to_select = n_features_to_select; self.step = step

    def fit(self, X, y):
        from .base import clone
        X, y = check_X_y(X, y)
        n = X.shape[1]
        target = self.n_features_to_select or max(n // 2, 1)
        support = np.ones(n, dtype=bool)
        ranking = np.ones(n, dtype=int)
        while support.sum() > target:
            e = clone(self.estimator).fit(X[:, support], y)
            if hasattr(e, "feature_importances_"):
                imp = e.feature_importances_
            else:
                imp = np.abs(np.asarray(e.coef_).ravel())
            order = np.argsort(imp)
            kill = order[:min(self.step, support.sum() - target)]
            idx = np.where(support)[0][kill]
            support[idx] = False
            ranking[idx] += 1
        self.support_ = support
        self.ranking_ = ranking
        self.estimator_ = clone(self.estimator).fit(X[:, support], y)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        return np.asarray(X)[:, self.support_]

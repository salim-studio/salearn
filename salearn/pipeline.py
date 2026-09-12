"""salearn.pipeline — Pipeline, FeatureUnion, make_pipeline (sklearn-compatible)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, clone


class Pipeline(BaseEstimator):
    def __init__(self, steps, memory=None, verbose=False):
        self.steps = steps; self.memory = memory; self.verbose = verbose

    @property
    def named_steps(self):
        return {n: e for n, e in self.steps}

    def _final(self):
        return self.steps[-1][1]

    def fit(self, X, y=None, **kw):
        Xt = X
        for name, est in self.steps[:-1]:
            if hasattr(est, "fit_transform"):
                Xt = est.fit_transform(Xt, y)
            else:
                est.fit(Xt, y)
                Xt = est.transform(Xt) if hasattr(est, "transform") else Xt
        self.steps[-1][1].fit(Xt, y)
        for a in ("classes_", "coef_", "intercept_", "feature_importances_", "labels_"):
            if hasattr(self._final(), a):
                setattr(self, a, getattr(self._final(), a))
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        Xt = X
        for _, est in self.steps:
            if hasattr(est, "transform"):
                Xt = est.transform(Xt)
        return Xt

    def predict(self, X):
        self._check_fitted()
        Xt = X
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return self._final().predict(Xt)

    def predict_proba(self, X):
        self._check_fitted()
        Xt = X
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return self._final().predict_proba(Xt)

    def decision_function(self, X):
        Xt = X
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return self._final().decision_function(Xt)

    def score(self, X, y, sample_weight=None):
        Xt = X
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        try:
            return self._final().score(Xt, y, sample_weight=sample_weight)
        except TypeError:
            return self._final().score(Xt, y)

    def get_params(self, deep=True):
        out = {"steps": self.steps, "memory": self.memory, "verbose": self.verbose}
        if deep:
            for n, e in self.steps:
                if hasattr(e, "get_params"):
                    for k, v in e.get_params(deep=True).items():
                        out[f"{n}__{k}"] = v
        return out

    def set_params(self, **params):
        for k, v in params.items():
            if "__" in k:
                name, sub = k.split("__", 1)
                dict(self.steps)[name].set_params(**{sub: v})
            elif k == "steps":
                self.steps = v
            else:
                setattr(self, k, v)
        return self


def make_pipeline(*steps, **kw):
    names = []
    seen = {}
    for s in steps:
        n = type(s).__name__.lower()
        seen[n] = seen.get(n, 0) + 1
        name = n if seen[n] == 1 else f"{n}-{seen[n]}"
        names.append((name, s))
    return Pipeline(names, **kw)


class FeatureUnion(BaseEstimator):
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights

    def fit(self, X, y=None):
        for _, t in self.transformer_list:
            t.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        parts = []
        for name, t in self.transformer_list:
            Xt = t.transform(X)
            if self.transformer_weights and name in self.transformer_weights:
                Xt = Xt * self.transformer_weights[name]
            parts.append(Xt)
        import scipy.sparse as sp
        if any(sp.issparse(p) for p in parts):
            return sp.hstack(parts).tocsr()
        return np.hstack([np.asarray(p) for p in parts])

    def fit_transform(self, X, y=None, **kw):
        parts = []
        for _, t in self.transformer_list:
            parts.append(t.fit_transform(X, y))
        import scipy.sparse as sp
        if any(sp.issparse(p) for p in parts):
            return sp.hstack(parts).tocsr()
        return np.hstack([np.asarray(p) for p in parts])


def make_union(*transformers, **kw):
    lst = [(type(t).__name__.lower() + (f"-{i}" if i else ""), t) for i, t in enumerate(transformers)]
    return FeatureUnion(lst, **kw)

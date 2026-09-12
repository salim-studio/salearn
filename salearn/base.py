"""salearn.base — sklearn-compatible base classes."""
from __future__ import annotations

import copy
import numpy as np


class NotFittedError(ValueError, AttributeError):
    pass


class BaseEstimator:
    """Minimal sklearn-compatible BaseEstimator."""

    def get_params(self, deep=True):
        out = {}
        for k in getattr(self, "_get_param_names", lambda: [])():
            v = getattr(self, k, None)
            out[k] = v
            if deep and hasattr(v, "get_params"):
                for sk, sv in v.get_params(deep=True).items():
                    out[f"{k}__{sk}"] = sv
        # fallback: inspect __init__ signature
        if not out:
            import inspect
            try:
                sig = inspect.signature(self.__init__)
                for p in sig.parameters:
                    if p == "self":
                        continue
                    if hasattr(self, p):
                        out[p] = getattr(self, p)
            except Exception:
                pass
        return out

    def _get_param_names(self):
        import inspect
        try:
            sig = inspect.signature(self.__init__)
            return [p for p in sig.parameters if p != "self"]
        except Exception:
            return []

    def set_params(self, **params):
        for k, v in params.items():
            if "__" in k:
                name, sub = k.split("__", 1)
                getattr(self, name).set_params(**{sub: v})
            else:
                if not hasattr(self, k):
                    raise ValueError(f"Invalid parameter {k!r} for {type(self).__name__}")
                setattr(self, k, v)
        return self

    def clone(self):
        return clone(self)

    def __repr__(self):
        try:
            p = self.get_params(deep=False)
            s = ", ".join(f"{k}={v!r}" for k, v in p.items())
            return f"{type(self).__name__}({s})"
        except Exception:
            return super().__repr__()

    def _check_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted yet. Call fit() first.")


class ClassifierMixin:
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class RegressorMixin:
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight)


class TransformerMixin:
    _estimator_type = "transformer"

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)


class ClusterMixin:
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None, **kw):
        return self.fit(X, y, **kw).labels_


def clone(estimator, safe=True):
    import inspect
    if hasattr(estimator, "get_params"):
        params = estimator.get_params(deep=False)
        try:
            return type(estimator)(**copy.deepcopy(params))
        except Exception:
            if safe:
                raise
            return copy.deepcopy(estimator)
    return copy.deepcopy(estimator)


def is_classifier(est):
    return getattr(est, "_estimator_type", None) == "classifier"


def is_regressor(est):
    return getattr(est, "_estimator_type", None) == "regressor"

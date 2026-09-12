"""salearn.calibration — CalibratedClassifierCV + isotonic/sigmoid calibration."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin, clone
from .model_selection import StratifiedKFold, KFold
from .utils import safe_indexing


def _sigmoid_calib(p, y):
    from scipy.optimize import minimize
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y)

    def obj(ab):
        a, b = ab
        q = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()

    r = minimize(obj, [1.0, 0.0], method="Nelder-Mead")
    return r.x


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, method="sigmoid", cv=5):
        self.estimator = estimator; self.method = method; self.cv = cv

    def fit(self, X, y, sample_weight=None):
        from .linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        cv = self.cv if hasattr(self.cv, "split") else (StratifiedKFold(n_splits=self.cv) if len(self.classes_) <= 20 else KFold(n_splits=self.cv if isinstance(self.cv, int) else 5))
        self.calibrators_ = []
        self.estimators_ = []
        for tr, te in cv.split(X, y):
            e = clone(self.estimator).fit(safe_indexing(X, tr), safe_indexing(y, tr))
            self.estimators_.append(e)
            if hasattr(e, "predict_proba"):
                p = e.predict_proba(safe_indexing(X, te))
                p = p[:, 1] if p.shape[1] == 2 else p.max(1)
            else:
                s = e.decision_function(safe_indexing(X, te))
                p = 1 / (1 + np.exp(-s))
            yte = (safe_indexing(y, te) == self.classes_[-1]).astype(int) if len(self.classes_) == 2 else safe_indexing(y, te)
            if self.method == "sigmoid":
                if len(self.classes_) == 2:
                    self.calibrators_.append(_sigmoid_calib(p, yte))
                else:
                    self.calibrators_.append(None)
            else:
                ir = IsotonicRegression(out_of_bounds="clip").fit(p, yte if len(self.classes_) == 2 else (yte == self.classes_[-1]).astype(int))
                self.calibrators_.append(ir)
        # refit on full data
        from .base import clone as _clone
        self.base_estimator_ = _clone(self.estimator).fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        P = np.mean([e.predict_proba(X) if hasattr(e, "predict_proba") else np.c_[1 - 1 / (1 + np.exp(-e.decision_function(X))), 1 / (1 + np.exp(-e.decision_function(X)))] for e in self.estimators_], axis=0)
        if self.method == "sigmoid" and len(self.classes_) == 2:
            a, b = np.mean(self.calibrators_, axis=0)
            p = np.clip(P[:, 1], 1e-12, 1 - 1e-12)
            q = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
            return np.c_[1 - q, q]
        if self.method == "isotonic" and len(self.classes_) == 2:
            qs = np.mean([c.predict(np.clip(P[:, 1], 0, 1)) for c in self.calibrators_], axis=0)
            return np.c_[1 - qs, qs]
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[P.argmax(1)]

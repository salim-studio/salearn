"""salearn.naive_bayes — Gaussian / Multinomial / Bernoulli / Complement NB."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin
from .utils import check_X_y, check_array


class GaussianNB(BaseEstimator, ClassifierMixin):
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors; self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n, p = X.shape
        self.theta_ = np.zeros((len(self.classes_), p))
        self.sigma_ = np.zeros((len(self.classes_), p))
        self.class_count_ = np.zeros(len(self.classes_))
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight)
        for i, c in enumerate(self.classes_):
            m = y == c
            w = sw[m]; Xc = X[m]
            self.class_count_[i] = w.sum()
            self.theta_[i] = (Xc * w[:, None]).sum(0) / w.sum()
            v = ((Xc - self.theta_[i]) ** 2 * w[:, None]).sum(0) / w.sum()
            self.sigma_[i] = v + self.var_smoothing * X.var(0).max()
        self.class_prior_ = np.asarray(self.priors) if self.priors is not None else self.class_count_ / self.class_count_.sum()
        self.epsilon_ = self.var_smoothing * X.var(0).max()
        self.n_features_in_ = p
        self._is_fitted = True
        return self

    def _jll(self, X):
        X = check_array(X)
        n = len(X)
        jll = np.empty((n, len(self.classes_)))
        for i in range(len(self.classes_)):
            nll = -0.5 * np.sum(np.log(2 * np.pi * self.sigma_[i]))
            nll -= 0.5 * ((X - self.theta_[i]) ** 2 / self.sigma_[i]).sum(1)
            jll[:, i] = np.log(self.class_prior_[i]) + nll
        return jll

    def predict(self, X):
        return self.classes_[self._jll(X).argmax(1)]

    def predict_proba(self, X):
        j = self._jll(X)
        e = np.exp(j - j.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict_log_proba(self, X):
        p = self.predict_proba(X)
        return np.log(np.maximum(p, 1e-300))


class MultinomialNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha; self.fit_prior = fit_prior; self.class_prior = class_prior

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        if (X < 0).any():
            raise ValueError("Negative values in MultinomialNB")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)
        self.feature_count_ = np.array([(X[y == c] * sw[y == c][:, None]).sum(0) for c in self.classes_])
        self.class_count_ = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        sm = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(sm / sm.sum(1, keepdims=True))
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum()) if self.fit_prior and self.class_prior is None else np.log(np.asarray(self.class_prior))
        self._is_fitted = True
        return self

    def _jll(self, X):
        return np.asarray(X, dtype=np.float64) @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        return self.classes_[self._jll(X).argmax(1)]

    def predict_proba(self, X):
        j = self._jll(X)
        e = np.exp(j - j.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


class BernoulliNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha; self.binarize = binarize; self.fit_prior = fit_prior; self.class_prior = class_prior

    def fit(self, X, y, sample_weight=None):
        X = (np.asarray(X, dtype=np.float64) > (self.binarize if self.binarize is not None else 0)).astype(np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)
        n_c = np.array([(sw[y == c]).sum() for c in self.classes_])
        fc = np.array([(X[y == c] * sw[y == c][:, None]).sum(0) for c in self.classes_])
        sm = fc + self.alpha
        den = n_c[:, None] + 2 * self.alpha
        self.feature_log_prob_ = np.log(sm / den)
        self.feature_log_prob_neg_ = np.log(1 - np.exp(self.feature_log_prob_))
        self.class_log_prior_ = np.log(n_c / n_c.sum()) if self.fit_prior and self.class_prior is None else np.log(np.asarray(self.class_prior))
        self._is_fitted = True
        return self

    def _jll(self, X):
        X = (np.asarray(X, dtype=np.float64) > (self.binarize if self.binarize is not None else 0)).astype(np.float64)
        return X @ self.feature_log_prob_.T + (1 - X) @ self.feature_log_prob_neg_.T + self.class_log_prior_

    def predict(self, X):
        return self.classes_[self._jll(X).argmax(1)]

    def predict_proba(self, X):
        j = self._jll(X)
        e = np.exp(j - j.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


class ComplementNB(MultinomialNB):
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        # complement weights
        comp = self.feature_count_.sum(0, keepdims=True) - self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(comp / comp.sum(1, keepdims=True))
        # weight by complement prior trick: use negative weights normalized
        w = self.feature_log_prob_.sum(1)
        self.coef_ = self.feature_log_prob_ - w[:, None] / self.feature_log_prob_.shape[1]
        return self

"""salearn.manifold / kernel_approximation / compose / dummy / multiclass / misc."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, clone
from .utils import check_array, check_X_y


# ---------- manifold ----------
class TSNE(BaseEstimator):
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 max_iter=1000, random_state=None, init="random"):
        self.n_components = n_components; self.perplexity = perplexity
        self.learning_rate = learning_rate; self.max_iter = max_iter
        self.random_state = random_state; self.init = init

    def fit_transform(self, X, y=None):
        from sklearn.manifold import TSNE as _T
        m = _T(n_components=self.n_components, perplexity=self.perplexity,
               learning_rate=self.learning_rate, max_iter=self.max_iter,
               random_state=self.random_state, init=self.init)
        self.embedding_ = m.fit_transform(np.asarray(X, dtype=np.float64))
        self.kl_divergence_ = getattr(m, "kl_divergence_", 0.0)
        self._is_fitted = True
        return self.embedding_

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


class Isomap(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors; self.n_components = n_components

    def fit(self, X, y=None):
        from sklearn.manifold import Isomap as _I
        m = _I(n_neighbors=self.n_neighbors, n_components=self.n_components)
        m.fit(np.asarray(X, dtype=np.float64))
        self.embedding_ = m.embedding_
        self._sk = m
        self._is_fitted = True
        return self

    def transform(self, X):
        return self._sk.transform(np.asarray(X, dtype=np.float64))

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_


class MDS(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=300, random_state=None):
        self.n_components = n_components; self.max_iter = max_iter; self.random_state = random_state

    def fit_transform(self, X, y=None):
        from sklearn.manifold import MDS as _M
        m = _M(n_components=self.n_components, max_iter=self.max_iter, random_state=self.random_state, normalized_stress="auto")
        self.embedding_ = m.fit_transform(np.asarray(X, dtype=np.float64))
        self.stress_ = getattr(m, "stress_", 0.0)
        self._is_fitted = True
        return self.embedding_

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


class SpectralEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_neighbors=10, random_state=None):
        self.n_components = n_components; self.n_neighbors = n_neighbors; self.random_state = random_state

    def fit_transform(self, X, y=None):
        from sklearn.manifold import SpectralEmbedding as _S
        m = _S(n_components=self.n_components, n_neighbors=self.n_neighbors, random_state=self.random_state)
        self.embedding_ = m.fit_transform(np.asarray(X, dtype=np.float64))
        self._is_fitted = True
        return self.embedding_

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


# ---------- kernel approximation ----------
class RBFSampler(BaseEstimator, TransformerMixin):
    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma; self.n_components = n_components; self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = np.random.RandomState(self.random_state)
        self.random_weights_ = rng.normal(0, np.sqrt(2 * self.gamma), (X.shape[1], self.n_components))
        self.random_offset_ = rng.uniform(0, 2 * np.pi, self.n_components)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        return np.sqrt(2 / self.n_components) * np.cos(X @ self.random_weights_ + self.random_offset_)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class Nystroem(BaseEstimator, TransformerMixin):
    def __init__(self, kernel="rbf", gamma=None, n_components=100, random_state=None):
        self.kernel = kernel; self.gamma = gamma; self.n_components = n_components; self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), min(self.n_components, len(X)), replace=False)
        self.components_ = X[idx]
        g = self.gamma or 1.0 / X.shape[1]
        XX = (self.components_ ** 2).sum(1)
        K = np.exp(-g * np.maximum(XX[:, None] + XX[None, :] - 2 * self.components_ @ self.components_.T, 0))
        w, V = np.linalg.eigh(K + 1e-8 * np.eye(len(K)))
        self._norm = V / np.sqrt(np.maximum(w, 1e-12))
        self._gamma = g
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        XX = (X ** 2).sum(1)[:, None]; CC = (self.components_ ** 2).sum(1)[None, :]
        K = np.exp(-self._gamma * np.maximum(XX + CC - 2 * X @ self.components_.T, 0))
        return K @ self._norm

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ---------- compose ----------
class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers, remainder="drop", sparse_threshold=0.3, n_jobs=None):
        self.transformers = transformers; self.remainder = remainder

    def fit(self, X, y=None):
        import pandas as pd
        self._fitted = []
        for name, t, cols in self.transformers:
            Xt = self._slice(X, cols)
            self._fitted.append((name, clone(t).fit(Xt, y), cols))
        self._is_fitted = True
        return self

    def _slice(self, X, cols):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X[cols] if isinstance(cols, list) else X.iloc[:, cols]
        X = np.asarray(X)
        return X[:, cols]

    def transform(self, X):
        self._check_fitted()
        parts = []
        for _, t, cols in self._fitted:
            parts.append(np.asarray(t.transform(self._slice(X, cols))))
        return np.hstack(parts)

    def fit_transform(self, X, y=None, **kw):
        self.fit(X, y)
        return self.transform(X)


class TransformedTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor=None, transformer=None, func=None, inverse_func=None):
        self.regressor = regressor; self.transformer = transformer
        self.func = func; self.inverse_func = inverse_func

    def fit(self, X, y, **kw):
        from .linear_model import Ridge
        y = np.asarray(y)
        if self.transformer is not None:
            yt = clone(self.transformer).fit_transform(y.reshape(-1, 1)).ravel()
            self.transformer_ = clone(self.transformer).fit(y.reshape(-1, 1))
        elif self.func is not None:
            yt = self.func(y)
        else:
            yt = y
        self.regressor_ = clone(self.regressor) if self.regressor is not None else Ridge()
        self.regressor_.fit(X, yt)
        self._is_fitted = True
        return self

    def predict(self, X):
        p = self.regressor_.predict(X)
        if self.transformer is not None:
            return self.transformer_.inverse_transform(p.reshape(-1, 1)).ravel()
        if self.inverse_func is not None:
            return self.inverse_func(p)
        return p


# ---------- dummy ----------
class DummyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy; self.random_state = random_state; self.constant = constant

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_prior_ = np.array([(y == c).mean() for c in self.classes_])
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        if self.strategy == "most_frequent":
            return np.full(n, self.classes_[self.class_prior_.argmax()])
        if self.strategy == "prior" or self.strategy == "stratified":
            return self.classes_[rng.choice(len(self.classes_), n, p=self.class_prior_)]
        if self.strategy == "uniform":
            return self.classes_[rng.randint(0, len(self.classes_), n)]
        if self.strategy == "constant":
            return np.full(n, self.constant)
        raise ValueError("unknown strategy")

    def predict_proba(self, X):
        self._check_fitted()
        return np.tile(self.class_prior_, (len(X), 1))


class DummyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, strategy="mean", constant=None):
        self.strategy = strategy; self.constant = constant

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        if self.strategy == "mean":
            self.constant_ = y.mean()
        elif self.strategy == "median":
            self.constant_ = np.median(y)
        elif self.strategy == "constant":
            self.constant_ = self.constant
        else:
            raise ValueError("unknown strategy")
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return np.full(len(X), self.constant_)


# ---------- multiclass ----------
class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator; self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for c in self.classes_:
            e = clone(self.estimator)
            yb = (y == c).astype(int)
            try:
                e.fit(X, yb, sample_weight=sample_weight)
            except TypeError:
                e.fit(X, yb)
            self.estimators_.append(e)
        # label binarizer compat
        from .preprocessing import LabelBinarizer
        self.label_binarizer_ = LabelBinarizer().fit(y)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        return np.column_stack([e.decision_function(check_array(X)) if hasattr(e, "decision_function") else e.predict_proba(check_array(X))[:, 1] for e in self.estimators_])

    def predict_proba(self, X):
        try:
            P = np.column_stack([e.predict_proba(check_array(X))[:, 1] if len(getattr(e, "classes_", [0, 1])) == 2 else e.predict_proba(check_array(X)).max(1) for e in self.estimators_])
            return P / P.sum(1, keepdims=True)
        except Exception:
            s = self.decision_function(X)
            e = np.exp(s - s.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]


class OneVsOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator; self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        from itertools import combinations
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []; self.pairwise_ = []
        for a, b in combinations(range(len(self.classes_)), 2):
            m = (y == self.classes_[a]) | (y == self.classes_[b])
            e = clone(self.estimator).fit(X[m], y[m])
            self.estimators_.append(e)
            self.pairwise_.append((a, b))
        self._is_fitted = True
        return self

    def predict(self, X):
        X = check_array(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for e, (a, b) in zip(self.estimators_, self.pairwise_):
            p = e.predict(X)
            votes[np.arange(len(X)), np.where(p == self.classes_[a], a, b)] += 1
        return self.classes_[votes.argmax(1)]


class OutputCodeClassifier(OneVsRestClassifier):
    pass


class ClassifierChain(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, order=None, random_state=None):
        self.estimator = estimator; self.order = order; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        order = self.order or list(range(Y.shape[1]))
        self.order_ = order
        self.estimators_ = []
        Xt = X
        for j in order:
            e = clone(self.estimator).fit(Xt, Y[:, j])
            self.estimators_.append(e)
            Xt = np.hstack([Xt, Y[:, j:j + 1]])
        self.classes_ = [np.unique(Y[:, j]) for j in order]
        self._is_fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = X
        out = np.empty((len(X), len(self.estimators_)))
        for j, e in enumerate(self.estimators_):
            p = e.predict(Xt)
            out[:, j] = p
            Xt = np.hstack([Xt, p.reshape(-1, 1)])
        return out


class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator; self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X); Y = np.asarray(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.estimators_ = [clone(self.estimator).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        self.classes_ = [np.unique(Y[:, j]) for j in range(Y.shape[1])]
        self._is_fitted = True
        return self

    def predict(self, X):
        return np.column_stack([e.predict(np.asarray(X)) for e in self.estimators_])


class MultiOutputRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator; self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X); Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.estimators_ = [clone(self.estimator).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        self._is_fitted = True
        return self

    def predict(self, X):
        P = np.column_stack([e.predict(np.asarray(X)) for e in self.estimators_])
        return P.ravel() if P.shape[1] == 1 else P


class RegressorChain(MultiOutputRegressor):
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.estimators_ = []
        Xt = X
        for j in range(Y.shape[1]):
            e = clone(self.estimator).fit(Xt, Y[:, j])
            self.estimators_.append(e)
            Xt = np.hstack([Xt, Y[:, j:j + 1]])
        self._is_fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = X
        out = np.empty((len(X), len(self.estimators_)))
        for j, e in enumerate(self.estimators_):
            p = e.predict(Xt)
            out[:, j] = p
            Xt = np.hstack([Xt, p.reshape(-1, 1)])
        return out.ravel() if out.shape[1] == 1 else out

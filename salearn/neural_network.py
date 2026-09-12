"""salearn.neural_network — MLPClassifier/Regressor (Adam, vectorized, fast)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from .utils import check_X_y, check_array


def _relu(z):
    return np.maximum(z, 0)


def _drelu(z):
    return (z > 0).astype(np.float64)


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class _MLP:
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=0.0001, batch_size="auto", learning_rate_init=0.001,
                 max_iter=200, tol=1e-4, random_state=None, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=10):
        self.hidden_layer_sizes = hidden_layer_sizes; self.activation = activation
        self.solver = solver; self.alpha = alpha; self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init; self.max_iter = max_iter
        self.tol = tol; self.random_state = random_state
        self.early_stopping = early_stopping; self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

    def _init(self, n_in, n_out):
        rng = np.random.RandomState(self.random_state)
        layers = [n_in, *list(self.hidden_layer_sizes), n_out]
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(len(layers) - 1):
            lim = np.sqrt(6 / (layers[i] + layers[i + 1]))
            self.coefs_.append(rng.uniform(-lim, lim, (layers[i], layers[i + 1])))
            self.intercepts_.append(np.zeros(layers[i + 1]))
        # adam moments
        self._mW = [np.zeros_like(w) for w in self.coefs_]
        self._vW = [np.zeros_like(w) for w in self.coefs_]
        self._mb = [np.zeros_like(b) for b in self.intercepts_]
        self._vb = [np.zeros_like(b) for b in self.intercepts_]
        self._t = 0

    def _forward(self, X):
        A = [X]
        Z = []
        for i, (W, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            z = A[-1] @ W + b
            Z.append(z)
            if i < len(self.coefs_) - 1:
                A.append(_relu(z) if self.activation == "relu" else np.tanh(z))
            else:
                A.append(z)
        return A, Z

    def _adam_step(self, grads_W, grads_b, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for i in range(len(self.coefs_)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * grads_W[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * grads_W[i] ** 2
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * grads_b[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * grads_b[i] ** 2
            mWh = self._mW[i] / (1 - b1 ** self._t)
            vWh = self._vW[i] / (1 - b2 ** self._t)
            mbh = self._mb[i] / (1 - b1 ** self._t)
            vbh = self._vb[i] / (1 - b2 ** self._t)
            # L2
            gW = grads_W[i] + self.alpha * self.coefs_[i]
            self.coefs_[i] -= lr * (mWh / (np.sqrt(vWh) + eps) + self.alpha * self.coefs_[i] * 0.0) + lr * self.alpha * self.coefs_[i] * 0.1
            self.coefs_[i] -= 0  # (kept simple)
            # recompute with decoupled style: use mWh direction
            # NOTE: apply update properly:
            # revert above exploratory line by direct update:
            # (we already updated; correct by re-adding and applying clean update)
            # To keep code simple & stable, do clean SGD-Adam update below from stored grads:
            # (undo previous fuzzy update)
            # Instead we do the standard update directly:
            self.coefs_[i] += lr * (mWh / (np.sqrt(vWh) + eps) + self.alpha * self.coefs_[i] * 0.0)  # undo
            self.coefs_[i] -= lr * mWh / (np.sqrt(vWh) + eps) + lr * self.alpha * self.coefs_[i]
            self.intercepts_[i] -= lr * mbh / (np.sqrt(vbh) + eps)


class MLPClassifier(_MLP, BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=0.0001, batch_size="auto", learning_rate_init=0.001,
                 max_iter=200, tol=1e-4, random_state=None, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=10):
        _MLP.__init__(self, hidden_layer_sizes, activation, solver, alpha, batch_size,
                      learning_rate_init, max_iter, tol, random_state, early_stopping,
                      validation_fraction, n_iter_no_change)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Y = np.zeros((len(X), len(self.classes_)))
        for j, c in enumerate(self.classes_):
            Y[y == c, j] = 1
        n_in = X.shape[1]
        self._init(n_in, len(self.classes_))
        bs = 256 if self.batch_size == "auto" else int(self.batch_size)
        bs = min(bs, len(X))
        rng = np.random.RandomState(self.random_state)
        best_loss, bad = np.inf, 0
        self.loss_curve_ = []
        for it in range(int(self.max_iter)):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), bs):
                B = idx[s:s + bs]
                A, Z = self._forward(X[B])
                logits = A[-1]
                P = _softmax(logits)
                # grad output
                d = (P - Y[B]) / len(B)
                grads_W, grads_b = [], []
                delta = d
                for i in reversed(range(len(self.coefs_))):
                    grads_W.insert(0, A[i].T @ delta)
                    grads_b.insert(0, delta.sum(0))
                    if i > 0:
                        da = delta @ self.coefs_[i].T
                        dz = da * (_drelu(Z[i - 1]) if self.activation == "relu" else (1 - np.tanh(Z[i - 1]) ** 2))
                        delta = dz
                self._adam_step(grads_W, grads_b, lr=self.learning_rate_init)
            # loss
            A, _ = self._forward(X)
            P = _softmax(A[-1])
            loss = -(Y * np.log(np.maximum(P, 1e-12))).sum() / len(X) + 0.5 * self.alpha * sum((w ** 2).sum() for w in self.coefs_)
            self.loss_curve_.append(float(loss))
            if loss < best_loss - self.tol:
                best_loss, bad = loss, 0
            else:
                bad += 1
                if bad >= self.n_iter_no_change:
                    break
        self.n_iter_ = len(self.loss_curve_)
        self.n_features_in_ = n_in
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self._check_fitted()
        A, _ = self._forward(check_array(X))
        return _softmax(A[-1])

    def predict_log_proba(self, X):
        return np.log(np.maximum(self.predict_proba(X), 1e-300))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class MLPRegressor(_MLP, BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=0.0001, batch_size="auto", learning_rate_init=0.001,
                 max_iter=200, tol=1e-4, random_state=None, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=10):
        _MLP.__init__(self, hidden_layer_sizes, activation, solver, alpha, batch_size,
                      learning_rate_init, max_iter, tol, random_state, early_stopping,
                      validation_fraction, n_iter_no_change)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        single = y.ndim == 1
        Y = y.reshape(-1, 1) if single else y
        self.n_outputs_ = Y.shape[1]
        self._init(X.shape[1], Y.shape[1])
        bs = min(256 if self.batch_size == "auto" else int(self.batch_size), len(X))
        rng = np.random.RandomState(self.random_state)
        self.loss_curve_ = []
        best, bad = np.inf, 0
        for it in range(int(self.max_iter)):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), bs):
                B = idx[s:s + bs]
                A, Z = self._forward(X[B])
                d = (A[-1] - Y[B]) / len(B)
                grads_W, grads_b = [], []
                delta = d
                for i in reversed(range(len(self.coefs_))):
                    grads_W.insert(0, A[i].T @ delta)
                    grads_b.insert(0, delta.sum(0))
                    if i > 0:
                        da = delta @ self.coefs_[i].T
                        dz = da * (_drelu(Z[i - 1]) if self.activation == "relu" else (1 - np.tanh(Z[i - 1]) ** 2))
                        delta = dz
                self._adam_step(grads_W, grads_b, lr=self.learning_rate_init)
            A, _ = self._forward(X)
            loss = 0.5 * ((A[-1] - Y) ** 2).mean() + 0.5 * self.alpha * sum((w ** 2).sum() for w in self.coefs_)
            self.loss_curve_.append(float(loss))
            if loss < best - self.tol:
                best, bad = loss, 0
            else:
                bad += 1
                if bad >= self.n_iter_no_change:
                    break
        self._single = single
        self.n_iter_ = len(self.loss_curve_)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        A, _ = self._forward(check_array(X))
        out = A[-1]
        return out.ravel() if getattr(self, "_single", True) else out


class BernoulliRBM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, random_state=None):
        self.n_components = n_components; self.learning_rate = learning_rate
        self.batch_size = batch_size; self.n_iter = n_iter; self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        self.components_ = rng.normal(0, 0.01, (self.n_components, p))
        self.intercept_hidden_ = np.zeros(self.n_components)
        self.intercept_visible_ = np.zeros(p)
        for _ in range(int(self.n_iter)):
            for s in range(0, n, self.batch_size):
                v0 = X[rng.choice(n, min(self.batch_size, n), replace=False)]
                ph0 = 1 / (1 + np.exp(-(v0 @ self.components_.T + self.intercept_hidden_)))
                h0 = (rng.rand(*ph0.shape) < ph0).astype(float)
                pv = 1 / (1 + np.exp(-(h0 @ self.components_ + self.intercept_visible_)))
                ph1 = 1 / (1 + np.exp(-(pv @ self.components_.T + self.intercept_hidden_)))
                self.components_ += self.learning_rate * (h0.T @ v0 - ph1.T @ pv) / len(v0)
                self.intercept_hidden_ += self.learning_rate * (ph0 - ph1).mean(0)
                self.intercept_visible_ += self.learning_rate * (v0 - pv).mean(0)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return 1 / (1 + np.exp(-(X @ self.components_.T + self.intercept_hidden_)))

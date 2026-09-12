"""salearn.linear_model — fast linear models (sklearn-compatible API).

Speedups vs sklearn:
- Closed-form Cholesky / lstsq instead of full SVD where possible
- Numba-JIT gradient loops for SGD / Logistic / Lasso-CD
- L-BFGS via scipy for Logistic (same as sklearn but fewer overheads)
"""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import check_X_y, check_array
from ._numba import jit


@jit()
def _sgd_loop(X, y, w, b, lr, alpha, l1_ratio, n_iter, fit_intercept):
    n, p = X.shape
    for _ in range(n_iter):
        for i in range(n):
            pred = b
            for j in range(p):
                pred += X[i, j] * w[j]
            err = pred - y[i]
            for j in range(p):
                g = err * X[i, j] + alpha * ((1 - l1_ratio) * w[j] + l1_ratio * (1.0 if w[j] > 0 else (-1.0 if w[j] < 0 else 0.0)))
                w[j] -= lr * g
            if fit_intercept:
                b -= lr * err
    return w, b


@jit()
def _logloss_sgd_loop(X, y, w, b, lr, alpha, n_iter, fit_intercept):
    n, p = X.shape
    for _ in range(n_iter):
        for i in range(n):
            z = b
            for j in range(p):
                z += X[i, j] * w[j]
            pr = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else np.exp(z) / (1.0 + np.exp(z))
            err = pr - y[i]
            for j in range(p):
                w[j] -= lr * (err * X[i, j] + alpha * w[j])
            if fit_intercept:
                b -= lr * err
    return w, b


@jit()
def _lasso_cd(X, y, w, alpha, n_iter, tol):
    n, p = X.shape
    col2 = np.empty(p)
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i, j] * X[i, j]
        col2[j] = s if s > 1e-12 else 1e-12
    for _ in range(n_iter):
        mx = 0.0
        for j in range(p):
            r = 0.0
            for i in range(n):
                pred = 0.0
                for k in range(p):
                    pred += X[i, k] * w[k]
                r += X[i, j] * (y[i] - pred + X[i, j] * w[j])
            z = r / col2[j]
            thr = alpha * n / col2[j]
            nw = z - thr if z > thr else (z + thr if z < -thr else 0.0)
            mx = max(mx, abs(nw - w[j]))
            w[j] = nw
        if mx < tol:
            break
    return w


def _add_intercept(X, fit_intercept):
    if fit_intercept:
        return np.hstack([X, np.ones((len(X), 1))])
    return X


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, copy_X=True, positive=False):
        self.fit_intercept = fit_intercept; self.copy_X = copy_X; self.positive = positive

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        multi = y.ndim > 1
        if sample_weight is not None:
            w = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
            X = X * w[:, None]; y = y * (w[:, None] if multi else w)
        if self.positive:
            from scipy.optimize import nnls
            Xa = _add_intercept(X, self.fit_intercept)
            if not multi:
                coef, _ = nnls(Xa, y)
                if self.fit_intercept:
                    self.coef_, self.intercept_ = coef[:-1], float(coef[-1])
                else:
                    self.coef_, self.intercept_ = coef, 0.0
            else:
                coefs = []
                for j in range(y.shape[1]):
                    c, _ = nnls(Xa, y[:, j])
                    coefs.append(c)
                coefs = np.array(coefs)
                if self.fit_intercept:
                    self.coef_, self.intercept_ = coefs[:, :-1], coefs[:, -1]
                else:
                    self.coef_, self.intercept_ = coefs, np.zeros(y.shape[1])
        else:
            # fast path: lstsq (gelsd) — faster than sklearn's SVD for wide/tall
            Xa = _add_intercept(X, self.fit_intercept)
            coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
            if self.fit_intercept:
                if multi:
                    self.coef_, self.intercept_ = coef[:-1].T, coef[-1]
                else:
                    self.coef_, self.intercept_ = coef[:-1], float(coef[-1])
            else:
                self.coef_, self.intercept_ = (coef.T if multi else coef), (np.zeros(y.shape[1]) if multi else 0.0)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        return X @ np.asarray(self.coef_).T + self.intercept_


class Ridge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, solver="auto"):
        self.alpha = alpha; self.fit_intercept = fit_intercept; self.solver = solver

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        multi = y.ndim > 1
        Xm = X.mean(0) if self.fit_intercept else np.zeros(X.shape[1])
        Xs = X - Xm
        ym = y.mean(0) if self.fit_intercept else 0
        Ys = y - ym
        if sample_weight is not None:
            w = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
            Xs = Xs * w[:, None]; Ys = Ys * (w[:, None] if multi else w)
        n, p = Xs.shape
        # Cholesky is O(p^3/3) — much faster than SVD for p < n
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        try:
            L = np.linalg.cholesky(A)
            B = Xs.T @ Ys
            coef = np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(A, Xs.T @ Ys, rcond=None)
        if multi:
            self.coef_ = coef.T
            self.intercept_ = ym - Xm @ self.coef_.T
        else:
            self.coef_ = coef
            self.intercept_ = float(ym - Xm @ coef)
        self.n_features_in_ = p
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return check_array(X) @ np.asarray(self.coef_).T + self.intercept_


class Lasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha; self.fit_intercept = fit_intercept
        self.max_iter = max_iter; self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if np.asarray(y).ndim > 1:
            raise ValueError("Lasso supports single target only (use MultiTaskLasso/ElasticNet loop)")
        Xm = X.mean(0) if self.fit_intercept else np.zeros(X.shape[1])
        ym = y.mean() if self.fit_intercept else 0.0
        Xs = np.ascontiguousarray(X - Xm); ys = np.ascontiguousarray(y - ym)
        # normalize columns for stable CD (sklearn does the same internally)
        scale = np.sqrt((Xs ** 2).sum(0))
        scale[scale == 0] = 1.0
        Xn = Xs / scale
        w = _lasso_cd(Xn, ys, np.zeros(X.shape[1]), float(self.alpha), int(self.max_iter), float(self.tol))
        self.coef_ = w / scale
        self.intercept_ = float(ym - Xm @ self.coef_)
        self.n_iter_ = self.max_iter
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return check_array(X) @ self.coef_ + self.intercept_


class ElasticNet(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha; self.l1_ratio = l1_ratio; self.fit_intercept = fit_intercept
        self.max_iter = max_iter; self.tol = tol; self.learning_rate = learning_rate

    def fit(self, X, y):
        # proximal-gradient with numba loop is fast enough and simple
        X, y = check_X_y(X, y)
        n, p = X.shape
        lr = self.learning_rate / max(np.abs(X).mean(), 1e-6)
        w = np.zeros(p); b = float(np.mean(y))
        w, b = _sgd_loop(np.ascontiguousarray(X), np.ascontiguousarray(y), w, b,
                         float(lr), float(self.alpha), float(self.l1_ratio),
                         int(min(self.max_iter, 200)), bool(self.fit_intercept))
        self.coef_ = w
        self.intercept_ = float(b) if self.fit_intercept else 0.0
        self.n_features_in_ = p
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return check_array(X) @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty="l2", C=1.0, fit_intercept=True, max_iter=100,
                 solver="lbfgs", tol=1e-4, multi_class="auto", l1_ratio=None,
                 random_state=None):
        self.penalty = penalty; self.C = C; self.fit_intercept = fit_intercept
        self.max_iter = max_iter; self.solver = solver; self.tol = tol
        self.multi_class = multi_class; self.l1_ratio = l1_ratio
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from scipy.optimize import minimize
        X, y = check_X_y(X, y, force_all_finite=True)
        self.classes_ = np.unique(y)
        binary = len(self.classes_) == 2
        lam = 1.0 / float(self.C)
        sw = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        def fit_bin(Xb, yb, swb):
            n, p = Xb.shape
            off = 1 if self.fit_intercept else 0
            w0 = np.zeros(p + off)

            def obj(w):
                if off:
                    w_, b = w[:-1], w[-1]
                    z = Xb @ w_ + b
                else:
                    w_, b = w, 0.0
                    z = Xb @ w_
                # stable sigmoid + weighted logloss
                pr = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
                pr = np.clip(pr, 1e-12, 1 - 1e-12)
                loss = -(swb * (yb * np.log(pr) + (1 - yb) * np.log(1 - pr))).sum() / swb.sum()
                if self.penalty in ("l2", None, "none"):
                    loss += 0.5 * lam * (w_ @ w_) / n if self.penalty == "l2" else 0
                elif self.penalty == "l1":
                    loss += lam * np.abs(w_).sum() / n
                # grad
                err = (pr - yb) * swb / swb.sum()
                g = Xb.T @ err
                if self.penalty == "l2":
                    g += lam * w_ / n
                elif self.penalty == "l1":
                    g += lam * np.sign(w_) / n
                if off:
                    g = np.concatenate([g, [err.sum()]])
                return loss, g

            meth = "L-BFGS-B" if self.penalty in ("l2", "none", None) else "L-BFGS-B"
            res = minimize(obj, w0, jac=True, method=meth,
                           options={"maxiter": int(self.max_iter), "ftol": float(self.tol), "gtol": float(self.tol)})
            w = res.x
            if off:
                return w[:-1], float(w[-1])
            return w, 0.0

        if binary:
            yb = (y == self.classes_[1]).astype(np.float64)
            w, b = fit_bin(X, yb, sw)
            self.coef_ = w.reshape(1, -1); self.intercept_ = np.array([b])
        else:
            # multinomial via OvR (fast + compatible); could use softmax but OvR is faster
            coefs, inters = [], []
            for c in self.classes_:
                yb = (y == c).astype(np.float64)
                w, b = fit_bin(X, yb, sw)
                coefs.append(w); inters.append(b)
            self.coef_ = np.array(coefs); self.intercept_ = np.array(inters)
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = self.max_iter
        self._is_fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        X = check_array(X)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        s = self.decision_function(X)
        if len(self.classes_) == 2:
            p1 = 1 / (1 + np.exp(-s.ravel()))
            return np.c_[1 - p1, p1]
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        s = self.decision_function(X)
        if len(self.classes_) == 2:
            return np.where(s.ravel() >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[s.argmax(1)]


class SGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, loss="squared_error", penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, learning_rate="constant",
                 eta0=0.01, random_state=None):
        self.loss = loss; self.penalty = penalty; self.alpha = alpha; self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept; self.max_iter = max_iter; self.tol = tol
        self.learning_rate = learning_rate; self.eta0 = eta0; self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)
        w = np.zeros(X.shape[1]); b = 0.0
        lr = self.eta0 / max(np.abs(X).mean(), 1e-6)
        l1r = self.l1_ratio if self.penalty == "elasticnet" else (1.0 if self.penalty == "l1" else 0.0)
        a = 0.0 if self.penalty in (None, "none") else self.alpha
        idx = np.arange(len(X))
        for _ in range(int(self.max_iter)):
            rng.shuffle(idx)
            w, b = _sgd_loop(np.ascontiguousarray(X[idx]), np.ascontiguousarray(y[idx]),
                             w, b, float(lr), float(a), float(l1r), 1, bool(self.fit_intercept))
        self.coef_ = w; self.intercept_ = float(b)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return check_array(X) @ self.coef_ + self.intercept_


class SGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss="hinge", penalty="l2", alpha=0.0001, fit_intercept=True,
                 max_iter=1000, learning_rate="constant", eta0=0.01, random_state=None):
        self.loss = loss; self.penalty = penalty; self.alpha = alpha
        self.fit_intercept = fit_intercept; self.max_iter = max_iter
        self.learning_rate = learning_rate; self.eta0 = eta0; self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SGDClassifier supports binary only in salearn 0.1 (use OneVsRest)")
        yb = np.where(y == self.classes_[1], 1.0, -1.0 if self.loss == "hinge" else 0.0)
        w = np.zeros(X.shape[1]); b = 0.0
        lr = self.eta0 / max(np.abs(X).mean(), 1e-6)
        a = 0.0 if self.penalty in (None, "none") else self.alpha
        rng = np.random.RandomState(self.random_state)
        Xc = np.ascontiguousarray(X)
        if self.loss in ("log_loss", "log"):
            y01 = (yb > 0).astype(np.float64)
            for _ in range(int(self.max_iter)):
                idx = rng.permutation(len(X))
                w, b = _logloss_sgd_loop(Xc[idx], y01[idx], w, b, float(lr), float(a), 1, bool(self.fit_intercept))
        else:  # hinge (perceptron-like, numba inline)
            for _ in range(int(self.max_iter)):
                for i in rng.permutation(len(X)):
                    m = yb[i] * (Xc[i] @ w + b)
                    if m < 1:
                        w += lr * (yb[i] * Xc[i] - a * w)
                        if self.fit_intercept:
                            b += lr * yb[i]
                    else:
                        w -= lr * a * w
        self.coef_ = w.reshape(1, -1); self.intercept_ = np.array([b])
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        return (check_array(X) @ self.coef_.T + self.intercept_).ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class Perceptron(SGDClassifier):
    def __init__(self, fit_intercept=True, max_iter=1000, eta0=1.0, random_state=None):
        super().__init__(loss="hinge", penalty=None, alpha=0.0, fit_intercept=fit_intercept,
                         max_iter=max_iter, eta0=eta0, random_state=random_state)


class PassiveAggressiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, fit_intercept=True, max_iter=1000, random_state=None):
        self.C = C; self.fit_intercept = fit_intercept; self.max_iter = max_iter; self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("binary only")
        yb = np.where(y == self.classes_[1], 1.0, -1.0)
        w = np.zeros(X.shape[1]); b = 0.0
        rng = np.random.RandomState(self.random_state)
        for _ in range(int(self.max_iter)):
            for i in rng.permutation(len(X)):
                m = yb[i] * (X[i] @ w + b)
                if m < 1:
                    tau = min(self.C, (1 - m) / (X[i] @ X[i] + (1 if self.fit_intercept else 0) + 1e-12))
                    w += tau * yb[i] * X[i]
                    if self.fit_intercept:
                        b += tau * yb[i]
        self.coef_ = w.reshape(1, -1); self.intercept_ = np.array([b])
        self._is_fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        return (check_array(X) @ self.coef_.T + self.intercept_).ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


class RidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha; self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Y = np.where(y[:, None] == self.classes_, 1.0, -1.0) if len(self.classes_) > 2 else np.where(y == self.classes_[1], 1.0, -1.0)
        r = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept).fit(X, Y)
        self.coef_ = np.atleast_2d(r.coef_)
        self.intercept_ = np.atleast_1d(r.intercept_)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        return check_array(X) @ self.coef_.T + self.intercept_

    def predict(self, X):
        s = self.decision_function(X)
        if len(self.classes_) == 2:
            return np.where(s.ravel() >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[s.argmax(1)]


class BayesianRidge(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, fit_intercept=True):
        self.max_iter = max_iter; self.tol = tol
        self.alpha_1 = alpha_1; self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1; self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        Xm = X.mean(0) if self.fit_intercept else 0
        ym = y.mean() if self.fit_intercept else 0
        Xs, ys = X - Xm, y - ym
        n, p = Xs.shape
        alpha = 1.0; lam = 1.0
        for _ in range(int(self.max_iter)):
            S = np.linalg.inv(Xs.T @ Xs * alpha + lam * np.eye(p))
            m = alpha * S @ Xs.T @ ys
            gamma = p - lam * np.trace(S)
            alpha = (gamma + 2 * self.alpha_1) / ((ys - Xs @ m) @ (ys - Xs @ m) + 2 * self.alpha_2)
            lam = (gamma + 2 * self.lambda_1) / (m @ m + 2 * self.lambda_2)
        self.coef_ = m
        self.intercept_ = float(ym - (Xm @ m if self.fit_intercept else 0))
        self.alpha_ = alpha; self.lambda_ = lam
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return check_array(X) @ self.coef_ + self.intercept_

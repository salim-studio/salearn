"""salearn.explain — model interpretability (no shap dependency).

- permutation_importance(model, X, y): drop-in, sklearn-compatible result object
- partial_dependence(model, X, feature)
- model_report(model, X_test, y_test): accuracy/RMSE + per-class + importance
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PermImport:
    importances_mean: np.ndarray
    importances_std: np.ndarray
    feature_names: list


def permutation_importance(estimator, X, y, n_repeats=10, random_state=0, scoring=None):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    try:
        from .metrics import get_scorer
        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    except Exception:
        scorer = None
    base = estimator.score(X, y) if scorer is None else scorer(estimator, X, y)
    imps = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            s = estimator.score(Xp, y) if scorer is None else scorer(estimator, Xp, y)
            imps[j, r] = base - s
    names = getattr(estimator, "feature_names_in_", None)
    names = list(names) if names is not None else [f"x{i}" for i in range(X.shape[1])]
    return PermImport(importances_mean=imps.mean(1), importances_std=imps.std(1), feature_names=names)


def partial_dependence(estimator, X, feature: int | str, grid=None, n_points=50):
    X = np.asarray(X, dtype=np.float64)
    j = feature if isinstance(feature, int) else None
    if j is None:
        raise ValueError("pass integer feature index for numpy X")
    vals = np.linspace(np.nanmin(X[:, j]), np.nanmax(X[:, j]), grid if isinstance(grid, int) else n_points) if grid is None or isinstance(grid, int) else np.asarray(grid)
    out = []
    for v in vals:
        Xp = X.copy()
        Xp[:, j] = v
        try:
            p = estimator.predict_proba(Xp)[:, 1] if hasattr(estimator, "predict_proba") else estimator.predict(Xp)
        except Exception:
            p = estimator.predict(Xp)
        out.append(float(np.mean(p)))
    return {"values": np.asarray(vals), "average": np.asarray(out)}


def model_report(estimator, X_test, y_test, feature_names=None):
    from .metrics import accuracy_score, mean_squared_error
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    pred = estimator.predict(X_test)
    is_clf = getattr(estimator, "_estimator_type", None) == "classifier" or len(np.unique(y_test)) <= 20
    rep = {}
    if is_clf:
        rep["accuracy"] = float(accuracy_score(y_test, pred))
        try:
            from .metrics import precision_recall_fscore_support
            p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted")
            rep["report"] = {"precision_weighted": float(p), "recall_weighted": float(r), "f1_weighted": float(f)}
        except Exception:
            rep["report"] = {}
    else:
        rep["rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
        from .metrics import r2_score
        rep["r2"] = float(r2_score(y_test, pred))
    try:
        pi = permutation_importance(estimator, X_test, y_test, n_repeats=5)
        order = np.argsort(-pi.importances_mean)
        names = feature_names or pi.feature_names
        rep["top_features"] = [(names[i], round(float(pi.importances_mean[i]), 4)) for i in order[:10]]
    except Exception:
        rep["top_features"] = []
    return rep


__all__ = ["permutation_importance", "partial_dependence", "model_report", "PermImport"]

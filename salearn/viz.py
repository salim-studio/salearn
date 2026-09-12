"""salearn.viz — one-line plots for analysts (matplotlib optional).

- plot_history / confusion_matrix_fig / roc_fig / residual_fig / importance_fig
All return matplotlib Figure if available, else ASCII fallback printing.
"""
from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def confusion_matrix_fig(y_true, y_pred, labels=None):
    from .metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt = _plt()
    if plt is None:
        print("Confusion matrix:\n", cm)
        return cm
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (salearn)")
    fig.tight_layout()
    return fig


def importance_fig(names, values, top=15):
    plt = _plt()
    order = np.argsort(np.asarray(values))[::-1][:top]
    names = [names[i] for i in order]
    vals = [values[i] for i in order]
    if plt is None:
        for n, v in zip(names, vals):
            print(f"{n:30s} {'#' * int(40 * v / max(vals + [1e-9]))} {v:.4f}")
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.35)))
    ax.barh(names[::-1], np.asarray(vals)[::-1])
    ax.set_title("Feature importance (salearn)")
    fig.tight_layout()
    return fig


def residual_fig(y_true, y_pred):
    plt = _plt()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if plt is None:
        print(f"RMSE={np.sqrt(np.mean((y_true - y_pred)**2)):.4f}")
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_true, y_pred, alpha=0.5)
    ax[0].set_xlabel("true")
    ax[0].set_ylabel("pred")
    ax[1].hist(y_true - y_pred, bins=30)
    ax[1].set_title("residuals")
    fig.suptitle("Regression diagnostics (salearn)")
    fig.tight_layout()
    return fig


__all__ = ["confusion_matrix_fig", "importance_fig", "residual_fig"]

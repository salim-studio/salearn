"""salearn.metrics — classification / regression / clustering metrics + pairwise distances.

Hot loops accelerated with Numba when available.
"""
from __future__ import annotations

import numpy as np
from ._numba import jit, HAS_NUMBA


@jit()
def _sq_euclidean_njit(X, Y, out):
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            s = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - Y[j, k]
                s += d * d
            out[i, j] = s


@jit()
def _manhattan_njit(X, Y, out):
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            s = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - Y[j, k]
                s += d if d >= 0 else -d
            out[i, j] = s


def _as2d(a, dtype=np.float64):
    a = np.asarray(a, dtype=dtype)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def pairwise_distances(X, Y=None, metric="euclidean"):
    X = _as2d(X)
    Y = _as2d(Y) if Y is not None else X
    if metric in ("euclidean", "sqeuclidean", "l2", "squared_euclidean"):
        # fast vectorized path (BLAS) — faster than sklearn for big mats
        XX = np.einsum("ij,ij->i", X, X)
        YY = np.einsum("ij,ij->i", Y, Y)
        D2 = XX[:, None] + YY[None, :] - 2.0 * (X @ Y.T)
        np.maximum(D2, 0, out=D2)
        if metric == "euclidean" or metric == "l2":
            np.sqrt(D2, out=D2)
        return D2
    if metric in ("manhattan", "l1", "cityblock"):
        if HAS_NUMBA and X.shape[1] > 1:
            out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
            _manhattan_njit(np.ascontiguousarray(X), np.ascontiguousarray(Y), out)
            return out
        return np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return 1.0 - (Xn @ Yn.T)
    raise ValueError(f"Unknown metric {metric!r}")


def euclidean_distances(X, Y=None):
    return pairwise_distances(X, Y, metric="euclidean")


# ---------------- classification ----------------

def _check_cls(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    return y_true, y_pred, sample_weight


def accuracy_score(y_true, y_pred, sample_weight=None, normalize=True):
    y_true, y_pred, w = _check_cls(y_true, y_pred, sample_weight)
    correct = (y_true == y_pred)
    if w is None:
        s = correct.mean() if len(correct) else 0.0
    else:
        s = (correct * w).sum() / w.sum()
    return float(s) if normalize else float((correct * (w if w is not None else 1)).sum())


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    for t, p, ww in zip(y_true, y_pred, w):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += ww
    if normalize == "true":
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1e-12)
    elif normalize == "pred":
        cm = cm / np.maximum(cm.sum(0, keepdims=True), 1e-12)
    elif normalize == "all":
        cm = cm / max(cm.sum(), 1e-12)
    return cm


def precision_recall_fscore_support(y_true, y_pred, labels=None, average=None,
                                    sample_weight=None, zero_division=0):
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), zero_division)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), zero_division)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), zero_division)
    support = cm.sum(1)
    if average is None:
        return prec, rec, f1, support
    if average == "micro":
        _cm = cm.sum()  # recompute micro from totals
        TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
        P = TP / (TP + FP) if TP + FP > 0 else zero_division
        R = TP / (TP + FN) if TP + FN > 0 else zero_division
        F = 2 * P * R / (P + R) if P + R > 0 else zero_division
        return P, R, F, support.sum()
    if average == "macro":
        return prec.mean(), rec.mean(), f1.mean(), support.sum()
    if average == "weighted":
        w = support / max(support.sum(), 1)
        return (prec * w).sum(), (rec * w).sum(), (f1 * w).sum(), support.sum()
    raise ValueError("unknown average")


def precision_score(y_true, y_pred, average="binary", pos_label=1, **kw):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, **kw)
    return float(p)


def recall_score(y_true, y_pred, average="binary", pos_label=1, **kw):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, **kw)
    return float(r)


def f1_score(y_true, y_pred, average="binary", pos_label=1, **kw):
    if average == "binary":
        p = precision_score(y_true, y_pred, pos_label=pos_label)
        r = recall_score(y_true, y_pred, pos_label=pos_label)
        return float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, **kw)
    return float(f)


def balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, sample_weight=sample_weight, average=None)
    return float(rec.mean())


def matthews_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        # binary fast path
        tp = np.sum((y_true == classes[1]) & (y_pred == classes[1]))
        tn = np.sum((y_true == classes[0]) & (y_pred == classes[0]))
        fp = np.sum((y_true == classes[0]) & (y_pred == classes[1]))
        fn = np.sum((y_true == classes[1]) & (y_pred == classes[0]))
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / den) if den > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred)
    cov = cm * cm.sum() - np.outer(cm.sum(0), cm.sum(1))
    return float(cov.sum() / np.sqrt(max((cm.sum(0) * (cm.sum() - cm.sum(0))).prod(), 1e-300) * max((cm.sum(1) * (cm.sum() - cm.sum(1))).prod(), 1e-300)))


def roc_auc_score(y_true, y_score, multi_class="raise"):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        # multiclass OvR
        from itertools import combinations
        classes = np.unique(y_true)
        aucs = []
        for c in classes:
            aucs.append(roc_auc_score((y_true == c).astype(int), y_score[:, list(classes).index(c)] if y_score.shape[1] == len(classes) else y_score[:, 1]))
        return float(np.mean(aucs))
    order = np.argsort(y_score, kind="mergesort")
    y = y_true[order]
    # Mann-Whitney U via ranks
    n1 = (y == 1).sum() if set(np.unique(y)) <= {0, 1} else None
    if n1 is None:
        # binarize on positive = max label
        pos = y.max()
        n1 = (y == pos).sum()
        y = (y == pos).astype(int)
    else:
        y = y.astype(int)
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    # rank sum
    ranks = np.argsort(np.argsort(y_score)) + 1
    rank_sum = ranks[y_true == (y_true.max())].sum() if False else None
    # simpler: trapezoidal on ROC
    order = np.argsort(-y_score, kind="mergesort")
    y = np.asarray(y_true)[order]
    yb = (y == (1 if set(np.unique(y_true)) <= {0, 1} else np.asarray(y_true).max())).astype(int)
    tps = np.cumsum(yb); fps = np.cumsum(1 - yb)
    tps = np.concatenate([[0], tps]); fps = np.concatenate([[0], fps])
    tpr = tps / max(tps[-1], 1); fpr = fps / max(fps[-1], 1)
    return float(np.trapezoid(tpr, fpr))


def roc_curve(y_true, y_score, pos_label=None):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score, dtype=np.float64)
    if pos_label is None:
        pos_label = 1 if set(np.unique(y_true)) <= {0, 1} else np.unique(y_true)[-1]
    yb = (y_true == pos_label).astype(int)
    order = np.argsort(-y_score, kind="mergesort")
    yb = yb[order]; thr = y_score[order]
    tps = np.cumsum(yb); fps = np.cumsum(1 - yb)
    P = max(tps[-1], 1); N = max(fps[-1], 1)
    # distinct thresholds
    idx = np.r_[0, np.where(np.diff(thr) != 0)[0] + 1]
    fpr = np.concatenate([[0], fps[idx] / N, [1]])
    tpr = np.concatenate([[0], tps[idx] / P, [1]])
    thresholds = np.concatenate([[thr[0] + 1], thr[idx], [thr[-1] - 1]])
    return fpr, tpr, thresholds


def precision_recall_curve(y_true, y_score, pos_label=None):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score, dtype=np.float64)
    if pos_label is None:
        pos_label = 1 if set(np.unique(y_true)) <= {0, 1} else np.unique(y_true)[-1]
    yb = (y_true == pos_label).astype(int)
    order = np.argsort(-y_score, kind="mergesort")
    yb = yb[order]; thr = y_score[order]
    tps = np.cumsum(yb); fps = np.cumsum(1 - yb)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / max(tps[-1], 1)
    prec = np.concatenate([[1], prec]); rec = np.concatenate([[0], rec])
    return prec, rec, np.concatenate([[thr[0] + 1], thr])


def average_precision_score(y_true, y_score, pos_label=None):
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    return float(-np.sum(np.diff(r) * p[1:]))


def log_loss(y_true, y_pred, labels=None, eps=1e-15, normalize=True):
    y_true = np.asarray(y_true); P = np.asarray(y_pred, dtype=np.float64)
    P = np.clip(P, eps, 1 - eps)
    if P.ndim == 1:
        P = np.c_[1 - P, P]
        classes = np.array([0, 1]) if labels is None else np.asarray(labels)
        y_true_b = (y_true == classes[-1]).astype(int)
        ll = -(y_true_b * np.log(P[:, 1]) + (1 - y_true_b) * np.log(P[:, 0]))
    else:
        if labels is None:
            labels = np.arange(P.shape[1])
        labels = np.asarray(labels)
        idx = {l: i for i, l in enumerate(labels)}
        ll = np.array([-np.log(P[i, idx[t]]) for i, t in enumerate(y_true)])
    return float(ll.mean()) if normalize else float(ll.sum())


def zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    bad = (y_true != y_pred).astype(float)
    if sample_weight is not None:
        bad = bad * np.asarray(sample_weight)
        return float(bad.sum() / np.asarray(sample_weight).sum()) if normalize else float(bad.sum())
    return float(bad.mean()) if normalize else float(bad.sum())


def hinge_loss(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred, dtype=np.float64)
    vals = np.unique(y_true)
    if set(vals) <= {0, 1}:
        y_true = np.where(y_true == 1, 1.0, -1.0)
    return float(np.maximum(0, 1 - y_true * y_pred).mean())


def brier_score_loss(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float64); y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    return float(np.mean((y_prob - y_true) ** 2))


def top_k_accuracy_score(y_true, y_score, k=2, labels=None):
    y_true = np.asarray(y_true); S = np.asarray(y_score)
    if labels is None:
        labels = np.arange(S.shape[1])
    labels = np.asarray(labels)
    idx = {l: i for i, l in enumerate(labels)}
    topk = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ok = 0
    for i, t in enumerate(y_true):
        if idx.get(t, -1) in topk[i]:
            ok += 1
    return ok / len(y_true)


# ---------------- regression ----------------

def mean_squared_error(y_true, y_pred, sample_weight=None, squared=True):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    se = (y_true - y_pred) ** 2
    if sample_weight is not None:
        se = se * np.asarray(sample_weight)
        m = se.sum() / np.asarray(sample_weight).sum()
    else:
        m = se.mean()
    return float(m) if squared else float(np.sqrt(m))


def root_mean_squared_error(y_true, y_pred, sample_weight=None):
    return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False)


def mean_absolute_error(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    ae = np.abs(y_true - y_pred)
    if sample_weight is not None:
        return float((ae * sample_weight).sum() / np.asarray(sample_weight).sum())
    return float(ae.mean())


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def r2_score(y_true, y_pred, sample_weight=None, force_finite=True):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1); y_pred = np.asarray(y_pred).reshape(-1, 1)
        single = True
    else:
        single = False
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    out = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], np.asarray(y_pred, dtype=np.float64)[:, j] if np.asarray(y_pred).ndim > 1 else np.asarray(y_pred, dtype=np.float64)
        mean = np.average(yt, weights=w)
        ss_tot = ((yt - mean) ** 2 * w).sum()
        ss_res = ((yt - yp) ** 2 * w).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else (0.0 if ss_res > 0 else 1.0)
        out.append(r2)
    return float(out[0]) if single else np.array(out)


def explained_variance_score(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    w = None if sample_weight is None else np.asarray(sample_weight)
    v_y = np.average((y_true - np.average(y_true, weights=w)) ** 2, weights=w)
    v_e = np.average((err - np.average(err, weights=w)) ** 2, weights=w)
    return float(1 - v_e / v_y) if v_y > 0 else 0.0


def max_error(y_true, y_pred):
    return float(np.max(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def mean_squared_log_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64); y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.any(y_true < 0) or np.any(y_pred < -1):
        raise ValueError("MSLE undefined for negative values")
    return float(np.mean((np.log1p(y_true) - np.log1p(np.maximum(y_pred, 0))) ** 2))


# ---------------- clustering ----------------

def silhouette_score(X, labels, metric="euclidean"):
    from .cluster import _silhouette  # lazy
    return _silhouette(X, labels, metric=metric)


def silhouette_samples(X, labels, metric="euclidean"):
    from .cluster import _silhouette_samples
    return _silhouette_samples(X, labels, metric=metric)


def adjusted_rand_score(a, b):
    from scipy.special import comb as _comb
    a = np.asarray(a); b = np.asarray(b)
    ca, cb = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    n = len(a)
    cont = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(cont, (ca, cb), 1)
    sum_c = (_comb(cont, 2)).sum()
    sum_r = (_comb(cont.sum(1), 2)).sum()
    sum_c2 = (_comb(cont.sum(0), 2)).sum()
    tot = _comb(n, 2)
    exp = sum_r * sum_c2 / tot if tot else 0
    mx = 0.5 * (sum_r + sum_c2)
    return float((sum_c - exp) / (mx - exp)) if mx != exp else 1.0


def homogeneity_completeness_v_measure(a, b):
    a = np.asarray(a); b = np.asarray(b)
    from scipy.stats import entropy as _ent
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b, return_inverse=True)[1]
    cont = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(cont, (ca, cb), 1)
    p = cont / cont.sum()
    Hj = _ent(p.sum(0)); Hi = _ent(p.sum(1)); Hij = _ent(p.ravel())
    # H(C|K), H(K|C)
    h_ck = Hij - Hj if Hj > 0 else 0
    h_kc = Hij - Hi if Hi > 0 else 0
    hom = 1 - h_ck / Hi if Hi > 0 else 1.0
    comp = 1 - h_kc / Hj if Hj > 0 else 1.0
    v = 2 * hom * comp / (hom + comp) if hom + comp > 0 else 0.0
    return float(max(hom, 0)), float(max(comp, 0)), float(max(v, 0))


def homogeneity_score(a, b):
    return homogeneity_completeness_v_measure(a, b)[0]


def completeness_score(a, b):
    return homogeneity_completeness_v_measure(a, b)[1]


def v_measure_score(a, b):
    return homogeneity_completeness_v_measure(a, b)[2]


def calinski_harabasz_score(X, labels):
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    n, k = len(X), len(np.unique(labels))
    if k < 2 or k >= n:
        raise ValueError("Need 2..n-1 clusters")
    overall = X.mean(0)
    B = sum(((X[labels == c].mean(0) - overall) ** 2).sum() * (labels == c).sum() for c in np.unique(labels))
    W = sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    return float((B / (k - 1)) / (W / (n - k))) if W > 0 else float("inf")


def davies_bouldin_score(X, labels):
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    classes = np.unique(labels)
    k = len(classes)
    centroids = np.array([X[labels == c].mean(0) for c in classes])
    S = np.array([np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1)) for i, c in enumerate(classes)])
    M = pairwise_distances(centroids)
    R = np.zeros(k)
    for i in range(k):
        with np.errstate(divide="ignore"):
            rij = (S[i] + S) / M[i]
        rij[i] = -np.inf
        R[i] = np.max(rij)
    return float(R.mean())


def mutual_info_score(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b, return_inverse=True)[1]
    cont = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(cont, (ca, cb), 1)
    p = cont / cont.sum()
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore"):
        mi = np.nansum(p * np.log(p / (px @ py)))
    return float(max(mi, 0))


def fowlkes_mallows_score(a, b):
    a = np.asarray(a); b = np.asarray(b)
    from scipy.special import comb as _comb
    ca = np.unique(a, return_inverse=True)[1]; cb = np.unique(b, return_inverse=True)[1]
    cont = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(cont, (ca, cb), 1)
    tp = _comb(cont, 2).sum()
    fp = _comb(cont.sum(0), 2).sum() - tp
    fn = _comb(cont.sum(1), 2).sum() - tp
    return float(tp / np.sqrt((tp + fp) * (tp + fn))) if (tp + fp) * (tp + fn) > 0 else 0.0


# ---------------- ranking / misc ----------------

def dcg_score(y_true, y_score, k=None):
    y_true = np.asarray(y_true, dtype=np.float64); y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[None, :]; y_score = y_score[None, :]
    order = np.argsort(-y_score, axis=1)
    gains = np.take_along_axis(y_true, order, axis=1)
    if k is not None:
        gains = gains[:, :k]
    discounts = 1.0 / np.log2(np.arange(gains.shape[1]) + 2)
    return float((gains * discounts).sum(1).mean())


def ndcg_score(y_true, y_score, k=None):
    ideal = dcg_score(y_true, y_true, k=k)
    if ideal == 0:
        return 0.0
    return float(dcg_score(y_true, y_score, k=k) / ideal)


def get_scorer(name):
    table = {
        "accuracy": lambda e, X, y: accuracy_score(y, e.predict(X)),
        "f1": lambda e, X, y: f1_score(y, e.predict(X), average="weighted" if len(np.unique(y)) > 2 else "binary"),
        "precision": lambda e, X, y: precision_score(y, e.predict(X), average="weighted" if len(np.unique(y)) > 2 else "binary"),
        "recall": lambda e, X, y: recall_score(y, e.predict(X), average="weighted" if len(np.unique(y)) > 2 else "binary"),
        "roc_auc": lambda e, X, y: roc_auc_score(y, e.predict_proba(X)[:, 1] if hasattr(e, "predict_proba") else e.decision_function(X)),
        "neg_mean_squared_error": lambda e, X, y: -mean_squared_error(y, e.predict(X)),
        "neg_mean_absolute_error": lambda e, X, y: -mean_absolute_error(y, e.predict(X)),
        "r2": lambda e, X, y: r2_score(y, e.predict(X)),
    }
    if name not in table:
        raise ValueError(f"Unknown scorer {name!r}")
    return table[name]

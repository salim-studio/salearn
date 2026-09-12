"""salearn.eda — one-line exploratory data analysis for analysts.

Works with pandas if installed, else with numpy/stdlib.

    from salearn.eda import profile, missing_report, correlation_matrix
    print(profile(df))              # dict summary
    print(profile(df, html_path="report.html"))  # + HTML report
"""
from __future__ import annotations

import math
from typing import Sequence


def _as_frame(data):
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data, True
    except ImportError:
        pass
    return data, False


def missing_report(data) -> dict:
    """{col: {n_missing, pct}} sorted worst-first."""
    df, ok = _as_frame(data)
    if ok:
        n = len(df)
        out = {c: {"n_missing": int(df[c].isna().sum()),
                   "pct": round(100 * df[c].isna().mean(), 2)} for c in df.columns}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["pct"]))
    import numpy as np
    rows, cols = data if isinstance(data, tuple) else ([], [])
    out = {}
    for j, c in enumerate(cols):
        miss = sum(1 for r in rows if r[j] in (None, "", "NaN", "nan") or (isinstance(r[j], float) and math.isnan(r[j])))
        out[c] = {"n_missing": miss, "pct": round(100 * miss / max(len(rows), 1), 2)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["pct"]))


def describe(data, percentiles=(0.25, 0.5, 0.75)) -> dict:
    """Numeric summary per column: count/mean/std/min/pXX/max."""
    import numpy as np
    df, ok = _as_frame(data)
    if ok:
        import pandas as pd
        num = df.select_dtypes(include=[np.number])
        d = {}
        for c in num.columns:
            s = num[c].dropna()
            d[c] = {"count": int(s.count()), "mean": float(s.mean()) if len(s) else 0.0,
                    "std": float(s.std()) if len(s) else 0.0,
                    "min": float(s.min()) if len(s) else 0.0,
                    "max": float(s.max()) if len(s) else 0.0,
                    "median": float(s.median()) if len(s) else 0.0}
        return d
    X, cols = (np.asarray(data, dtype=np.float64), [f"x{i}" for i in range(np.asarray(data).shape[1])]) if not isinstance(data, tuple) else (None, None)
    if X is None:
        rows, cols = data
        def _n(v):
            try:
                return float(v)
            except Exception:
                return float("nan")
        X = np.array([[_n(v) for v in r] for r in rows], dtype=np.float64) if rows else np.empty((0, len(cols)))
    return {cols[j]: {"count": int(np.isfinite(X[:, j]).sum()),
                      "mean": float(np.nanmean(X[:, j])) if len(X) else 0.0,
                      "std": float(np.nanstd(X[:, j])) if len(X) else 0.0,
                      "min": float(np.nanmin(X[:, j])) if len(X) else 0.0,
                      "max": float(np.nanmax(X[:, j])) if len(X) else 0.0,
                      "median": float(np.nanmedian(X[:, j])) if len(X) else 0.0}
            for j in range(X.shape[1])}


def correlation_matrix(data, method="pearson"):
    import numpy as np
    df, ok = _as_frame(data)
    if ok:
        return df.select_dtypes(include=[np.number]).corr(method=method)
    X = np.asarray(data[0], dtype=np.float64) if isinstance(data, tuple) else np.asarray(data, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X - np.nanmean(X, axis=0)
    cov = np.nan_to_num(X.T @ X) / max(len(X) - 1, 1)
    d = np.sqrt(np.diag(cov))
    return cov / np.maximum.outer(d, d).clip(min=1e-12)


def outlier_report(data, method="iqr", thresh=1.5) -> dict:
    """Count outliers per numeric column (IQR or z-score)."""
    import numpy as np
    desc = describe(data)
    df, ok = _as_frame(data)
    out = {}
    if ok:
        for c in desc:
            s = df[c].dropna().to_numpy(dtype=np.float64)
            if method == "zscore":
                z = np.abs((s - s.mean()) / (s.std() or 1))
                out[c] = int((z > thresh).sum())
            else:
                q1, q3 = np.percentile(s, [25, 75]) if len(s) else (0, 0)
                iqr = q3 - q1
                out[c] = int(((s < q1 - thresh * iqr) | (s > q3 + thresh * iqr)).sum())
        return out
    X = np.asarray(data[0], dtype=np.float64) if isinstance(data, tuple) else np.asarray(data, dtype=np.float64)
    cols = list(describe(data).keys())
    for j, c in enumerate(cols):
        s = X[:, j]
        s = s[np.isfinite(s)]
        if method == "zscore":
            z = np.abs((s - s.mean()) / (s.std() or 1)) if len(s) else np.array([])
            out[c] = int((z > thresh).sum())
        else:
            q1, q3 = np.percentile(s, [25, 75]) if len(s) else (0, 0)
            iqr = q3 - q1
            out[c] = int(((s < q1 - thresh * iqr) | (s > q3 + thresh * iqr)).sum())
    return out


def value_counts(data, col: str, top: int = 10) -> dict:
    df, ok = _as_frame(data)
    if ok:
        return df[col].value_counts().head(top).to_dict()
    rows, cols = data
    j = cols.index(col)
    from collections import Counter
    return dict(Counter(r[j] for r in rows).most_common(top))


def profile(data, html_path: str | None = None) -> dict:
    """Full dataset profile: shape, dtypes, missing, numeric summary, outliers."""
    df, ok = _as_frame(data)
    rep = {"n_rows": len(df) if ok else len(data[0]),
           "n_cols": len(df.columns) if ok else len(data[1]),
           "columns": list(df.columns) if ok else list(data[1]),
           "missing": missing_report(data),
           "numeric_summary": describe(data),
           "outliers_iqr": outlier_report(data)}
    if ok:
        rep["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
        rep["duplicates"] = int(df.duplicated().sum())
        rep["memory_mb"] = round(float(df.memory_usage(deep=True).sum()) / 1e6, 3)
    if html_path:
        _write_html(rep, html_path)
    return rep


def _write_html(rep: dict, path: str):
    rows = "".join(f"<tr><td>{c}</td><td>{v['n_missing']}</td><td>{v['pct']}%</td></tr>"
                   for c, v in rep["missing"].items())
    nums = "".join(f"<tr><td>{c}</td><td>{v['count']}</td><td>{v['mean']:.3f}</td>"
                   f"<td>{v['std']:.3f}</td><td>{v['min']:.3f}</td><td>{v['max']:.3f}</td></tr>"
                   for c, v in rep["numeric_summary"].items())
    html = f"""<html><head><meta charset='utf-8'><title>salearn EDA report</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}</style>
</head><body><h1>salearn — EDA report</h1>
<p>rows={rep['n_rows']} cols={rep['n_cols']}</p>
<h2>Missing</h2><table><tr><th>col</th><th>n</th><th>%</th></tr>{rows}</table>
<h2>Numeric</h2><table><tr><th>col</th><th>n</th><th>mean</th><th>std</th><th>min</th><th>max</th></tr>{nums}</table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


__all__ = ["missing_report", "describe", "correlation_matrix", "outlier_report", "value_counts", "profile"]

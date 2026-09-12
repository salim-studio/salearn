"""salearn.io — universal data loading for analysts (pandas-optional).

- load_csv / load_parquet / load_json / load_excel / load_url
- save_dataset / detect_format / chunked_csv
- to_X_y : DataFrame-or-numpy -> (X float, y, feature_names)
- dumb-safe: works with stdlib only (csv/json) when pandas/pyarrow missing.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Iterator, Sequence


def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"csv": "csv", "tsv": "csv", "txt": "csv", "parquet": "parquet",
            "pq": "parquet", "json": "json", "jsonl": "json",
            "xlsx": "excel", "xls": "excel"}.get(ext, "csv")


def _has(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def load_csv(path_or_buf, target: str | None = None, sep: str | None = None, **kw):
    """Return DataFrame if pandas exists else (rows, columns)."""
    if _has("pandas"):
        import pandas as pd
        return pd.read_csv(path_or_buf, sep=sep or ",", **kw)
    # stdlib fallback
    if hasattr(path_or_buf, "read"):
        text = path_or_buf.read()
    else:
        with open(path_or_buf, newline="", encoding="utf-8-sig") as f:
            text = f.read()
    dialect = csv.Sniffer().sniff(text[:4096], delimiters=[sep] if sep else [",", ";", "\t", "|"]) if text.strip() else None
    rdr = csv.DictReader(io.StringIO(text), dialect=dialect) if dialect else csv.DictReader(io.StringIO(text), delimiter=sep or ",")
    rows = list(rdr)
    return rows, (rdr.fieldnames or [])


def chunked_csv(path: str, chunksize: int = 10_000, **kw) -> Iterator:
    """Yield DataFrame chunks (pandas) or list-of-dict chunks (stdlib)."""
    if _has("pandas"):
        import pandas as pd
        yield from pd.read_csv(path, chunksize=chunksize, **kw)
        return
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        chunk: list = []
        for row in rdr:
            chunk.append(row)
            if len(chunk) >= chunksize:
                yield chunk, (rdr.fieldnames or [])
                chunk = []
        if chunk:
            yield chunk, (rdr.fieldnames or [])


def load_parquet(path: str, **kw):
    if _has("pandas"):
        import pandas as pd
        try:
            return pd.read_parquet(path, **kw)
        except Exception as e:
            raise ImportError(f"Need pyarrow/fastparquet for parquet: {e}") from e
    # duckdb fallback -> numpy
    if _has("duckdb"):
        import duckdb
        rel = duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')").fetchall()
        cols = [d[0] for d in duckdb.connect().description] if False else []
        return rel, cols
    raise ImportError("pip install pandas pyarrow  (or duckdb) to read parquet")


def load_json(path_or_buf, **kw):
    if _has("pandas"):
        import pandas as pd
        try:
            return pd.read_json(path_or_buf, **kw)
        except Exception:
            pass
    if hasattr(path_or_buf, "read"):
        return json.load(path_or_buf)
    with open(path_or_buf, encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def load_excel(path: str, sheet: str | int = 0, **kw):
    if not _has("pandas"):
        raise ImportError("pip install pandas openpyxl to read excel")
    import pandas as pd
    return pd.read_excel(path, sheet_name=sheet, **kw)


def load_url(url: str, **kw):
    """Download http(s) CSV/JSON/parquet to memory."""
    import urllib.request
    with urllib.request.urlopen(url) as r:
        raw = r.read()
    name = url.split("?")[0].lower()
    if name.endswith(".json") or name.endswith(".jsonl"):
        return json.loads(raw.decode("utf-8"))
    if name.endswith(".parquet") or name.endswith(".pq"):
        if not _has("pandas"):
            raise ImportError("pip install pandas pyarrow for remote parquet")
        import pandas as pd, io as _io
        return pd.read_parquet(_io.BytesIO(raw))
    # default csv
    if _has("pandas"):
        import pandas as pd, io as _io
        return pd.read_csv(_io.BytesIO(raw), **kw)
    return load_csv(io.StringIO(raw.decode("utf-8")))


def save_dataset(data, path: str, **kw):
    fmt = detect_format(path)
    if _has("pandas"):
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            if fmt == "parquet":
                return data.to_parquet(path, index=False, **kw)
            if fmt == "json":
                return data.to_json(path, orient="records", indent=2)
            if fmt == "excel":
                return data.to_excel(path, index=False)
            return data.to_csv(path, index=False, **kw)
    # stdlib csv/json
    if fmt == "json":
        rows = data if isinstance(data, list) else data[0]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return path
    rows, cols = data if isinstance(data, tuple) else (data, None)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        cols = cols or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return path
    raise ValueError("Unsupported data type for save_dataset without pandas")


def to_X_y(data, target: str | None = None, drop: Sequence[str] = ()):
    """DataFrame / (rows,cols) / numpy -> (X: float64, y or None, feature_names)."""
    import numpy as np
    if target is None:
        X = np.asarray(data, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, None, [f"x{i}" for i in range(X.shape[1])]
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            feats = [c for c in data.columns if c != target and c not in drop]
            X = data[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            return X, data[target].to_numpy(), feats
    except ImportError:
        pass
    if isinstance(data, tuple):
        rows, cols = data
        ti = cols.index(target)
        fi = [i for i, c in enumerate(cols) if c != target and c not in drop]
        feats = [cols[i] for i in fi]
        def _n(v):
            try:
                return float(v)
            except Exception:
                return float(hash(str(v)) % 997) / 997.0
        X = np.array([[_n(r[i] if isinstance(r, (list, tuple)) else r[cols[i]]) for i in fi] for r in rows], dtype=np.float64) if rows else np.empty((0, len(fi)))
        y = np.array([r[ti] if isinstance(r, (list, tuple)) else r[cols[ti]] for r in rows])
        return X, y, feats
    raise ValueError("Cannot convert to X,y — pass DataFrame or (rows, columns)")


__all__ = ["detect_format", "load_csv", "chunked_csv", "load_parquet", "load_json",
           "load_excel", "load_url", "save_dataset", "to_X_y"]

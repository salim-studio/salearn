"""salearn.db — unified database toolkit for data scientists & developers.

Covers SQLite (stdlib, zero-dep) + optional SQLAlchemy/DuckDB/Postgres/MySQL.

Quick start::

    from salearn.db import Database
    db = Database.sqlite_memory()
    db.execute("CREATE TABLE iris (sepal REAL, petal REAL, label INT)")
    db.executemany("INSERT INTO iris VALUES (?,?,?)", [(5.1, 1.4, 0)])
    X, y = db.load_X_y("SELECT * FROM iris", target="label")
    # train directly from SQL:
    from salearn.ensemble import RandomForestClassifier
    model = db.train_sql("SELECT * FROM iris", target="label",
                         estimator=RandomForestClassifier())

Features beyond sklearn:
- URI factory: sqlite, duckdb, postgresql, mysql, sqlalchemy passthrough
- read_sql / to_sql with or without pandas
- schema helpers: list_tables, table_info, table_counts
- QueryBuilder: fluent SELECT builder (no ORM needed)
- ML bridge: load_X_y, train_test_from_sql, train_sql
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


def _rows_to_numpy(rows, columns=None):
    import numpy as np
    if not rows:
        return np.empty((0, 0)), list(columns or [])
    arr = np.array(rows, dtype=object)
    return arr, list(columns or [])


class QueryBuilder:
    """Tiny fluent SELECT builder: QueryBuilder('t').select('a','b').where('a>5').limit(10).build()."""

    def __init__(self, table: str):
        self.table = table
        self._select = ["*"]
        self._where: list[str] = []
        self._order = ""
        self._limit = ""
        self._offset = ""

    def select(self, *cols: str):
        if cols:
            self._select = list(cols)
        return self

    def where(self, cond: str):
        self._where.append(f"({cond})")
        return self

    def order_by(self, expr: str):
        self._order = f"ORDER BY {expr}"
        return self

    def limit(self, n: int):
        self._limit = f"LIMIT {int(n)}"
        return self

    def offset(self, n: int):
        self._offset = f"OFFSET {int(n)}"
        return self

    def build(self) -> str:
        q = f"SELECT {', '.join(self._select)} FROM {self.table}"
        if self._where:
            q += " WHERE " + " AND ".join(self._where)
        if self._order:
            q += " " + self._order
        if self._limit:
            q += " " + self._limit
        if self._offset:
            q += " " + self._offset
        return q + ";"


@dataclass
class TableProfile:
    name: str
    n_rows: int
    columns: list
    dtypes: list


class Database:
    """DB-agnostic wrapper. Defaults to sqlite3; upgrades to SQLAlchemy/DuckDB if installed."""

    def __init__(self, conn=None, uri: str | None = None):
        self.uri = uri
        self._sa_engine = None
        self._duck = None
        if conn is not None:
            self.conn = conn
        elif uri is None or uri.startswith("sqlite"):
            path = ":memory:"
            if uri and uri not in ("sqlite:///:memory:", "sqlite://", "sqlite"):
                # sqlite:///file.db  or  sqlite:////abs/path
                p = uri.split(":///", 1)[-1] if ":///" in uri else ":memory:"
                path = p or ":memory:"
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        else:
            # Try SQLAlchemy for postgres/mysql/duckdb URIs
            try:
                import sqlalchemy as sa  # type: ignore
                self._sa_engine = sa.create_engine(uri)
                self.conn = self._sa_engine.connect()
            except Exception as e:
                raise ImportError(
                    f"Cannot open {uri!r}. Install sqlalchemy + driver "
                    f"(e.g. pip install 'salearn[db]') — {e}"
                ) from e

    # ---- factories ----
    @classmethod
    def sqlite_memory(cls):
        return cls(uri="sqlite:///:memory:")

    @classmethod
    def sqlite_file(cls, path: str):
        return cls(uri=f"sqlite:///{path}")

    @classmethod
    def from_uri(cls, uri: str):
        if uri.startswith("duckdb"):
            return DuckDBDatabase(uri)
        return cls(uri=uri)

    # ---- low level ----
    def execute(self, sql: str, params: Sequence | None = None):
        cur = self.conn.execute(sql, params or []) if hasattr(self.conn, "execute") else None
        if cur is None:  # SQLAlchemy connection
            from sqlalchemy import text as _t
            cur = self.conn.execute(_t(sql), params or {})
        try:
            self.conn.commit()
        except Exception:
            pass
        return cur

    def executemany(self, sql: str, rows: Iterable[Sequence]):
        rows = list(rows)
        if self._sa_engine is not None:
            from sqlalchemy import text as _t
            # convert ? placeholders to named for sqlalchemy
            self.conn.execute(_t(sql.replace("?", ":p")), [{"p": v} for r in rows for v in [r]])
            self.conn.commit()
            return
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def query(self, sql: str, params: Sequence | None = None):
        """Return (rows:list[tuple], columns:list[str]). Works with/without pandas."""
        if self._sa_engine is not None:
            from sqlalchemy import text as _t
            r = self.conn.execute(_t(sql), params or {})
            cols = list(r.keys())
            return [tuple(x) for x in r.fetchall()], cols
        cur = self.conn.execute(sql, params or [])
        cols = [d[0] for d in cur.description] if cur.description else []
        return [tuple(x) for x in cur.fetchall()], cols

    # ---- pandas bridge (optional) ----
    def read_sql(self, sql: str, params=None):
        """Return pandas.DataFrame if pandas installed, else (rows, columns)."""
        rows, cols = self.query(sql, params)
        try:
            import pandas as pd
            return pd.DataFrame(rows, columns=cols)
        except ImportError:
            return rows, cols

    def to_sql(self, table: str, rows=None, columns=None, df=None, if_exists="append"):
        """Write rows or DataFrame to table. Creates table if needed."""
        import sqlite3 as _s
        if df is not None:
            try:
                import pandas as pd  # noqa
                assert isinstance(df, pd.DataFrame)
                if self._sa_engine is not None:
                    df.to_sql(table, self._sa_engine, if_exists=if_exists, index=False)
                else:
                    df.to_sql(table, self.conn, if_exists=if_exists, index=False)
                return self
            except ImportError:
                rows, columns = df.values.tolist(), list(df.columns)
        assert rows is not None and columns is not None
        if if_exists == "replace":
            try:
                self.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass
        placeholders = ",".join(["?"] * len(columns))
        # create table if missing (all TEXT affinity -> flexible)
        coldefs = ", ".join(f'"{c}"' for c in columns)
        self.execute(f"CREATE TABLE IF NOT EXISTS {table} ({coldefs})")
        reps = ",".join(["?"] * len(columns))
        if self._sa_engine is None:
            self.conn.executemany(f"INSERT INTO {table} VALUES ({reps})", rows)
            self.conn.commit()
        else:
            self.executemany(f"INSERT INTO {table} VALUES ({reps})", rows)
        return self

    # ---- schema ----
    def list_tables(self) -> list[str]:
        try:
            rows, _ = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            if rows:
                return [r[0] for r in rows]
        except Exception:
            pass
        try:  # sqlalchemy inspector
            import sqlalchemy as sa
            return sa.inspect(self._sa_engine).get_table_names()
        except Exception:
            return []

    def table_info(self, table: str):
        rows, _ = self.query(f"PRAGMA table_info({table})")
        return rows

    def count(self, table: str, where: str = "") -> int:
        q = f"SELECT COUNT(*) FROM {table}" + (f" WHERE {where}" if where else "")
        rows, _ = self.query(q)
        return int(rows[0][0]) if rows else 0

    def profile(self, table: str) -> TableProfile:
        info = self.table_info(table)
        cols = [r[1] for r in info] if info and len(info[0]) > 2 else []
        return TableProfile(name=table, n_rows=self.count(table), columns=cols,
                            dtypes=[r[2] for r in info] if info else [])

    # ---- ML bridge ----
    def load_X_y(self, sql: str, target: str, drop: Sequence[str] = ()):
        """Run SELECT and split into (X float matrix, y array, feature_names)."""
        import numpy as np
        data = self.read_sql(sql)
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                feats = [c for c in data.columns if c != target and c not in drop]
                X = data[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
                return X, data[target].to_numpy(), feats
        except ImportError:
            pass
        rows, cols = data if isinstance(data, tuple) else (data, [])
        ti = cols.index(target)
        feat_idx = [i for i, c in enumerate(cols) if c != target and c not in drop]
        feats = [cols[i] for i in feat_idx]
        M = np.array(rows, dtype=object)
        def _num(v):
            try:
                return float(v)
            except Exception:
                return 0.0
        X = np.array([[_num(v) for v in r] for r in M[:, feat_idx].tolist()], dtype=np.float64) if len(rows) else np.empty((0, len(feat_idx)))
        y = np.array([r[ti] for r in rows])
        try:
            y = y.astype(np.float64) if all(isinstance(v, (int, float)) for v in y) else y
        except Exception:
            pass
        return X, y, feats

    def train_test_from_sql(self, sql: str, target: str, test_size=0.2, random_state=0):
        from .model_selection import train_test_split
        X, y, feats = self.load_X_y(sql, target)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return (Xtr, Xte, ytr, yte, feats)

    def train_sql(self, sql: str, target: str, estimator=None):
        from .ensemble import RandomForestClassifier
        est = estimator or RandomForestClassifier(n_estimators=100)
        X, y, _ = self.load_X_y(sql, target)
        return est.fit(X, y)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class DuckDBDatabase(Database):
    """DuckDB backend (OLAP, parquet-native). Requires `pip install duckdb`."""

    def __init__(self, uri: str = "duckdb:///:memory:"):
        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise ImportError("pip install duckdb  (or pip install 'salearn[db]')") from e
        import duckdb
        path = uri.split("duckdb:///", 1)[-1] if "duckdb:///" in uri else ":memory:"
        self.uri = uri
        self._sa_engine = None
        self.conn = duckdb.connect(path or ":memory:")
        self._duck = self.conn

    def query(self, sql: str, params: Sequence | None = None):
        rel = self.conn.execute(sql, params or []).fetchall()
        cols = [d[0] for d in self.conn.description] if self.conn.description else []
        return [tuple(r) for r in rel], cols

    def execute(self, sql: str, params: Sequence | None = None):
        self.conn.execute(sql, params or [])
        return self.conn

    def read_parquet(self, path: str, sql_filter: str = ""):
        q = f"SELECT * FROM read_parquet('{path}')"
        if sql_filter:
            q = f"SELECT * FROM ({q}) WHERE {sql_filter}"
        return self.read_sql(q)


__all__ = ["Database", "DuckDBDatabase", "QueryBuilder", "TableProfile"]

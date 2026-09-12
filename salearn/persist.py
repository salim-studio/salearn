"""salearn.persist — save/load + versioned model registry.

    from salearn.persist import save_model, load_model, ModelRegistry
    save_model(pipe, "model.ssl")
    pipe2 = load_model("model.ssl")
    reg = ModelRegistry("./store"); reg.register(pipe, name="churn", metrics={"acc": .9})
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pickle


def save_model(estimator, path: str, metadata: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"estimator": estimator, "metadata": metadata or {},
                     "salearn_version": _version(), "saved_at": _dt.datetime.now(_dt.UTC).isoformat()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if metadata:
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["estimator"] if isinstance(obj, dict) and "estimator" in obj else obj


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0"


class ModelRegistry:
    """Folder registry: store/<name>/v<N>.ssl + meta.json ; latest() loads newest."""

    def __init__(self, root="./model_store"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def register(self, estimator, name: str, metrics: dict | None = None, metadata: dict | None = None):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        existing = [f for f in os.listdir(d) if f.startswith("v") and f.endswith(".ssl")]
        v = len(existing) + 1
        path = os.path.join(d, f"v{v}.ssl")
        meta = {"name": name, "version": v, "metrics": metrics or {},
                "extra": metadata or {}, "saved_at": _dt.datetime.now(_dt.UTC).isoformat(),
                "salearn_version": _version()}
        save_model(estimator, path, metadata=meta)
        with open(os.path.join(d, f"v{v}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(d, "latest.txt"), "w") as f:
            f.write(f"v{v}.ssl")
        return path, meta

    def latest(self, name: str):
        d = os.path.join(self.root, name)
        with open(os.path.join(d, "latest.txt")) as f:
            fn = f.read().strip()
        return load_model(os.path.join(d, fn))

    def list(self, name: str | None = None):
        if name:
            d = os.path.join(self.root, name)
            return sorted(f for f in os.listdir(d) if f.endswith(".ssl")) if os.path.isdir(d) else []
        return sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []


__all__ = ["save_model", "load_model", "ModelRegistry"]

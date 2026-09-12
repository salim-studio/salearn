"""salearn._numba — optional Numba acceleration dispatcher.

If numba is installed we use njit(parallel/fastmath) for hot loops.
Otherwise pure-NumPy fallbacks are used so salearn works everywhere.
"""
from __future__ import annotations

try:
    from numba import njit, prange  # type: ignore
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        if a and callable(a[0]) and len(a) == 1 and not k:
            return a[0]
        return deco

    def prange(*a):
        return range(*a)


def jit(nopython=True, cache=True, fastmath=True, parallel=False, **kw):
    """Decorator: njit when numba exists, identity otherwise."""
    if HAS_NUMBA:
        from numba import njit as _njit
        def deco(f):
            return _njit(nopython=nopython, cache=cache, fastmath=fastmath, parallel=parallel, **kw)(f)
        return deco
    def deco(f):
        return f
    return deco

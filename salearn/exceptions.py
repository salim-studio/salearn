"""salearn.exceptions."""
from __future__ import annotations


class NotFittedError(ValueError, AttributeError):
    pass


class ConvergenceWarning(UserWarning):
    pass

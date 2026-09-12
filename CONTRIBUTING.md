# Contributing to salearn

Thanks for helping! Quick rules:

1. **Setup**
   ```bash
   pip install -e ".[all]"
   python -m pytest -q
   ```
2. **Style** — keep the sklearn API (`fit/predict/score/transform`), stay dependency-light
   (NumPy/SciPy core; pandas/matplotlib/SQLAlchemy only as optional extras with graceful fallbacks).
3. **Tests** — every new module needs tests in `tests/` (see `test_stack.py` for the pattern).
4. **Docs** — update `README.md` module map + add an example to `examples_fullstack.py` when relevant.
5. **Commits** — small, descriptive messages. PRs welcome; issues first for big features.

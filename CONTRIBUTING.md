# Contributing to CausalSieve

Contributions, bug reports, reproducible benchmarks, and ecosystem adapters are
welcome. CausalSieve enforces strict engineering invariants across the C++20
core, C99 ABI, and Python surface.

## Engineering Invariants

1. **Zero Runtime Python Dependencies:**
   `pyproject.toml` MUST keep `dependencies = []`. Core bindings, CLI, MCP
   server, and ecosystem adapters (`analyze_dataframe`, `analyze_prometheus`,
   `rcaeval_rca`) must run on bare stdlib Python 3.10+ without requiring
   `numpy`, `pandas`, `polars`, or `scipy` (use duck typing / `__array_interface__`
   for optional zero-copy acceleration).
2. **Strict Causality (Zero Look-Ahead):**
   Any feature, imputation, normalization, or scoring change at tick `t` must be
   bit-identical regardless of observations at ticks `> t`. Verified by
   `tests/test_causality_and_numeric.py` and `tests/breaker_audit.py`.
3. **Mathematical Exactness & Memory Safety:**
   Tier-1 Block-Schur Granger and precision updates must match exact batch FP64
   SVD refits within `< 1e-10` and pass ASan + UBSan with zero findings.
4. **C99 ABI Stability:**
   All exported symbols in `csrc/causalsieve.h` must remain C99-compatible,
   thread-safe across independent engine instances, and mirrored in `setup.py`
   MSVC exports and `python/causalsieve/_native.py`.
5. **Honest Benchmarks & Clean-Room Hygiene:**
   Report failures, crossovers, and limits alongside wins. Never commit private
   paths, credentials, or external datasets under `benchmarks/data/`.

## Local Development Setup

```sh
# 1. Build C++20 shared libraries, native CLI, and ASan/UBSan test binaries
make all

# 2. Install Python package in editable mode (or build a wheel)
pip install -e ".[dev]"

# 3. Verify active SIMD ISA and library path
causalsieve info
```

## Verification Checklist (Run Before Opening a PR)

```sh
# C++20 ASan + UBSan + AVX2/scalar parity suite
make test

# Python unit, exact-oracle, causality, RCA, digest, and MCP tests
pytest tests/ -v

# Adversarial breaker audit (numerical torture, MCP red-team, clean-room scan)
pytest tests/breaker_audit.py -v

# Linting and strict static typing
ruff check python/causalsieve setup.py
mypy --strict --ignore-missing-imports python/causalsieve

# Standalone wheel build & bare-venv smoke test
CAUSALSIEVE_PORTABLE=1 python -m build --wheel --sdist
```

## Pull Request Process

1. Open an issue or PR describing the motivation, design tradeoff, and test
   coverage.
2. Keep changes focused and include deterministic unit tests covering edge cases
   (such as `NaN`/`Inf`, empty inputs, flat channels, or boundary dimensions).
3. Update `SKILL.md` and `skills/causalsieve/SKILL.md` together if public CLI,
   MCP, or Python adapter signatures change.
4. By submitting a contribution, you agree that your work is licensed under the
   MIT License (see [LICENSE](LICENSE)).

"""Optional Numba-accelerated inner loops.

This module provides JIT-compiled versions of the hot inner loops
in the Bellman operator. If Numba is not installed, the functions
fall back to pure-Python/NumPy implementations.
"""

from __future__ import annotations

try:
    import numba  # type: ignore[import-not-found]  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Numba kernels will be added in a future version.
# The pure-Python bellman.py implementation is the current baseline.

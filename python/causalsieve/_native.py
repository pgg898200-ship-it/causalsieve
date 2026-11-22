"""Zero-dependency ctypes bindings and high-level Python API for libcausalsieve.

Binds strictly to the C99 ABI defined in ``csrc/causalsieve.h``.
Supports zero-copy ``float64`` C-contiguous NumPy arrays (detected via
``__array_interface__`` without importing NumPy), zero-copy ``array('d')``
buffers, and arbitrary Python sequences using only the Python standard library.
"""

from __future__ import annotations

import array
import csv
import ctypes
import ctypes.util
import math
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causalsieve.digest import CausalDigest, StreamStore

CAUSALSIEVE_ABI_VERSION = 1

_NATIVE_F8_TYPESTR = "<f8" if sys.byteorder == "little" else ">f8"
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_INT32_P = ctypes.POINTER(ctypes.c_int32)
_INT64_P = ctypes.POINTER(ctypes.c_int64)
_UINT32_P = ctypes.POINTER(ctypes.c_uint32)
_UINT64_P = ctypes.POINTER(ctypes.c_uint64)
_SIZE_T_P = ctypes.POINTER(ctypes.c_size_t)

_TIMESTAMP_HEADER_NAMES = frozenset(
    {
        "",
        "timestamp",
        "time",
        "ts",
        "datetime",
        "date",
        "utc",
        "epoch",
        "unix_time",
        "time_s",
        "time_ms",
        "time_ns",
        "tick",
        "step",
        "index",
        "idx",
        "row",
    }
)


class CausalSieveError(RuntimeError):
    """Base error raised by CausalSieve operations."""


class CausalSieveLibraryError(OSError):
    """Raised when ``libcausalsieve`` cannot be located or loaded."""


# =========================================================================
# Exact C99 ABI struct definitions (csrc/causalsieve.h)
# =========================================================================


class CSConfig(ctypes.Structure):
    """Exact ctypes mirror of ``cs_config_t``."""

    _fields_ = [
        ("dim", ctypes.c_uint32),
        ("lag", ctypes.c_uint32),
        ("forgetting", ctypes.c_double),
        ("ridge", ctypes.c_double),
        ("precision_ridge", ctypes.c_double),
        ("shock_threshold", ctypes.c_double),
        ("e_alpha", ctypes.c_double),
        ("warmup_ticks", ctypes.c_uint32),
        ("refresh_interval", ctypes.c_uint32),
        ("nonlinear_hinge", ctypes.c_uint32),
        ("raw_ring_capacity", ctypes.c_uint32),
        ("normalize_scale", ctypes.c_uint32),
    ]


class CSTickResult(ctypes.Structure):
    """Exact ctypes mirror of ``cs_tick_result_t``."""

    _fields_ = [
        ("tick", ctypes.c_uint64),
        ("mahalanobis_sq", ctypes.c_double),
        ("e_value", ctypes.c_double),
        ("shock_weight", ctypes.c_double),
        ("max_exo_z", ctypes.c_double),
        ("top_root_channel", ctypes.c_int32),
        ("alarm", ctypes.c_uint32),
        ("refresh_count", ctypes.c_uint32),
    ]


# =========================================================================
# Native shared library discovery and lazy binding
# =========================================================================

_LIB_CACHE: ctypes.CDLL | None = None
_LIB_PATH_CACHE: str | None = None


def _platform_lib_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libcausalsieve.dylib", "libcausalsieve.so")
    if os.name == "nt":
        return ("causalsieve.dll", "libcausalsieve.dll")
    return ("libcausalsieve.so",)


def _bundled_library_paths() -> list[Path]:
    here_file = Path(__file__).resolve()
    return [here_file.with_name(name) for name in _platform_lib_names()]


def _fallback_library_paths() -> list[Path]:
    names = _platform_lib_names()
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    candidates: list[Path] = []
    for name in names:
        candidates.append(repo_root / "build" / name)
    cwd_build = Path.cwd() / "build"
    for name in names:
        cand = cwd_build / name
        if cand not in candidates:
            candidates.append(cand)
    return candidates


def _candidate_library_paths() -> list[Path]:
    return [*_bundled_library_paths(), *_fallback_library_paths()]


def _bind_functions(lib: ctypes.CDLL) -> None:
    """Attach exact C99 ABI prototypes from ``csrc/causalsieve.h``."""
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_uint32 = ctypes.c_uint32
    c_int32 = ctypes.c_int32
    c_uint64 = ctypes.c_uint64
    c_size_t = ctypes.c_size_t
    c_double = ctypes.c_double
    c_char_p = ctypes.c_char_p

    lib.cs_config_default.argtypes = [ctypes.POINTER(CSConfig), c_uint32, c_uint32]
    lib.cs_config_default.restype = None

    lib.cs_create.argtypes = [ctypes.POINTER(CSConfig)]
    lib.cs_create.restype = c_void_p

    lib.cs_destroy.argtypes = [c_void_p]
    lib.cs_destroy.restype = None

    lib.cs_reset.argtypes = [c_void_p]
    lib.cs_reset.restype = None

    lib.cs_simd_isa.argtypes = []
    lib.cs_simd_isa.restype = c_char_p

    lib.cs_last_error.argtypes = [c_void_p]
    lib.cs_last_error.restype = c_char_p

    lib.cs_tick_count.argtypes = [c_void_p]
    lib.cs_tick_count.restype = c_uint64

    lib.cs_dim.argtypes = [c_void_p]
    lib.cs_dim.restype = c_uint32

    lib.cs_lag.argtypes = [c_void_p]
    lib.cs_lag.restype = c_uint32

    lib.cs_feature_dim.argtypes = [c_void_p]
    lib.cs_feature_dim.restype = c_uint32

    lib.cs_push.argtypes = [
        c_void_p,
        _DOUBLE_P,
        ctypes.POINTER(CSTickResult),
        _DOUBLE_P,
        _DOUBLE_P,
    ]
    lib.cs_push.restype = c_int

    lib.cs_push_batch.argtypes = [
        c_void_p,
        _DOUBLE_P,
        c_size_t,
        ctypes.POINTER(CSTickResult),
        _DOUBLE_P,
        _DOUBLE_P,
    ]
    lib.cs_push_batch.restype = c_int

    lib.cs_granger_matrix.argtypes = [c_void_p, _DOUBLE_P, _DOUBLE_P]
    lib.cs_granger_matrix.restype = c_int

    lib.cs_precision_matrix.argtypes = [c_void_p, _DOUBLE_P, _DOUBLE_P]
    lib.cs_precision_matrix.restype = c_int

    lib.cs_contemporaneous_orientation.argtypes = [c_void_p, _DOUBLE_P]
    lib.cs_contemporaneous_orientation.restype = c_int

    lib.cs_root_attribution.argtypes = [
        c_void_p,
        c_uint32,
        _DOUBLE_P,
        _DOUBLE_P,
        _DOUBLE_P,
        _INT64_P,
        _INT32_P,
    ]
    lib.cs_root_attribution.restype = c_int

    lib.cs_counterfactual_do.argtypes = [
        c_void_p,
        c_int32,
        c_double,
        c_uint32,
        _DOUBLE_P,
    ]
    lib.cs_counterfactual_do.restype = c_int

    lib.cs_get_raw_slice.argtypes = [
        c_void_p,
        c_uint32,
        c_uint64,
        c_uint64,
        _DOUBLE_P,
        _UINT64_P,
        _UINT64_P,
    ]
    lib.cs_get_raw_slice.restype = c_int

    lib.cs_compress_channel_pla.argtypes = [
        c_void_p,
        c_uint32,
        c_uint64,
        c_uint64,
        c_double,
        _UINT64_P,
        _DOUBLE_P,
        c_size_t,
        _SIZE_T_P,
    ]
    lib.cs_compress_channel_pla.restype = c_int

    lib.cs_inspect_state.argtypes = [c_void_p, _DOUBLE_P, _DOUBLE_P]
    lib.cs_inspect_state.restype = c_int

    lib.cs_force_refresh.argtypes = [c_void_p]
    lib.cs_force_refresh.restype = c_int

    lib.cs_screened_create.argtypes = [
        c_uint32,
        c_uint32,
        c_uint32,
        c_double,
        c_double,
    ]
    lib.cs_screened_create.restype = c_void_p

    lib.cs_screened_destroy.argtypes = [c_void_p]
    lib.cs_screened_destroy.restype = None

    lib.cs_screened_push_batch.argtypes = [
        c_void_p,
        _DOUBLE_P,
        c_size_t,
        _DOUBLE_P,
        _INT32_P,
        _DOUBLE_P,
        c_uint32,
    ]
    lib.cs_screened_push_batch.restype = c_int

    lib.cs_screened_active_graph.argtypes = [
        c_void_p,
        _INT32_P,
        _DOUBLE_P,
        _DOUBLE_P,
        _UINT32_P,
    ]
    lib.cs_screened_active_graph.restype = c_int


def load_library(explicit_path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    """Locate, load, bind, and cache ``libcausalsieve``.

    Search order:
      1. ``explicit_path`` argument (if provided)
      2. Bundled shared library next to ``_native.py`` (zero-config wheel install)
      3. ``CAUSALSIEVE_LIB`` environment variable
      4. Repository ``build/`` directory and ``./build/`` in the current working directory
      5. System dynamic linker search path
    """
    global _LIB_CACHE, _LIB_PATH_CACHE
    if _LIB_CACHE is not None and explicit_path is None:
        return _LIB_CACHE

    searched: list[str] = []

    def _try_load(path_str: str) -> ctypes.CDLL:
        dll_dir_handle: Any = None
        if os.name == "nt":
            add_dll_dir = getattr(os, "add_dll_directory", None)
            if add_dll_dir is not None:
                parent = str(Path(path_str).resolve().parent)
                if os.path.isdir(parent):
                    dll_dir_handle = add_dll_dir(parent)
        try:
            lib = ctypes.CDLL(path_str)
            _bind_functions(lib)
            return lib
        finally:
            if dll_dir_handle is not None:
                dll_dir_handle.close()

    if explicit_path is not None:
        p = Path(explicit_path)
        searched.append(str(p))
        if not p.is_file():
            raise CausalSieveLibraryError(
                f"Specified libcausalsieve path does not exist: {p}. "
                "Install a prebuilt wheel (`pip install causalsieve`) or build with `make`."
            )
        lib = _try_load(str(p))
        _LIB_CACHE = lib
        _LIB_PATH_CACHE = str(p.resolve())
        return lib

    for cand in _bundled_library_paths():
        searched.append(str(cand))
        if cand.is_file():
            lib = _try_load(str(cand))
            _LIB_CACHE = lib
            _LIB_PATH_CACHE = str(cand.resolve())
            return lib

    env_path = os.environ.get("CAUSALSIEVE_LIB")
    if env_path:
        p = Path(env_path)
        searched.append(f"CAUSALSIEVE_LIB={p}")
        if not p.is_file():
            raise CausalSieveLibraryError(
                f"CAUSALSIEVE_LIB points to a missing file: {p}. "
                "Compile `build/libcausalsieve.so` via `make` or fix CAUSALSIEVE_LIB."
            )
        lib = _try_load(str(p))
        _LIB_CACHE = lib
        _LIB_PATH_CACHE = str(p.resolve())
        return lib

    for cand in _fallback_library_paths():
        searched.append(str(cand))
        if cand.is_file():
            lib = _try_load(str(cand))
            _LIB_CACHE = lib
            _LIB_PATH_CACHE = str(cand.resolve())
            return lib

    sys_found = ctypes.util.find_library("causalsieve")
    if sys_found:
        searched.append(f"system:{sys_found}")
        try:
            lib = _try_load(sys_found)
            _LIB_CACHE = lib
            _LIB_PATH_CACHE = sys_found
            return lib
        except OSError:
            pass

    locs = "\n  - ".join(searched)
    raise CausalSieveLibraryError(
        "Could not locate the native CausalSieve shared library (libcausalsieve).\n"
        f"Searched:\n  - {locs}\n"
        "Install a wheel (`pip install causalsieve`), build it with `make` in the repository "
        "root (produces `build/libcausalsieve.so`), or set CAUSALSIEVE_LIB=/path/to/libcausalsieve.so."
    )


def library_path() -> str | None:
    """Return the resolved path of the loaded native library, loading if available."""
    if _LIB_PATH_CACHE is not None:
        return _LIB_PATH_CACHE
    try:
        load_library()
    except CausalSieveLibraryError:
        return None
    return _LIB_PATH_CACHE


def simd_isa() -> str:
    """Return the active SIMD ISA string reported by the native library."""
    lib = load_library()
    raw = lib.cs_simd_isa()
    return raw.decode("utf-8", errors="replace") if raw else "unknown"


# =========================================================================
# Zero-copy float64 buffer packing helpers (NumPy-optional)
# =========================================================================


def _double_array_ptr(buf: array.array[float]) -> Any:
    addr, _ = buf.buffer_info()
    return ctypes.cast(addr, _DOUBLE_P)


def _as_double_ptr_1d(x: Any, expected_dim: int) -> tuple[Any, Any]:
    """Return ``(double*, keepalive_ref)`` for a 1D vector of length ``expected_dim``.

    Zero-copy for C-contiguous native ``float64`` NumPy arrays and ``array('d')``.
    """
    iface = getattr(x, "__array_interface__", None)
    if isinstance(iface, dict):
        shape = iface.get("shape")
        if not shape or len(shape) != 1:
            raise ValueError(f"Expected 1D array of length {expected_dim}, got shape {shape}")
        if int(shape[0]) != expected_dim:
            raise ValueError(f"Expected vector of length {expected_dim}, got {shape[0]}")
        typestr = iface.get("typestr")
        strides = iface.get("strides")
        is_c_contig = strides is None or bool(
            getattr(getattr(x, "flags", None), "c_contiguous", False)
        )
        if typestr == _NATIVE_F8_TYPESTR and is_c_contig:
            data = iface.get("data")
            if isinstance(data, tuple) and isinstance(data[0], int):
                return ctypes.cast(data[0], _DOUBLE_P), x
            if hasattr(x, "ctypes"):
                return ctypes.cast(int(x.ctypes.data), _DOUBLE_P), x
        np_mod = sys.modules.get("numpy")
        if np_mod is not None:
            arr = np_mod.ascontiguousarray(x, dtype=np_mod.float64)
            return ctypes.cast(int(arr.ctypes.data), _DOUBLE_P), arr
        buf = array.array("d", (float(v) for v in x))
        return _double_array_ptr(buf), buf

    if isinstance(x, array.array) and x.typecode == "d":
        if len(x) != expected_dim:
            raise ValueError(f"Expected array('d') of length {expected_dim}, got {len(x)}")
        return _double_array_ptr(x), x

    if isinstance(x, memoryview) and x.format == "d" and x.c_contiguous and x.ndim == 1:
        mv_shape = x.shape
        if mv_shape is None or mv_shape[0] != expected_dim:
            raise ValueError(f"Expected memoryview of length {expected_dim}, got {mv_shape}")
        if not x.readonly:
            c_arr = (ctypes.c_double * expected_dim).from_buffer(x)
            return ctypes.cast(c_arr, _DOUBLE_P), (x, c_arr)
        buf = array.array("d", x)
        return _double_array_ptr(buf), buf

    buf = array.array("d", x)
    if len(buf) != expected_dim:
        raise ValueError(f"Expected vector of length {expected_dim}, got {len(buf)}")
    return _double_array_ptr(buf), buf


def _as_double_ptr_2d(X: Any, expected_dim: int) -> tuple[Any, int, Any]:
    """Return ``(double*, n_ticks, keepalive_ref)`` for a row-major ``(n_ticks, D)`` matrix.

    Zero-copy for C-contiguous native ``float64`` 2D NumPy arrays and flat ``array('d')``.
    """
    iface = getattr(X, "__array_interface__", None)
    if isinstance(iface, dict):
        shape = iface.get("shape")
        if not shape or len(shape) != 2:
            raise ValueError(f"Expected 2D array of shape (n_ticks, {expected_dim}), got {shape}")
        n_ticks, cols = int(shape[0]), int(shape[1])
        if cols != expected_dim:
            raise ValueError(f"Expected {expected_dim} columns, got {cols}")
        if n_ticks == 0:
            return None, 0, X
        typestr = iface.get("typestr")
        strides = iface.get("strides")
        is_c_contig = strides is None or bool(
            getattr(getattr(X, "flags", None), "c_contiguous", False)
        )
        if typestr == _NATIVE_F8_TYPESTR and is_c_contig:
            data = iface.get("data")
            if isinstance(data, tuple) and isinstance(data[0], int):
                return ctypes.cast(data[0], _DOUBLE_P), n_ticks, X
            if hasattr(X, "ctypes"):
                return ctypes.cast(int(X.ctypes.data), _DOUBLE_P), n_ticks, X
        np_mod = sys.modules.get("numpy")
        if np_mod is not None:
            arr = np_mod.ascontiguousarray(X, dtype=np_mod.float64)
            return ctypes.cast(int(arr.ctypes.data), _DOUBLE_P), n_ticks, arr
        buf = array.array("d")
        for row in X:
            buf.extend(float(v) for v in row)
        return _double_array_ptr(buf), n_ticks, buf

    if isinstance(X, array.array) and X.typecode == "d":
        total = len(X)
        if total % expected_dim != 0:
            raise ValueError(
                f"Flat array('d') length {total} is not divisible by dim={expected_dim}"
            )
        n_ticks = total // expected_dim
        if n_ticks == 0:
            return None, 0, X
        return _double_array_ptr(X), n_ticks, X

    if isinstance(X, memoryview) and X.format == "d" and X.c_contiguous and X.ndim == 2:
        mv_shape = X.shape
        if mv_shape is None:
            raise ValueError(f"Expected 2D memoryview with {expected_dim} columns")
        n_ticks, cols = int(mv_shape[0]), int(mv_shape[1])
        if cols != expected_dim:
            raise ValueError(f"Expected {expected_dim} columns, got {cols}")
        if n_ticks == 0:
            return None, 0, X
        if not X.readonly:
            c_arr = (ctypes.c_double * (n_ticks * expected_dim)).from_buffer(X)
            return ctypes.cast(c_arr, _DOUBLE_P), n_ticks, (X, c_arr)
        buf = array.array("d", X.cast("d"))
        return _double_array_ptr(buf), n_ticks, buf

    buf = array.array("d")
    n_ticks = 0
    for row in X:
        prev = len(buf)
        buf.extend(row)
        row_len = len(buf) - prev
        if row_len != expected_dim:
            raise ValueError(
                f"Row {n_ticks} has length {row_len}, expected dim={expected_dim}"
            )
        n_ticks += 1
    if n_ticks == 0:
        return None, 0, buf
    return _double_array_ptr(buf), n_ticks, buf


def _alloc_double_buffer(size: int) -> tuple[array.array[float], Any]:
    buf = array.array("d", [0.0]) * size
    return buf, _double_array_ptr(buf) if size > 0 else None


def _reshape_2d(flat: array.array[float], rows: int, cols: int) -> list[list[float]]:
    return [flat[r * cols : (r + 1) * cols].tolist() for r in range(rows)]


# =========================================================================
# High-level result dataclasses
# =========================================================================


@dataclass(frozen=True, slots=True)
class TickResult:
    """Result of pushing a single multivariate tick into :class:`CausalSieve`."""

    tick: int
    mahalanobis_sq: float
    e_value: float
    shock_weight: float
    max_exo_z: float
    top_root_channel: int
    top_root_name: str | None
    alarm: bool
    refresh_count: int
    exo_z: list[float] | None = None
    innov: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tick": self.tick,
            "mahalanobis_sq": self.mahalanobis_sq,
            "e_value": self.e_value,
            "shock_weight": self.shock_weight,
            "max_exo_z": self.max_exo_z,
            "top_root_channel": self.top_root_channel,
            "top_root_name": self.top_root_name,
            "alarm": self.alarm,
            "refresh_count": self.refresh_count,
        }
        if self.exo_z is not None:
            d["exo_z"] = self.exo_z
        if self.innov is not None:
            d["innov"] = self.innov
        return d


@dataclass(frozen=True, slots=True)
class RootAttributionItem:
    """Single ranked channel entry in :class:`RootAttribution`."""

    rank: int
    channel: int
    name: str
    root_score: float
    peak_exo_z: float
    raw_z: float
    onset_tick: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "channel": self.channel,
            "name": self.name,
            "root_score": self.root_score,
            "peak_exo_z": self.peak_exo_z,
            "raw_z": self.raw_z,
            "onset_tick": self.onset_tick,
        }


@dataclass(frozen=True, slots=True)
class RootAttribution:
    """Ranked causal root-cause attribution across all ``D`` channels."""

    ranked_channels: list[int]
    ranked_names: list[str]
    root_scores: list[float]
    peak_exo_z: list[float]
    raw_z: list[float]
    onset_tick: list[int]
    items: list[RootAttributionItem]

    @property
    def top_channel(self) -> int:
        return self.ranked_channels[0] if self.ranked_channels else -1

    @property
    def top_name(self) -> str | None:
        return self.ranked_names[0] if self.ranked_names else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_channel": self.top_channel,
            "top_name": self.top_name,
            "ranked_channels": self.ranked_channels,
            "ranked_names": self.ranked_names,
            "root_scores": self.root_scores,
            "peak_exo_z": self.peak_exo_z,
            "raw_z": self.raw_z,
            "onset_tick": self.onset_tick,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class GrangerResult:
    """Exact Block-Schur conditional Granger causality matrices (``[src][dst]``)."""

    f_stat: list[list[float]]
    partial_r2: list[list[float]]
    channel_names: list[str]

    def top_edges(
        self,
        *,
        min_f: float = 3.0,
        min_partial_r2: float = 0.02,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Return statistically meaningful lagged directed edges sorted by F-stat."""
        dim = len(self.channel_names)
        edges: list[dict[str, Any]] = []
        for src in range(dim):
            for dst in range(dim):
                if src == dst:
                    continue
                f_val = self.f_stat[src][dst]
                r2_val = self.partial_r2[src][dst]
                if math.isfinite(f_val) and f_val >= min_f and r2_val >= min_partial_r2:
                    edges.append(
                        {
                            "src_index": src,
                            "dst_index": dst,
                            "src": self.channel_names[src],
                            "dst": self.channel_names[dst],
                            "f_stat": f_val,
                            "partial_r2": r2_val,
                        }
                    )
        edges.sort(key=lambda e: (e["f_stat"], e["partial_r2"]), reverse=True)
        return edges[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_names": self.channel_names,
            "f_stat": self.f_stat,
            "partial_r2": self.partial_r2,
        }


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    """Online innovation precision matrix ``Theta`` and partial correlations."""

    theta: list[list[float]]
    pcorr: list[list[float]]
    channel_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_names": self.channel_names,
            "theta": self.theta,
            "pcorr": self.pcorr,
        }


@dataclass(frozen=True, slots=True)
class RawSlice:
    """100% lossless exact FP64 values recovered from the native ring buffer."""

    channel: int
    channel_name: str
    actual_start: int
    end_tick: int
    values: list[float]

    @property
    def ticks(self) -> list[int]:
        return list(range(self.actual_start, self.actual_start + len(self.values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "channel_name": self.channel_name,
            "actual_start": self.actual_start,
            "end_tick": self.end_tick,
            "count": len(self.values),
            "values": self.values,
        }


@dataclass(frozen=True, slots=True)
class PLACompression:
    """L_inf-error-bounded Swing-Filter piecewise linear approximation knots."""

    channel: int
    channel_name: str
    start_tick: int
    end_tick: int
    max_abs_err: float
    knots: list[tuple[int, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "channel_name": self.channel_name,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "max_abs_err": self.max_abs_err,
            "n_knots": len(self.knots),
            "knots": [[t, v] for t, v in self.knots],
        }


@dataclass(frozen=True, slots=True)
class EngineState:
    """Snapshot of internal VAR(p) weights ``B`` and Freivalds inverse residual."""

    B: list[list[float]]
    freivalds_rel_err: float
    feature_dim: int
    dim: int
    lag: int
    ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "lag": self.lag,
            "feature_dim": self.feature_dim,
            "ticks": self.ticks,
            "freivalds_rel_err": self.freivalds_rel_err,
            "B": self.B,
        }


@dataclass(frozen=True, slots=True)
class ScreenedBatchResult:
    """Output of Tier-2 high-dimensional :meth:`ScreenedSieve.push_batch`."""

    top_channels: list[int]
    top_names: list[str]
    top_scores: list[float]
    last_z_scores: list[float] | None
    ticks_pushed: int

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ticks_pushed": self.ticks_pushed,
            "top_channels": self.top_channels,
            "top_names": self.top_names,
            "top_scores": self.top_scores,
        }
        if self.last_z_scores is not None:
            d["last_z_scores"] = self.last_z_scores
        return d


@dataclass(frozen=True, slots=True)
class ScreenedActiveGraph:
    """Promoted active Tier-1 causal subgraph from :class:`ScreenedSieve`."""

    active_channels: list[int]
    active_names: list[str]
    granger_f: list[list[float]]
    pcorr: list[list[float]]
    active_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_count": self.active_count,
            "active_channels": self.active_channels,
            "active_names": self.active_names,
            "granger_f": self.granger_f,
            "pcorr": self.pcorr,
        }


# =========================================================================
# Tier-1 CausalSieve Engine
# =========================================================================


def _normalize_channel_names(dim: int, channel_names: Sequence[str] | None) -> list[str]:
    if channel_names is None:
        return [f"ch_{i}" for i in range(dim)]
    names = [str(n) for n in channel_names]
    if len(names) != dim:
        raise ValueError(f"channel_names length ({len(names)}) must equal dim ({dim})")
    return names


class CausalSieve:
    """Streaming multivariate causal discovery & root-cause localization engine.

    Wraps ``cs_engine_t`` from ``libcausalsieve``. Supports context-manager
    cleanup, zero-copy NumPy / ``array('d')`` ingest, exact Block-Schur
    conditional Granger causality, innovation precision whitening, online
    Hyvärinen-Smith contemporaneous orientation, counterfactual interventions,
    Swing-Filter L_inf PLA compression, and lossless raw FP64 slice recovery.
    """

    def __init__(
        self,
        dim: int,
        lag: int = 2,
        channel_names: Sequence[str] | None = None,
        forgetting: float = 0.995,
        ridge: float = 1e-2,
        precision_ridge: float = 1e-2,
        shock_threshold: float = 4.0,
        e_alpha: float = 0.01,
        warmup_ticks: int = 0,
        refresh_interval: int = 0,
        nonlinear_hinge: bool = False,
        raw_ring_capacity: int = 8192,
        normalize_scale: bool = True,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if lag < 1 or lag > 16:
            raise ValueError(f"lag must be in [1, 16], got {lag}")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}")
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if precision_ridge <= 0.0:
            raise ValueError(f"precision_ridge must be > 0, got {precision_ridge}")
        if shock_threshold < 0.0:
            raise ValueError(f"shock_threshold must be >= 0, got {shock_threshold}")
        if not (0.0 < e_alpha < 1.0):
            raise ValueError(f"e_alpha must be in (0, 1), got {e_alpha}")

        self._dim = int(dim)
        self._lag = int(lag)
        self._channel_names = _normalize_channel_names(self._dim, channel_names)
        self._name_to_idx = {name: i for i, name in enumerate(self._channel_names)}
        self._e_alpha = float(e_alpha)
        self._forgetting = float(forgetting)
        self._raw_ring_capacity = int(raw_ring_capacity) if raw_ring_capacity > 0 else 8192
        self._normalize_scale = bool(normalize_scale)
        self._nonlinear_hinge = bool(nonlinear_hinge)

        self._lib = load_library()
        cfg = CSConfig()
        self._lib.cs_config_default(ctypes.byref(cfg), self._dim, self._lag)
        cfg.forgetting = float(forgetting)
        cfg.ridge = float(ridge)
        cfg.precision_ridge = float(precision_ridge)
        cfg.shock_threshold = float(shock_threshold)
        cfg.e_alpha = float(e_alpha)
        cfg.warmup_ticks = int(warmup_ticks)
        cfg.refresh_interval = int(refresh_interval)
        cfg.nonlinear_hinge = 1 if nonlinear_hinge else 0
        cfg.raw_ring_capacity = int(raw_ring_capacity)
        cfg.normalize_scale = 1 if normalize_scale else 0

        eng = self._lib.cs_create(ctypes.byref(cfg))
        if not eng:
            raise CausalSieveError("cs_create returned NULL (invalid config or allocation failure)")
        self._eng: int | None = eng
        self._last_result: TickResult | None = None
        self._max_e_value: float = 1.0
        self._any_alarm: bool = False
        self._alarm_first_tick: int | None = None

    def _require_open(self) -> int:
        if self._eng is None:
            raise CausalSieveError("CausalSieve engine has been closed")
        return self._eng

    def _last_error(self) -> str:
        if self._eng is None:
            return "engine closed"
        raw = self._lib.cs_last_error(self._eng)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def _check_rc(self, rc: int, op: str) -> None:
        if rc != 0:
            err = self._last_error()
            detail = f": {err}" if err else ""
            raise CausalSieveError(f"{op} failed with status {rc}{detail}")

    def resolve_channel(self, channel: int | str) -> int:
        """Resolve a channel index or name string into an integer channel index."""
        if isinstance(channel, bool):
            raise TypeError("channel must be int or str, not bool")
        if isinstance(channel, int):
            if channel < 0 or channel >= self._dim:
                raise IndexError(f"channel index {channel} out of range [0, {self._dim})")
            return channel
        if isinstance(channel, str):
            idx = self._name_to_idx.get(channel)
            if idx is not None:
                return idx
            # Allow numeric string fallback (e.g. from CLI/MCP)
            if channel.isdigit():
                num = int(channel)
                if 0 <= num < self._dim:
                    return num
            raise KeyError(
                f"Unknown channel {channel!r}. Available channels: {self._channel_names}"
            )
        raise TypeError(f"channel must be int or str, got {type(channel).__name__}")

    def _wrap_tick_result(
        self,
        res: CSTickResult,
        exo_z: list[float] | None = None,
        innov: list[float] | None = None,
    ) -> TickResult:
        top_ch = int(res.top_root_channel)
        top_name = self._channel_names[top_ch] if 0 <= top_ch < self._dim else None
        alarm = bool(res.alarm)
        e_val = float(res.e_value)
        if e_val > self._max_e_value:
            self._max_e_value = e_val
        if alarm:
            if not self._any_alarm:
                self._alarm_first_tick = int(res.tick)
            self._any_alarm = True
        wrapped = TickResult(
            tick=int(res.tick),
            mahalanobis_sq=float(res.mahalanobis_sq),
            e_value=e_val,
            shock_weight=float(res.shock_weight),
            max_exo_z=float(res.max_exo_z),
            top_root_channel=top_ch,
            top_root_name=top_name,
            alarm=alarm,
            refresh_count=int(res.refresh_count),
            exo_z=exo_z,
            innov=innov,
        )
        self._last_result = wrapped
        return wrapped

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def feature_dim(self) -> int:
        eng = self._require_open()
        return int(self._lib.cs_feature_dim(eng))

    @property
    def ticks(self) -> int:
        eng = self._require_open()
        return int(self._lib.cs_tick_count(eng))

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def simd_isa(self) -> str:
        raw = self._lib.cs_simd_isa()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    @property
    def e_alpha(self) -> float:
        return self._e_alpha

    @property
    def raw_ring_capacity(self) -> int:
        return self._raw_ring_capacity

    @property
    def last_result(self) -> TickResult | None:
        return self._last_result

    @property
    def max_e_value(self) -> float:
        return self._max_e_value

    @property
    def any_alarm(self) -> bool:
        return self._any_alarm

    @property
    def alarm_first_tick(self) -> int | None:
        return self._alarm_first_tick

    @property
    def closed(self) -> bool:
        return self._eng is None

    def push(self, x: Any, *, return_vectors: bool = True) -> TickResult:
        """Strictly causal single-tick ingest.

        Raises :class:`ValueError` with zero state mutation if any entry is NaN/Inf.
        """
        eng = self._require_open()
        x_ptr, _keep = _as_double_ptr_1d(x, self._dim)
        out = CSTickResult()
        exo_buf: array.array[float] | None = None
        inn_buf: array.array[float] | None = None
        exo_ptr: Any = None
        inn_ptr: Any = None
        if return_vectors:
            exo_buf, exo_ptr = _alloc_double_buffer(self._dim)
            inn_buf, inn_ptr = _alloc_double_buffer(self._dim)

        rc = self._lib.cs_push(eng, x_ptr, ctypes.byref(out), exo_ptr, inn_ptr)
        if rc != 0:
            err = self._last_error()
            raise ValueError(
                f"cs_push rejected input (status {rc})" + (f": {err}" if err else "")
            )
        return self._wrap_tick_result(
            out,
            exo_z=exo_buf.tolist() if exo_buf is not None else None,
            innov=inn_buf.tolist() if inn_buf is not None else None,
        )

    def push_batch(
        self,
        X: Any,
        *,
        return_results: bool = True,
        return_exo_z: bool = False,
        return_innov: bool = False,
    ) -> list[TickResult] | None:
        """High-throughput atomic batch ingest over ``(n_ticks, D)`` rows.

        Pre-validates all inputs in C before mutating state (all-or-nothing).
        """
        eng = self._require_open()
        x_ptr, n_ticks, _keep = _as_double_ptr_2d(X, self._dim)
        if n_ticks == 0:
            return [] if return_results else None

        capture_c_results = return_results or n_ticks <= 250_000
        res_arr = (CSTickResult * n_ticks)() if capture_c_results else None
        res_ptr = ctypes.cast(res_arr, ctypes.POINTER(CSTickResult)) if res_arr is not None else None

        exo_buf = inn_buf = None
        exo_ptr = inn_ptr = None
        if return_results and return_exo_z:
            exo_buf, exo_ptr = _alloc_double_buffer(n_ticks * self._dim)
        if return_results and return_innov:
            inn_buf, inn_ptr = _alloc_double_buffer(n_ticks * self._dim)

        rc = self._lib.cs_push_batch(eng, x_ptr, n_ticks, res_ptr, exo_ptr, inn_ptr)
        if rc != 0:
            err = self._last_error()
            raise ValueError(
                f"cs_push_batch rejected batch (status {rc})" + (f": {err}" if err else "")
            )

        if not return_results:
            if res_arr is not None:
                for idx in range(n_ticks):
                    r = res_arr[idx]
                    e_v = float(r.e_value)
                    if e_v > self._max_e_value:
                        self._max_e_value = e_v
                    if r.alarm:
                        if not self._any_alarm:
                            self._alarm_first_tick = int(r.tick)
                        self._any_alarm = True
                self._wrap_tick_result(res_arr[n_ticks - 1])
            return None

        assert res_arr is not None
        out_list: list[TickResult] = []
        d = self._dim
        for i in range(n_ticks):
            ez = exo_buf[i * d : (i + 1) * d].tolist() if exo_buf is not None else None
            iv = inn_buf[i * d : (i + 1) * d].tolist() if inn_buf is not None else None
            out_list.append(self._wrap_tick_result(res_arr[i], exo_z=ez, innov=iv))
        return out_list

    def granger_matrix(self) -> GrangerResult:
        """Compute exact Block-Schur conditional Granger causality across all ``D x D`` pairs."""
        eng = self._require_open()
        total = self._dim * self._dim
        f_buf, f_ptr = _alloc_double_buffer(total)
        r2_buf, r2_ptr = _alloc_double_buffer(total)
        rc = self._lib.cs_granger_matrix(eng, f_ptr, r2_ptr)
        self._check_rc(rc, "cs_granger_matrix")
        return GrangerResult(
            f_stat=_reshape_2d(f_buf, self._dim, self._dim),
            partial_r2=_reshape_2d(r2_buf, self._dim, self._dim),
            channel_names=list(self._channel_names),
        )

    def precision_matrix(self) -> PrecisionResult:
        """Return online innovation precision matrix ``Theta`` and partial correlations."""
        eng = self._require_open()
        total = self._dim * self._dim
        th_buf, th_ptr = _alloc_double_buffer(total)
        pc_buf, pc_ptr = _alloc_double_buffer(total)
        rc = self._lib.cs_precision_matrix(eng, th_ptr, pc_ptr)
        self._check_rc(rc, "cs_precision_matrix")
        return PrecisionResult(
            theta=_reshape_2d(th_buf, self._dim, self._dim),
            pcorr=_reshape_2d(pc_buf, self._dim, self._dim),
            channel_names=list(self._channel_names),
        )

    def contemporaneous_orientation(self) -> list[list[float]]:
        """Return online Hyvärinen-Smith non-Gaussian LiNGAM orientation matrix ``R`` (``D x D``).

        Positive ``R[i][j]`` indicates contemporaneous direction ``i -> j``.
        """
        eng = self._require_open()
        total = self._dim * self._dim
        r_buf, r_ptr = _alloc_double_buffer(total)
        rc = self._lib.cs_contemporaneous_orientation(eng, r_ptr)
        self._check_rc(rc, "cs_contemporaneous_orientation")
        return _reshape_2d(r_buf, self._dim, self._dim)

    def root_attribution(self, lookback: int = 0) -> RootAttribution:
        """Compute causal root-cause attribution over the last ``lookback`` ticks."""
        if lookback < 0:
            raise ValueError(f"lookback must be >= 0, got {lookback}")
        eng = self._require_open()
        d = self._dim
        scores_buf, scores_ptr = _alloc_double_buffer(d)
        exo_buf, exo_ptr = _alloc_double_buffer(d)
        raw_buf, raw_ptr = _alloc_double_buffer(d)
        onset_arr = (ctypes.c_int64 * d)()
        ranked_arr = (ctypes.c_int32 * d)()

        rc = self._lib.cs_root_attribution(
            eng,
            int(lookback),
            scores_ptr,
            exo_ptr,
            raw_ptr,
            ctypes.cast(onset_arr, _INT64_P),
            ctypes.cast(ranked_arr, _INT32_P),
        )
        self._check_rc(rc, "cs_root_attribution")

        scores = scores_buf.tolist()
        peak_exo = exo_buf.tolist()
        raw_z = raw_buf.tolist()
        onsets = [int(onset_arr[i]) for i in range(d)]
        ranked = [int(ranked_arr[i]) for i in range(d)]
        ranked_names = [self._channel_names[c] for c in ranked]
        items = [
            RootAttributionItem(
                rank=rank_idx + 1,
                channel=ch,
                name=self._channel_names[ch],
                root_score=scores[ch],
                peak_exo_z=peak_exo[ch],
                raw_z=raw_z[ch],
                onset_tick=onsets[ch],
            )
            for rank_idx, ch in enumerate(ranked)
        ]
        return RootAttribution(
            ranked_channels=ranked,
            ranked_names=ranked_names,
            root_scores=scores,
            peak_exo_z=peak_exo,
            raw_z=raw_z,
            onset_tick=onsets,
            items=items,
        )

    def counterfactual_do(
        self,
        channel: int | str,
        delta: float,
        horizon: int = 10,
    ) -> list[list[float]]:
        """Simulate exact ``h``-step counterfactual ``do(channel += delta)`` propagation."""
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not math.isfinite(delta):
            raise ValueError(f"delta must be finite, got {delta}")
        eng = self._require_open()
        ch_idx = self.resolve_channel(channel)
        out_buf, out_ptr = _alloc_double_buffer(int(horizon) * self._dim)
        rc = self._lib.cs_counterfactual_do(eng, ch_idx, float(delta), int(horizon), out_ptr)
        self._check_rc(rc, "cs_counterfactual_do")
        return _reshape_2d(out_buf, int(horizon), self._dim)

    def get_raw_slice(
        self,
        channel: int | str,
        start_tick: int,
        end_tick: int,
    ) -> RawSlice:
        """Recover 100% lossless exact FP64 raw values in ``[start_tick, end_tick)``."""
        if start_tick < 0 or end_tick < start_tick:
            raise ValueError(
                f"Invalid tick range [{start_tick}, {end_tick}); require 0 <= start_tick <= end_tick"
            )
        eng = self._require_open()
        ch_idx = self.resolve_channel(channel)
        name = self._channel_names[ch_idx]
        req_len = int(end_tick - start_tick)
        if req_len == 0:
            return RawSlice(
                channel=ch_idx,
                channel_name=name,
                actual_start=int(start_tick),
                end_tick=int(start_tick),
                values=[],
            )
        ticks_now = self.ticks
        ring_oldest = max(0, ticks_now - self._raw_ring_capacity)
        eff_end = min(int(end_tick), ticks_now)
        eff_start = max(int(start_tick), ring_oldest)
        if eff_end <= eff_start:
            return RawSlice(
                channel=ch_idx,
                channel_name=name,
                actual_start=eff_start,
                end_tick=eff_start,
                values=[],
            )
        alloc_n = eff_end - eff_start
        vals_buf, vals_ptr = _alloc_double_buffer(alloc_n)
        out_count = ctypes.c_uint64(0)
        out_actual_start = ctypes.c_uint64(0)
        rc = self._lib.cs_get_raw_slice(
            eng,
            ch_idx,
            eff_start,
            eff_end,
            vals_ptr,
            ctypes.byref(out_count),
            ctypes.byref(out_actual_start),
        )
        self._check_rc(rc, "cs_get_raw_slice")
        n = int(out_count.value)
        actual_start = int(out_actual_start.value)
        return RawSlice(
            channel=ch_idx,
            channel_name=name,
            actual_start=actual_start,
            end_tick=actual_start + n,
            values=vals_buf[:n].tolist(),
        )

    def compress_pla(
        self,
        channel: int | str,
        start_tick: int,
        end_tick: int,
        max_abs_err: float,
    ) -> PLACompression:
        """Compute L_inf-error-bounded Swing-Filter PLA knots over ``[start_tick, end_tick)``."""
        if start_tick < 0 or end_tick < start_tick:
            raise ValueError(
                f"Invalid tick range [{start_tick}, {end_tick}); require 0 <= start_tick <= end_tick"
            )
        if not math.isfinite(max_abs_err) or max_abs_err < 0.0:
            raise ValueError(f"max_abs_err must be finite and >= 0, got {max_abs_err}")
        eng = self._require_open()
        ch_idx = self.resolve_channel(channel)
        name = self._channel_names[ch_idx]
        ticks_now = self.ticks
        ring_oldest = max(0, ticks_now - self._raw_ring_capacity)
        eff_end = min(int(end_tick), ticks_now)
        eff_start = max(int(start_tick), ring_oldest)
        span = eff_end - eff_start
        if span <= 0:
            return PLACompression(
                channel=ch_idx,
                channel_name=name,
                start_tick=eff_start,
                end_tick=eff_start,
                max_abs_err=float(max_abs_err),
                knots=[],
            )
        max_knots = span + 2
        knot_ticks = (ctypes.c_uint64 * max_knots)()
        knot_vals_buf, knot_vals_ptr = _alloc_double_buffer(max_knots)
        out_n = ctypes.c_size_t(0)

        rc = self._lib.cs_compress_channel_pla(
            eng,
            ch_idx,
            eff_start,
            eff_end,
            float(max_abs_err),
            ctypes.cast(knot_ticks, _UINT64_P),
            knot_vals_ptr,
            max_knots,
            ctypes.byref(out_n),
        )
        self._check_rc(rc, "cs_compress_channel_pla")
        n = int(out_n.value)
        knots = [(int(knot_ticks[i]), float(knot_vals_buf[i])) for i in range(n)]
        return PLACompression(
            channel=ch_idx,
            channel_name=name,
            start_tick=eff_start,
            end_tick=eff_end,
            max_abs_err=float(max_abs_err),
            knots=knots,
        )

    def inspect_state(self) -> EngineState:
        """Inspect VAR(p) weights ``B`` (``K x D``) and Freivalds inverse residual."""
        eng = self._require_open()
        k = int(self._lib.cs_feature_dim(eng))
        d = self._dim
        b_buf, b_ptr = _alloc_double_buffer(k * d)
        rel_err = ctypes.c_double(0.0)
        rc = self._lib.cs_inspect_state(eng, b_ptr, ctypes.byref(rel_err))
        self._check_rc(rc, "cs_inspect_state")
        return EngineState(
            B=_reshape_2d(b_buf, k, d),
            freivalds_rel_err=float(rel_err.value),
            feature_dim=k,
            dim=d,
            lag=self._lag,
            ticks=self.ticks,
        )

    def force_refresh(self) -> None:
        """Force an exact Cholesky refresh from sufficient statistics."""
        eng = self._require_open()
        rc = self._lib.cs_force_refresh(eng)
        self._check_rc(rc, "cs_force_refresh")

    def reset(self) -> None:
        """Reset all engine state back to tick 0 while keeping configuration."""
        eng = self._require_open()
        self._lib.cs_reset(eng)
        self._last_result = None
        self._max_e_value = 1.0
        self._any_alarm = False
        self._alarm_first_tick = None

    def digest(
        self,
        *,
        top_k: int = 5,
        lookback: int = 0,
        max_edges: int = 8,
        min_granger_f: float = 3.0,
        min_partial_r2: float = 0.02,
        min_pcorr: float = 0.25,
        max_err: float = 0.05,
        pla_channels: int = 3,
        pla_max_knots: int = 12,
        tokens_budget: int | None = None,
        stream_id: str | None = None,
        raw_bytes: int | None = None,
        store: StreamStore | None = None,
    ) -> CausalDigest:
        """Build a Caveman-compressed, lossless-recoverable Causal Digest."""
        from causalsieve.digest import build_digest

        return build_digest(
            self,
            top_k=top_k,
            lookback=lookback,
            max_edges=max_edges,
            min_granger_f=min_granger_f,
            min_partial_r2=min_partial_r2,
            min_pcorr=min_pcorr,
            max_err=max_err,
            pla_channels=pla_channels,
            pla_max_knots=pla_max_knots,
            tokens_budget=tokens_budget,
            stream_id=stream_id,
            raw_bytes=raw_bytes,
            store=store,
        )

    def close(self) -> None:
        """Release underlying native engine memory (idempotent)."""
        if self._eng is not None:
            eng = self._eng
            self._eng = None
            self._lib.cs_destroy(eng)

    def __del__(self) -> None:
        if getattr(self, "_eng", None) is not None and getattr(self, "_lib", None) is not None:
            self.close()

    def __enter__(self) -> CausalSieve:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._eng is None else f"ticks={self.ticks}"
        return f"CausalSieve(dim={self._dim}, lag={self._lag}, {state})"


# =========================================================================
# Tier-2 ScreenedSieve Engine (D = 128 .. 16,384+)
# =========================================================================


class ScreenedSieve:
    """High-dimensional Tier-2 screened causal engine (``D = 128 .. 16,384+``).

    Performs ``O(D)`` scale-free robust pre-whitening and promotes the Top-M
    most active channels into an embedded Tier-1 causal engine.
    """

    def __init__(
        self,
        dim: int,
        active_m: int = 32,
        lag: int = 2,
        channel_names: Sequence[str] | None = None,
        forgetting: float = 0.995,
        ridge: float = 1e-2,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if active_m < 1:
            raise ValueError(f"active_m must be >= 1, got {active_m}")
        if lag < 1 or lag > 16:
            raise ValueError(f"lag must be in [1, 16], got {lag}")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}")
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")

        self._dim = int(dim)
        self._active_m = min(int(active_m), self._dim)
        self._lag = int(lag)
        self._channel_names = _normalize_channel_names(self._dim, channel_names)
        self._ticks = 0

        self._lib = load_library()
        handle = self._lib.cs_screened_create(
            self._dim,
            self._active_m,
            self._lag,
            float(forgetting),
            float(ridge),
        )
        if not handle:
            raise CausalSieveError("cs_screened_create returned NULL")
        self._handle: int | None = handle

    def _require_open(self) -> int:
        if self._handle is None:
            raise CausalSieveError("ScreenedSieve engine has been closed")
        return self._handle

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def active_m(self) -> int:
        return self._active_m

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def closed(self) -> bool:
        return self._handle is None

    def push_batch(
        self,
        X: Any,
        *,
        top_k: int = 10,
        return_all_z: bool = False,
    ) -> ScreenedBatchResult:
        """Push ``(n_ticks, D)`` batch through Tier-2 screening + Top-M promotion."""
        handle = self._require_open()
        x_ptr, n_ticks, _keep = _as_double_ptr_2d(X, self._dim)
        if n_ticks == 0:
            return ScreenedBatchResult(
                top_channels=[],
                top_names=[],
                top_scores=[],
                last_z_scores=[] if return_all_z else None,
                ticks_pushed=0,
            )

        k = max(1, min(int(top_k), self._dim))
        if return_all_z:
            z_buf, z_ptr = _alloc_double_buffer(self._dim)
        else:
            z_buf, z_ptr = None, None

        top_ch_arr = (ctypes.c_int32 * k)()
        top_sc_buf, top_sc_ptr = _alloc_double_buffer(k)

        rc = self._lib.cs_screened_push_batch(
            handle,
            x_ptr,
            n_ticks,
            z_ptr,
            ctypes.cast(top_ch_arr, _INT32_P),
            top_sc_ptr,
            k,
        )
        if rc != 0:
            raise ValueError(f"cs_screened_push_batch failed with status {rc}")

        self._ticks += n_ticks
        top_channels: list[int] = []
        top_names: list[str] = []
        top_scores: list[float] = []
        for i in range(k):
            ch = int(top_ch_arr[i])
            if 0 <= ch < self._dim:
                top_channels.append(ch)
                top_names.append(self._channel_names[ch])
                top_scores.append(float(top_sc_buf[i]))

        return ScreenedBatchResult(
            top_channels=top_channels,
            top_names=top_names,
            top_scores=top_scores,
            last_z_scores=z_buf.tolist() if z_buf is not None else None,
            ticks_pushed=n_ticks,
        )

    def active_graph(self) -> ScreenedActiveGraph:
        """Retrieve the promoted active subgraph (channels, Granger F, partial corr)."""
        handle = self._require_open()
        m = self._active_m
        ch_arr = (ctypes.c_int32 * m)()
        gf_buf, gf_ptr = _alloc_double_buffer(m * m)
        pc_buf, pc_ptr = _alloc_double_buffer(m * m)
        out_count = ctypes.c_uint32(0)

        rc = self._lib.cs_screened_active_graph(
            handle,
            ctypes.cast(ch_arr, _INT32_P),
            gf_ptr,
            pc_ptr,
            ctypes.byref(out_count),
        )
        if rc != 0:
            raise CausalSieveError(f"cs_screened_active_graph failed with status {rc}")

        count = min(int(out_count.value), m)
        active_ch = [int(ch_arr[i]) for i in range(count) if 0 <= int(ch_arr[i]) < self._dim]
        count = len(active_ch)
        active_names = [self._channel_names[c] for c in active_ch]
        if count == 0:
            return ScreenedActiveGraph(
                active_channels=[],
                active_names=[],
                granger_f=[],
                pcorr=[],
                active_count=0,
            )
        return ScreenedActiveGraph(
            active_channels=active_ch,
            active_names=active_names,
            granger_f=_reshape_2d(gf_buf, count, count),
            pcorr=_reshape_2d(pc_buf, count, count),
            active_count=count,
        )

    def close(self) -> None:
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            self._lib.cs_screened_destroy(handle)

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None and getattr(self, "_lib", None) is not None:
            self.close()

    def __enter__(self) -> ScreenedSieve:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._handle is None else f"ticks={self._ticks}"
        return f"ScreenedSieve(dim={self._dim}, active_m={self._active_m}, lag={self._lag}, {state})"


# =========================================================================
# Zero-dependency CSV streaming helper (stdlib csv only)
# =========================================================================


@dataclass
class CSVAnalysisResult:
    """Result of streaming a CSV file through :class:`CausalSieve` or :class:`ScreenedSieve`."""

    path: str
    sieve: CausalSieve | ScreenedSieve
    channel_names: list[str]
    timestamp_column: str | None
    timestamps: list[str]
    skipped_columns: list[str]
    n_rows: int
    raw_bytes: int
    imputed_cells: int
    skipped_rows: int
    screened: bool
    screened_result: ScreenedBatchResult | None = None

    def digest(self, **kwargs: Any) -> CausalDigest:
        """Build a Causal Digest for a Tier-1 CSV analysis result."""
        if not isinstance(self.sieve, CausalSieve):
            raise TypeError(
                "CausalDigest requires a Tier-1 CausalSieve instance "
                "(pass use_screened=False or reduce columns below screened_threshold)."
            )
        kwargs.setdefault("raw_bytes", self.raw_bytes)
        return self.sieve.digest(**kwargs)

    def close(self) -> None:
        self.sieve.close()

    def __enter__(self) -> CSVAnalysisResult:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _try_parse_finite_float(cell: str) -> float | None:
    s = cell.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _detect_delimiter(sample_line: str) -> str:
    best = ","
    best_count = sample_line.count(",")
    for delim in ("\t", ";", "|"):
        c = sample_line.count(delim)
        if c > best_count:
            best = delim
            best_count = c
    return best


def _is_monotonic_nondecreasing(vals: Sequence[float]) -> bool:
    if len(vals) < 2:
        return True
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def analyze_csv(
    path: str | os.PathLike[str],
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    ridge: float = 1e-2,
    precision_ridge: float = 1e-2,
    shock_threshold: float = 4.0,
    e_alpha: float = 0.01,
    warmup_ticks: int = 0,
    refresh_interval: int = 0,
    nonlinear_hinge: bool = False,
    raw_ring_capacity: int = 0,
    normalize_scale: bool = True,
    screened_threshold: int = 128,
    use_screened: bool | None = None,
    active_m: int = 32,
    delimiter: str | None = None,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
    top_k: int = 10,
) -> CSVAnalysisResult:
    """Stream a multivariate CSV file into :class:`CausalSieve` or :class:`ScreenedSieve`.

    Uses only Python's standard ``csv`` module. Automatically detects header
    rows, timestamp/index columns, and non-numeric columns. Any isolated
    missing/NaN cells after the first valid row are strictly causally
    forward-filled (never look-ahead imputed).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    raw_bytes = file_path.stat().st_size

    with file_path.open("r", encoding="utf-8-sig", newline="") as fh:
        first_nonempty = ""
        for line in fh:
            if line.strip():
                first_nonempty = line
                break
        if not first_nonempty:
            raise ValueError(f"CSV file is empty: {file_path}")
        delim = delimiter if delimiter is not None else _detect_delimiter(first_nonempty)
        fh.seek(0)
        reader = csv.reader(fh, delimiter=delim)

        raw_rows: list[list[str]] = []
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            raw_rows.append(row)
            if max_rows is not None and len(raw_rows) >= max_rows + 1:
                break

    if not raw_rows:
        raise ValueError(f"CSV file contains no data rows: {file_path}")

    # Detect whether row 0 is a header or purely numeric data.
    first_row = raw_rows[0]
    first_row_all_numeric = all(_try_parse_finite_float(c) is not None for c in first_row)
    if first_row_all_numeric:
        headers = [f"ch_{i}" for i in range(len(first_row))]
        data_rows = raw_rows[:max_rows] if max_rows is not None else raw_rows
    else:
        headers = [c.strip() or f"col_{i}" for i, c in enumerate(first_row)]
        data_rows = raw_rows[1:]
        if max_rows is not None:
            data_rows = data_rows[:max_rows]

    if not data_rows:
        raise ValueError(f"CSV file has header but zero data rows: {file_path}")

    sample_rows = data_rows[: min(len(data_rows), 32)]

    # Classify columns: numeric vs timestamp vs skipped non-numeric.
    requested_set = set(columns) if columns is not None else None
    timestamp_col_idx: int | None = None
    numeric_indices: list[int] = []
    skipped_columns: list[str] = []

    for col_idx, name in enumerate(headers):
        norm_name = name.strip().lower()
        sample_vals: list[float] = []
        non_empty = 0
        for r in sample_rows:
            if col_idx >= len(r):
                continue
            cell = r[col_idx].strip()
            if not cell:
                continue
            non_empty += 1
            parsed = _try_parse_finite_float(cell)
            if parsed is not None:
                sample_vals.append(parsed)

        is_mostly_numeric = non_empty > 0 and len(sample_vals) >= max(1, (non_empty + 1) // 2)

        if requested_set is not None:
            if name in requested_set:
                if not is_mostly_numeric:
                    raise ValueError(f"Requested column {name!r} is not numeric")
                numeric_indices.append(col_idx)
            elif timestamp_col_idx is None and norm_name in _TIMESTAMP_HEADER_NAMES:
                timestamp_col_idx = col_idx
            else:
                skipped_columns.append(name)
            continue

        # Automatic timestamp / index column detection
        if timestamp_col_idx is None:
            if not is_mostly_numeric and (col_idx == 0 or norm_name in _TIMESTAMP_HEADER_NAMES):
                timestamp_col_idx = col_idx
                continue
            if (
                is_mostly_numeric
                and norm_name in _TIMESTAMP_HEADER_NAMES
                and (col_idx == 0 or len(headers) > 2)
                and _is_monotonic_nondecreasing(sample_vals)
            ):
                timestamp_col_idx = col_idx
                continue

        if is_mostly_numeric:
            numeric_indices.append(col_idx)
        else:
            skipped_columns.append(name)

    if requested_set is not None:
        found_names = {headers[i] for i in numeric_indices}
        missing = [c for c in columns if c not in found_names] if columns else []
        if missing:
            raise ValueError(f"Requested columns not found in CSV: {missing}")

    if not numeric_indices:
        raise ValueError(f"No numeric metric columns found in CSV: {file_path}")

    channel_names = [headers[i] for i in numeric_indices]
    dim = len(numeric_indices)
    ts_col_name = headers[timestamp_col_idx] if timestamp_col_idx is not None else None

    # Causal forward-fill buffer for isolated missing cells (zero look-ahead).
    flat = array.array("d")
    timestamps: list[str] = []
    last_valid: list[float | None] = [None] * dim
    imputed_cells = 0
    skipped_rows = 0
    n_valid_rows = 0

    for row in data_rows:
        parsed_row: list[float] = []
        row_ok = True
        row_imputes = 0
        for j, col_idx in enumerate(numeric_indices):
            cell = row[col_idx] if col_idx < len(row) else ""
            val = _try_parse_finite_float(cell)
            if val is None:
                prev = last_valid[j]
                if prev is None:
                    row_ok = False
                    break
                val = prev
                row_imputes += 1
            parsed_row.append(val)

        if not row_ok:
            skipped_rows += 1
            continue

        for j, val in enumerate(parsed_row):
            last_valid[j] = val
        flat.extend(parsed_row)
        imputed_cells += row_imputes
        if timestamp_col_idx is not None:
            ts_val = row[timestamp_col_idx].strip() if timestamp_col_idx < len(row) else ""
            timestamps.append(ts_val)
        n_valid_rows += 1

    if n_valid_rows == 0:
        raise ValueError(f"No complete numeric rows found in CSV: {file_path}")

    should_screen = (
        use_screened if use_screened is not None else (dim >= int(screened_threshold))
    )

    if should_screen:
        screened_sieve = ScreenedSieve(
            dim=dim,
            active_m=active_m,
            lag=lag,
            channel_names=channel_names,
            forgetting=forgetting,
            ridge=ridge,
        )
        screened_res = screened_sieve.push_batch(flat, top_k=top_k, return_all_z=False)
        return CSVAnalysisResult(
            path=str(file_path),
            sieve=screened_sieve,
            channel_names=channel_names,
            timestamp_column=ts_col_name,
            timestamps=timestamps,
            skipped_columns=skipped_columns,
            n_rows=n_valid_rows,
            raw_bytes=raw_bytes,
            imputed_cells=imputed_cells,
            skipped_rows=skipped_rows,
            screened=True,
            screened_result=screened_res,
        )

    ring_cap = raw_ring_capacity if raw_ring_capacity > 0 else min(max(8192, n_valid_rows), 65536)
    sieve = CausalSieve(
        dim=dim,
        lag=lag,
        channel_names=channel_names,
        forgetting=forgetting,
        ridge=ridge,
        precision_ridge=precision_ridge,
        shock_threshold=shock_threshold,
        e_alpha=e_alpha,
        warmup_ticks=warmup_ticks,
        refresh_interval=refresh_interval,
        nonlinear_hinge=nonlinear_hinge,
        raw_ring_capacity=ring_cap,
        normalize_scale=normalize_scale,
    )
    sieve.push_batch(flat, return_results=False)
    return CSVAnalysisResult(
        path=str(file_path),
        sieve=sieve,
        channel_names=channel_names,
        timestamp_column=ts_col_name,
        timestamps=timestamps,
        skipped_columns=skipped_columns,
        n_rows=n_valid_rows,
        raw_bytes=raw_bytes,
        imputed_cells=imputed_cells,
        skipped_rows=skipped_rows,
        screened=False,
        screened_result=None,
    )


def iterate_csv_rows(
    path: str | os.PathLike[str],
    *,
    delimiter: str | None = None,
) -> Iterable[list[str]]:
    """Lightweight streaming CSV row iterator (pure stdlib helper)."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as fh:
        first = ""
        for line in fh:
            if line.strip():
                first = line
                break
        if not first:
            return
        delim = delimiter if delimiter is not None else _detect_delimiter(first)
        fh.seek(0)
        yield from csv.reader(fh, delimiter=delim)

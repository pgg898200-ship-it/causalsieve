"""Direct ctypes binding to csrc/causalsieve.h for mathematical oracle tests and benchmarks.

This module mirrors the frozen C99 ABI in csrc/causalsieve.h (CAUSALSIEVE_ABI_VERSION == 1)
so verification tests can inspect internal state (cs_inspect_state, cs_force_refresh,
exact normalize_scale=0 mode, scalar vs AVX2 shared libraries) independently of the
high-level Python wrapper.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIB_PATH = REPO_ROOT / "build" / "libcausalsieve.so"
SCALAR_LIB_PATH = REPO_ROOT / "build" / "libcausalsieve_scalar.so"


class CSConfig(ctypes.Structure):
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


_VOID_P = ctypes.c_void_p
_DBL_P = ctypes.POINTER(ctypes.c_double)
_I32_P = ctypes.POINTER(ctypes.c_int32)
_I64_P = ctypes.POINTER(ctypes.c_int64)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_U64_P = ctypes.POINTER(ctypes.c_uint64)
_SIZE_P = ctypes.POINTER(ctypes.c_size_t)


def load_lib(lib_path: Optional[Path | str] = None) -> ctypes.CDLL:
    """Load libcausalsieve.so (or scalar build) and bind full C99 ABI prototypes."""
    path = Path(lib_path) if lib_path is not None else DEFAULT_LIB_PATH
    if not path.exists():
        # Also check python/causalsieve/ bundled library path if installed there
        alt = REPO_ROOT / "python" / "causalsieve" / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"CausalSieve shared library not found at {path} (or {alt}). "
                "Build csrc/ first."
            )

    lib = ctypes.CDLL(str(path))

    lib.cs_config_default.argtypes = [ctypes.POINTER(CSConfig), ctypes.c_uint32, ctypes.c_uint32]
    lib.cs_config_default.restype = None

    lib.cs_create.argtypes = [ctypes.POINTER(CSConfig)]
    lib.cs_create.restype = _VOID_P

    lib.cs_destroy.argtypes = [_VOID_P]
    lib.cs_destroy.restype = None

    lib.cs_reset.argtypes = [_VOID_P]
    lib.cs_reset.restype = None

    lib.cs_simd_isa.argtypes = []
    lib.cs_simd_isa.restype = ctypes.c_char_p

    lib.cs_last_error.argtypes = [_VOID_P]
    lib.cs_last_error.restype = ctypes.c_char_p

    lib.cs_tick_count.argtypes = [_VOID_P]
    lib.cs_tick_count.restype = ctypes.c_uint64

    lib.cs_dim.argtypes = [_VOID_P]
    lib.cs_dim.restype = ctypes.c_uint32

    lib.cs_lag.argtypes = [_VOID_P]
    lib.cs_lag.restype = ctypes.c_uint32

    lib.cs_feature_dim.argtypes = [_VOID_P]
    lib.cs_feature_dim.restype = ctypes.c_uint32

    lib.cs_push.argtypes = [
        _VOID_P,
        _DBL_P,
        ctypes.POINTER(CSTickResult),
        _DBL_P,
        _DBL_P,
    ]
    lib.cs_push.restype = ctypes.c_int

    lib.cs_push_batch.argtypes = [
        _VOID_P,
        _DBL_P,
        ctypes.c_size_t,
        ctypes.POINTER(CSTickResult),
        _DBL_P,
        _DBL_P,
    ]
    lib.cs_push_batch.restype = ctypes.c_int

    lib.cs_granger_matrix.argtypes = [_VOID_P, _DBL_P, _DBL_P]
    lib.cs_granger_matrix.restype = ctypes.c_int

    lib.cs_precision_matrix.argtypes = [_VOID_P, _DBL_P, _DBL_P]
    lib.cs_precision_matrix.restype = ctypes.c_int

    lib.cs_contemporaneous_orientation.argtypes = [_VOID_P, _DBL_P]
    lib.cs_contemporaneous_orientation.restype = ctypes.c_int

    lib.cs_root_attribution.argtypes = [
        _VOID_P,
        ctypes.c_uint32,
        _DBL_P,
        _DBL_P,
        _DBL_P,
        _I64_P,
        _I32_P,
    ]
    lib.cs_root_attribution.restype = ctypes.c_int

    lib.cs_counterfactual_do.argtypes = [
        _VOID_P,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_uint32,
        _DBL_P,
    ]
    lib.cs_counterfactual_do.restype = ctypes.c_int

    lib.cs_get_raw_slice.argtypes = [
        _VOID_P,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        _DBL_P,
        _U64_P,
        _U64_P,
    ]
    lib.cs_get_raw_slice.restype = ctypes.c_int

    lib.cs_compress_channel_pla.argtypes = [
        _VOID_P,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_double,
        _U64_P,
        _DBL_P,
        ctypes.c_size_t,
        _SIZE_P,
    ]
    lib.cs_compress_channel_pla.restype = ctypes.c_int

    lib.cs_inspect_state.argtypes = [_VOID_P, _DBL_P, _DBL_P]
    lib.cs_inspect_state.restype = ctypes.c_int

    lib.cs_force_refresh.argtypes = [_VOID_P]
    lib.cs_force_refresh.restype = ctypes.c_int

    # Tier-2 Screened Engine
    lib.cs_screened_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.cs_screened_create.restype = _VOID_P

    lib.cs_screened_destroy.argtypes = [_VOID_P]
    lib.cs_screened_destroy.restype = None

    lib.cs_screened_push_batch.argtypes = [
        _VOID_P,
        _DBL_P,
        ctypes.c_size_t,
        _DBL_P,
        _I32_P,
        _DBL_P,
        ctypes.c_uint32,
    ]
    lib.cs_screened_push_batch.restype = ctypes.c_int

    lib.cs_screened_active_graph.argtypes = [
        _VOID_P,
        _I32_P,
        _DBL_P,
        _DBL_P,
        _U32_P,
    ]
    lib.cs_screened_active_graph.restype = ctypes.c_int

    return lib


class RawEngine:
    """Thin RAII wrapper around cs_engine_t for clean test and benchmark code."""

    def __init__(
        self,
        dim: int,
        lag: int = 2,
        *,
        forgetting: Optional[float] = None,
        ridge: Optional[float] = None,
        precision_ridge: Optional[float] = None,
        shock_threshold: Optional[float] = None,
        e_alpha: Optional[float] = None,
        warmup_ticks: Optional[int] = None,
        refresh_interval: Optional[int] = None,
        nonlinear_hinge: Optional[int] = None,
        raw_ring_capacity: Optional[int] = None,
        normalize_scale: Optional[int] = None,
        lib_path: Optional[Path | str] = None,
    ) -> None:
        self.lib = load_lib(lib_path)
        cfg = CSConfig()
        self.lib.cs_config_default(ctypes.byref(cfg), int(dim), int(lag))
        if forgetting is not None:
            cfg.forgetting = float(forgetting)
        if ridge is not None:
            cfg.ridge = float(ridge)
        if precision_ridge is not None:
            cfg.precision_ridge = float(precision_ridge)
        if shock_threshold is not None:
            cfg.shock_threshold = float(shock_threshold)
        if e_alpha is not None:
            cfg.e_alpha = float(e_alpha)
        if warmup_ticks is not None:
            cfg.warmup_ticks = int(warmup_ticks)
        if refresh_interval is not None:
            cfg.refresh_interval = int(refresh_interval)
        if nonlinear_hinge is not None:
            cfg.nonlinear_hinge = int(nonlinear_hinge)
        if raw_ring_capacity is not None:
            cfg.raw_ring_capacity = int(raw_ring_capacity)
        if normalize_scale is not None:
            cfg.normalize_scale = int(normalize_scale)

        self.cfg = cfg
        self.handle = self.lib.cs_create(ctypes.byref(cfg))
        if not self.handle:
            raise RuntimeError(f"cs_create failed for dim={dim}, lag={lag}")
        self.dim = int(self.lib.cs_dim(self.handle))
        self.lag = int(self.lib.cs_lag(self.handle))
        self.feature_dim = int(self.lib.cs_feature_dim(self.handle))

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.cs_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RawEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def simd_isa(self) -> str:
        raw = self.lib.cs_simd_isa()
        return raw.decode("utf-8") if raw else "unknown"

    @property
    def tick_count(self) -> int:
        return int(self.lib.cs_tick_count(self.handle))

    @property
    def last_error(self) -> str:
        raw = self.lib.cs_last_error(self.handle)
        return raw.decode("utf-8") if raw else ""

    def reset(self) -> None:
        self.lib.cs_reset(self.handle)

    def force_refresh(self) -> int:
        return int(self.lib.cs_force_refresh(self.handle))

    def push(
        self,
        x: np.ndarray,
        *,
        return_vectors: bool = True,
    ) -> tuple[int, CSTickResult, Optional[np.ndarray], Optional[np.ndarray]]:
        arr = np.ascontiguousarray(x, dtype=np.float64)
        assert arr.shape == (self.dim,), f"Expected ({self.dim},), got {arr.shape}"
        res = CSTickResult()
        if return_vectors:
            exo = np.empty(self.dim, dtype=np.float64)
            innov = np.empty(self.dim, dtype=np.float64)
            rc = self.lib.cs_push(
                self.handle,
                arr.ctypes.data_as(_DBL_P),
                ctypes.byref(res),
                exo.ctypes.data_as(_DBL_P),
                innov.ctypes.data_as(_DBL_P),
            )
            return int(rc), res, exo, innov
        rc = self.lib.cs_push(
            self.handle,
            arr.ctypes.data_as(_DBL_P),
            ctypes.byref(res),
            None,
            None,
        )
        return int(rc), res, None, None

    def push_batch(
        self,
        X: np.ndarray,
        *,
        return_results: bool = True,
        return_matrices: bool = False,
    ) -> tuple[int, Optional[ctypes.Array], Optional[np.ndarray], Optional[np.ndarray]]:
        arr = np.ascontiguousarray(X, dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[1] == self.dim
        n = arr.shape[0]
        res_arr = (CSTickResult * n)() if return_results else None
        exo_mat = np.empty((n, self.dim), dtype=np.float64) if return_matrices else None
        innov_mat = np.empty((n, self.dim), dtype=np.float64) if return_matrices else None
        rc = self.lib.cs_push_batch(
            self.handle,
            arr.ctypes.data_as(_DBL_P),
            n,
            res_arr,
            exo_mat.ctypes.data_as(_DBL_P) if exo_mat is not None else None,
            innov_mat.ctypes.data_as(_DBL_P) if innov_mat is not None else None,
        )
        return int(rc), res_arr, exo_mat, innov_mat

    def granger_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        f_stat = np.empty((self.dim, self.dim), dtype=np.float64)
        pr2 = np.empty((self.dim, self.dim), dtype=np.float64)
        rc = self.lib.cs_granger_matrix(
            self.handle,
            f_stat.ctypes.data_as(_DBL_P),
            pr2.ctypes.data_as(_DBL_P),
        )
        if rc != 0:
            raise RuntimeError(f"cs_granger_matrix failed rc={rc}: {self.last_error}")
        return f_stat, pr2

    def precision_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        theta = np.empty((self.dim, self.dim), dtype=np.float64)
        pcorr = np.empty((self.dim, self.dim), dtype=np.float64)
        rc = self.lib.cs_precision_matrix(
            self.handle,
            theta.ctypes.data_as(_DBL_P),
            pcorr.ctypes.data_as(_DBL_P),
        )
        if rc != 0:
            raise RuntimeError(f"cs_precision_matrix failed rc={rc}: {self.last_error}")
        return theta, pcorr

    def contemporaneous_orientation(self) -> np.ndarray:
        r_mat = np.empty((self.dim, self.dim), dtype=np.float64)
        rc = self.lib.cs_contemporaneous_orientation(
            self.handle,
            r_mat.ctypes.data_as(_DBL_P),
        )
        if rc != 0:
            raise RuntimeError(f"cs_contemporaneous_orientation failed rc={rc}: {self.last_error}")
        return r_mat

    def root_attribution(self, lookback_ticks: int = 0) -> dict[str, np.ndarray]:
        root_scores = np.empty(self.dim, dtype=np.float64)
        peak_exo_z = np.empty(self.dim, dtype=np.float64)
        raw_z = np.empty(self.dim, dtype=np.float64)
        onset_tick = np.empty(self.dim, dtype=np.int64)
        ranked = np.empty(self.dim, dtype=np.int32)
        rc = self.lib.cs_root_attribution(
            self.handle,
            int(lookback_ticks),
            root_scores.ctypes.data_as(_DBL_P),
            peak_exo_z.ctypes.data_as(_DBL_P),
            raw_z.ctypes.data_as(_DBL_P),
            onset_tick.ctypes.data_as(_I64_P),
            ranked.ctypes.data_as(_I32_P),
        )
        if rc != 0:
            raise RuntimeError(f"cs_root_attribution failed rc={rc}: {self.last_error}")
        return {
            "root_scores": root_scores,
            "peak_exo_z": peak_exo_z,
            "raw_z": raw_z,
            "onset_tick": onset_tick,
            "ranked_channels": ranked,
        }

    def counterfactual_do(
        self, clamp_channel: int, delta_value: float, horizon: int
    ) -> np.ndarray:
        traj = np.empty((horizon, self.dim), dtype=np.float64)
        rc = self.lib.cs_counterfactual_do(
            self.handle,
            int(clamp_channel),
            float(delta_value),
            int(horizon),
            traj.ctypes.data_as(_DBL_P),
        )
        if rc != 0:
            raise RuntimeError(f"cs_counterfactual_do failed rc={rc}: {self.last_error}")
        return traj

    def get_raw_slice(
        self, channel: int, start_tick: int, end_tick: int
    ) -> tuple[np.ndarray, int]:
        n_req = max(0, int(end_tick) - int(start_tick))
        buf = np.empty(max(1, n_req), dtype=np.float64)
        out_count = ctypes.c_uint64(0)
        out_actual = ctypes.c_uint64(0)
        rc = self.lib.cs_get_raw_slice(
            self.handle,
            int(channel),
            int(start_tick),
            int(end_tick),
            buf.ctypes.data_as(_DBL_P),
            ctypes.byref(out_count),
            ctypes.byref(out_actual),
        )
        if rc != 0:
            raise RuntimeError(f"cs_get_raw_slice failed rc={rc}: {self.last_error}")
        return buf[: out_count.value].copy(), int(out_actual.value)

    def compress_channel_pla(
        self,
        channel: int,
        start_tick: int,
        end_tick: int,
        max_abs_err: float,
        max_knots: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_req = max(2, int(end_tick) - int(start_tick))
        cap = int(max_knots) if max_knots is not None else n_req + 2
        ticks = np.empty(cap, dtype=np.uint64)
        vals = np.empty(cap, dtype=np.float64)
        out_n = ctypes.c_size_t(0)
        rc = self.lib.cs_compress_channel_pla(
            self.handle,
            int(channel),
            int(start_tick),
            int(end_tick),
            float(max_abs_err),
            ticks.ctypes.data_as(_U64_P),
            vals.ctypes.data_as(_DBL_P),
            cap,
            ctypes.byref(out_n),
        )
        if rc != 0:
            raise RuntimeError(f"cs_compress_channel_pla failed rc={rc}: {self.last_error}")
        k = int(out_n.value)
        return ticks[:k].copy(), vals[:k].copy()

    def inspect_state(self) -> tuple[np.ndarray, float]:
        B = np.empty((self.feature_dim, self.dim), dtype=np.float64)
        err = ctypes.c_double(0.0)
        rc = self.lib.cs_inspect_state(
            self.handle,
            B.ctypes.data_as(_DBL_P),
            ctypes.byref(err),
        )
        if rc != 0:
            raise RuntimeError(f"cs_inspect_state failed rc={rc}: {self.last_error}")
        return B, float(err.value)


class RawScreenedEngine:
    """Thin RAII wrapper around Tier-2 cs_screened_t."""

    def __init__(
        self,
        dim: int,
        active_m: int = 32,
        lag: int = 2,
        forgetting: float = 0.995,
        ridge: float = 1e-2,
        lib_path: Optional[Path | str] = None,
    ) -> None:
        self.lib = load_lib(lib_path)
        self.dim = int(dim)
        self.active_m = int(active_m)
        self.handle = self.lib.cs_screened_create(
            int(dim), int(active_m), int(lag), float(forgetting), float(ridge)
        )
        if not self.handle:
            raise RuntimeError(f"cs_screened_create failed for dim={dim}, active_m={active_m}")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.cs_screened_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "RawScreenedEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def push_batch(
        self, X: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.ascontiguousarray(X, dtype=np.float64)
        assert arr.ndim == 2 and arr.shape[1] == self.dim
        last_z = np.empty(self.dim, dtype=np.float64)
        k = min(int(top_k), self.dim)
        top_ch = np.empty(k, dtype=np.int32)
        top_sc = np.empty(k, dtype=np.float64)
        rc = self.lib.cs_screened_push_batch(
            self.handle,
            arr.ctypes.data_as(_DBL_P),
            arr.shape[0],
            last_z.ctypes.data_as(_DBL_P),
            top_ch.ctypes.data_as(_I32_P),
            top_sc.ctypes.data_as(_DBL_P),
            k,
        )
        if rc != 0:
            raise RuntimeError(f"cs_screened_push_batch failed rc={rc}")
        return last_z, top_ch, top_sc

    def active_graph(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_ch = np.empty(self.active_m, dtype=np.int32)
        granger_f = np.empty((self.active_m, self.active_m), dtype=np.float64)
        pcorr = np.empty((self.active_m, self.active_m), dtype=np.float64)
        active_cnt = ctypes.c_uint32(0)
        rc = self.lib.cs_screened_active_graph(
            self.handle,
            active_ch.ctypes.data_as(_I32_P),
            granger_f.ctypes.data_as(_DBL_P),
            pcorr.ctypes.data_as(_DBL_P),
            ctypes.byref(active_cnt),
        )
        if rc != 0:
            raise RuntimeError(f"cs_screened_active_graph failed rc={rc}")
        m = int(active_cnt.value)
        return active_ch[:m].copy(), granger_f[:m, :m].copy(), pcorr[:m, :m].copy()

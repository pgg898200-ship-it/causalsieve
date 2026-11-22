"""Zero-dependency drop-in ecosystem adapters for CausalSieve.

All adapters are strictly duck-typed using only the Python standard library
(zero hard runtime dependency on ``pandas``, ``polars``, ``numpy``, or HTTP
clients):

  - :func:`analyze_dataframe`: Pandas DataFrame, Polars DataFrame, or
    ``Mapping[str, Sequence[float]]`` dict-of-lists -> ``(sieve, digest)``
  - :func:`analyze_prometheus`: Prometheus HTTP API ``/api/v1/query_range``
    matrix JSON (dict or string) -> ``(sieve, digest)``
  - :func:`analyze_otlp_metrics`: OpenTelemetry OTLP/JSON
    ``ExportMetricsServiceRequest`` (gauge, sum, histogram) -> ``(sieve, digest)``
  - :func:`analyze_jsonl`: NDJSON / JSONL metric streams (file path, raw
    multiline string, or iterable of JSON lines / dicts) -> ``(sieve, digest)``
  - :func:`rcaeval_rca`: Drop-in ``RCAEval`` microservice root-cause interface
    supporting both symmetric ``inject_time`` and blind (``inject_time=None``)
    localization -> ``{"ranks", "scores", "metric_top_k", "digest"}``
"""

from __future__ import annotations

import array
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causalsieve._native import (
    _TIMESTAMP_HEADER_NAMES,
    CausalSieve,
    ScreenedBatchResult,
    ScreenedSieve,
    simd_isa,
)
from causalsieve.digest import (
    CausalDigest,
    CausalEdge,
    RootRow,
    StreamStore,
    _finalize_token_stats,
    build_digest,
    estimate_raw_tokens,
)

if TYPE_CHECKING:
    pass


# =========================================================================
# Duck-typed tabular & numeric helpers (zero external dependencies)
# =========================================================================


def _sequence_to_list(col: Any) -> list[Any]:
    """Convert a Pandas Series, Polars Series, NumPy array, or Python sequence to a list."""
    if isinstance(col, list):
        return col
    if isinstance(col, tuple):
        return list(col)
    to_list = getattr(col, "to_list", None)
    if callable(to_list):
        res = to_list()
        if isinstance(res, list):
            return res
    tolist = getattr(col, "tolist", None)
    if callable(tolist):
        res = tolist()
        if isinstance(res, list):
            return res
    return list(col)


def _extract_tabular_columns(data: Any) -> tuple[list[str], dict[str, list[Any]], int]:
    """Extract ordered column names and column value lists via pure duck typing.

    Supports:
      - ``Mapping[str, Sequence[Any]]`` (dict-of-lists / dict-of-arrays)
      - Polars ``DataFrame`` (via ``to_dict(as_series=False)`` or ``df[col].to_list()``)
      - Pandas ``DataFrame`` (via ``df.columns`` and ``df[col].tolist()``)
    """
    col_names: list[str] = []
    cols: dict[str, list[Any]] = {}

    if isinstance(data, Mapping):
        for raw_key, raw_vals in data.items():
            name = str(raw_key)
            col_names.append(name)
            cols[name] = _sequence_to_list(raw_vals)
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        raw_cols = list(data.columns)
        # Fast path for Polars DataFrame: to_dict(as_series=False)
        polars_dict: dict[str, Any] | None = None
        to_dict = getattr(data, "to_dict", None)
        if callable(to_dict):
            try:
                candidate = to_dict(as_series=False)
                if isinstance(candidate, Mapping) and all(
                    isinstance(v, list) for v in candidate.values()
                ):
                    polars_dict = {str(k): v for k, v in candidate.items()}
            except TypeError:
                polars_dict = None

        if polars_dict is not None:
            for c in raw_cols:
                name = str(c)
                col_names.append(name)
                cols[name] = list(polars_dict[name])
        else:
            for c in raw_cols:
                name = str(c)
                col_names.append(name)
                cols[name] = _sequence_to_list(data[c])
    else:
        raise TypeError(
            "Expected a Pandas DataFrame, Polars DataFrame, or Mapping[str, Sequence[float]], "
            f"got {type(data).__name__}"
        )

    if not col_names:
        raise ValueError("Input table has zero columns")

    lengths = {len(cols[name]) for name in col_names}
    if len(lengths) != 1:
        detail = {name: len(cols[name]) for name in col_names[:8]}
        raise ValueError(f"All columns must have the same length; got mismatched lengths: {detail}")

    n_rows = next(iter(lengths))
    if n_rows == 0:
        raise ValueError("Input table has zero rows")
    return col_names, cols, n_rows


def _coerce_finite_float(val: Any) -> float | None:
    """Convert a scalar cell to a finite Python ``float``, or ``None`` if null/non-numeric."""
    if val is None or isinstance(val, (bool, str, bytes, bytearray)):
        return None
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        kind = getattr(dtype, "kind", "")
        if kind in ("M", "m", "b", "U", "S"):
            return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _parse_prom_float(val: Any) -> float | None:
    """Parse a Prometheus sample value (numeric string or float) into a finite float."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, (str, bytes, bytearray)):
        s = val.decode("utf-8", errors="ignore").strip() if not isinstance(val, str) else val.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return _coerce_finite_float(val)


def _is_monotonic_nondecreasing(vals: Sequence[float]) -> bool:
    if len(vals) < 2:
        return True
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def _column_is_numeric(values: Sequence[Any]) -> bool:
    """Return True if a column contains finite numeric observations (not strings/dates/bools)."""
    n = len(values)
    # Sample up to 64 evenly spaced positions plus head/tail so sparse leading NaNs don't hide data.
    indices: Sequence[int]
    if n <= 64:
        indices = range(n)
    else:
        step = max(1, n // 64)
        idx_set = set(range(0, n, step))
        idx_set.update(range(min(16, n)))
        idx_set.update(range(max(0, n - 16), n))
        indices = sorted(idx_set)

    non_null = 0
    finite_numeric = 0
    non_numeric = 0

    for i in indices:
        v = values[i]
        if v is None:
            continue
        if isinstance(v, (bool, str, bytes, bytearray)):
            non_numeric += 1
            continue
        # Detect NaN/NaT/pd.NA without importing pandas or numpy
        if isinstance(v, float) and not math.isfinite(v):
            continue
        cls_name = type(v).__name__
        if cls_name in ("NAType", "NaTType"):
            continue
        f = _coerce_finite_float(v)
        if f is not None:
            non_null += 1
            finite_numeric += 1
        else:
            # Distinguish float NaN/Inf scalars (e.g., np.float64('nan')) from strings/datetimes
            try:
                raw_f = float(v)
                if not math.isfinite(raw_f):
                    continue
            except (TypeError, ValueError, OverflowError):
                non_numeric += 1

    if finite_numeric > 0:
        return finite_numeric >= non_numeric
    # Fallback full scan if the sample happened to hit only NaNs/None
    if non_numeric == 0:
        for v in values:
            if _coerce_finite_float(v) is not None:
                return True
    return False


def _select_columns(
    col_names: list[str],
    cols: dict[str, list[Any]],
    *,
    timestamp_col: str | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[list[str], str | None]:
    """Select numeric feature columns and identify an optional timestamp column."""
    if timestamp_col is not None and timestamp_col not in cols:
        raise ValueError(
            f"Specified timestamp_col {timestamp_col!r} not found in columns: {col_names}"
        )

    detected_ts: str | None = timestamp_col
    requested_set = set(columns) if columns is not None else None
    numeric_cols: list[str] = []

    for idx, name in enumerate(col_names):
        if name == detected_ts:
            continue
        vals = cols[name]
        is_num = _column_is_numeric(vals)
        norm = name.strip().lower()

        if requested_set is not None:
            if name in requested_set:
                if not is_num:
                    raise ValueError(f"Requested column {name!r} is not numeric")
                numeric_cols.append(name)
            elif detected_ts is None and norm in _TIMESTAMP_HEADER_NAMES:
                detected_ts = name
            continue

        if detected_ts is None and norm in _TIMESTAMP_HEADER_NAMES:
            if not is_num:
                detected_ts = name
                continue
            sample_floats = [
                f for f in (_coerce_finite_float(v) for v in vals[:32]) if f is not None
            ]
            if (idx == 0 or len(col_names) > 2) and _is_monotonic_nondecreasing(sample_floats):
                detected_ts = name
                continue

        if is_num:
            numeric_cols.append(name)

    if requested_set is not None:
        found = set(numeric_cols)
        missing = [c for c in (columns or ()) if c not in found]
        if missing:
            raise ValueError(f"Requested columns not found: {missing}")

    if not numeric_cols:
        raise ValueError("No numeric metric columns found in input data")

    return numeric_cols, detected_ts


def _pack_causal_ffill_matrix(
    col_lists: list[list[Any]],
    row_order: Sequence[int] | None = None,
) -> tuple[array.array[float], int]:
    """Pack columns into a row-major ``array('d')`` with strict causal forward-fill.

    Never performs look-ahead or backward-fill: isolated NaNs/nulls inherit the
    most recent prior finite observation on that channel (or ``0.0`` prior to
    the first finite observation).
    """
    dim = len(col_lists)
    n_rows = len(col_lists[0]) if dim > 0 else 0
    order = range(n_rows) if row_order is None else row_order

    flat = array.array("d")
    last_valid = [0.0] * dim
    for r_idx in order:
        for j in range(dim):
            parsed = _coerce_finite_float(col_lists[j][r_idx])
            if parsed is not None:
                last_valid[j] = parsed
                flat.append(parsed)
            else:
                flat.append(last_valid[j])
    return flat, n_rows


def _build_screened_digest(
    screened: ScreenedSieve,
    batch_res: ScreenedBatchResult,
    *,
    top_k: int = 5,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_pcorr: float = 0.25,
    stream_id: str | None = None,
    raw_bytes: int | None = None,
) -> CausalDigest:
    """Build a :class:`CausalDigest` for a Tier-2 :class:`ScreenedSieve` result."""
    dim = screened.dim
    ticks = screened.ticks
    names = screened.channel_names

    if stream_id is not None:
        sid = stream_id
    else:
        h = hashlib.sha256()
        h.update(f"screened;d={dim};m={screened.active_m};p={screened.lag};t={ticks};".encode())
        h.update("|".join(names[:64]).encode("utf-8"))
        sid = h.hexdigest()[:12]

    eff_k = max(1, min(int(top_k), len(batch_res.top_channels)))
    root_rows: list[RootRow] = []
    for idx in range(eff_k):
        ch = batch_res.top_channels[idx]
        ch_name = batch_res.top_names[idx]
        sc = float(batch_res.top_scores[idx])
        verdict = "EXOGENOUS_ROOT" if idx == 0 and sc >= 2.0 else (
            "CONTRIBUTING_ANOMALY" if sc >= 2.0 else "NOMINAL"
        )
        root_rows.append(
            RootRow(
                rank=idx + 1,
                channel=ch,
                name=ch_name,
                root_score=sc,
                peak_exo_z=sc,
                raw_z=sc,
                onset_tick=-1,
                verdict=verdict,
            )
        )

    graph = screened.active_graph()
    lagged_candidates: list[tuple[float, CausalEdge]] = []
    contemp_candidates: list[tuple[float, CausalEdge]] = []
    m = graph.active_count
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            f_val = graph.granger_f[i][j]
            if math.isfinite(f_val) and f_val >= min_granger_f:
                lagged_candidates.append(
                    (
                        f_val,
                        CausalEdge(
                            kind="lagged",
                            src=graph.active_names[i],
                            dst=graph.active_names[j],
                            directed=True,
                            f_stat=f_val,
                            partial_r2=0.0,
                        ),
                    )
                )
        for j in range(i + 1, m):
            pc = graph.pcorr[i][j]
            if math.isfinite(pc) and abs(pc) >= min_pcorr:
                contemp_candidates.append(
                    (
                        abs(pc),
                        CausalEdge(
                            kind="contemp",
                            src=graph.active_names[i],
                            dst=graph.active_names[j],
                            directed=False,
                            pcorr=pc,
                            lingam_r=0.0,
                        ),
                    )
                )

    lagged_candidates.sort(key=lambda pair: pair[0], reverse=True)
    contemp_candidates.sort(key=lambda pair: pair[0], reverse=True)
    n_lagged = min(len(lagged_candidates), max(1, (max_edges * 2 + 2) // 3))
    n_contemp = min(len(contemp_candidates), max(0, max_edges - n_lagged))
    edges = [e for _, e in lagged_candidates[:n_lagged]] + [
        e for _, e in contemp_candidates[:n_contemp]
    ]

    top_score = batch_res.top_scores[0] if batch_res.top_scores else 0.0
    raw_tokens_est = estimate_raw_tokens(ticks, dim, names, raw_bytes=raw_bytes)
    handles = [f"slice://{sid}/{r.name}/0..{ticks}" for r in root_rows[:3]]

    digest = CausalDigest(
        stream_id=sid,
        ticks=ticks,
        dim=dim,
        lag=screened.lag,
        simd_isa=simd_isa(),
        alarm=top_score >= 3.0,
        e_value=1.0,
        e_threshold=100.0,
        freivalds_rel_err=0.0,
        refresh_count=0,
        retained_start=0,
        retained_end=ticks,
        root_causes=root_rows,
        victim_note=None,
        edges=edges,
        trajectories=[],
        recovery_handles=handles,
        raw_tokens_est=raw_tokens_est,
    )
    _finalize_token_stats(digest)
    return digest


def _run_matrix_analysis(
    flat: array.array[float],
    n_rows: int,
    channel_names: list[str],
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    top_k: int = 5,
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
    lookback: int = 0,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_partial_r2: float = 0.02,
    min_pcorr: float = 0.25,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    raw_bytes: int | None = None,
    store: StreamStore | None = None,
) -> tuple[CausalSieve | ScreenedSieve, CausalDigest]:
    """Execute Tier-1 or Tier-2 engine on a flat row-major matrix and return ``(sieve, digest)``."""
    dim = len(channel_names)
    should_screen = (
        use_screened if use_screened is not None else (dim >= int(screened_threshold))
    )

    if should_screen:
        eff_m = min(int(active_m), dim)
        screened = ScreenedSieve(
            dim=dim,
            active_m=eff_m,
            lag=lag,
            channel_names=channel_names,
            forgetting=forgetting,
            ridge=ridge,
        )
        batch_res = screened.push_batch(flat, top_k=max(int(top_k), eff_m), return_all_z=False)
        digest = _build_screened_digest(
            screened,
            batch_res,
            top_k=top_k,
            max_edges=max_edges,
            min_granger_f=min_granger_f,
            min_pcorr=min_pcorr,
            stream_id=stream_id,
            raw_bytes=raw_bytes,
        )
        return screened, digest

    ring_cap = raw_ring_capacity if raw_ring_capacity > 0 else min(max(8192, n_rows), 65536)
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
    digest = build_digest(
        sieve,
        top_k=top_k,
        lookback=lookback,
        max_edges=max_edges,
        min_granger_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        min_pcorr=min_pcorr,
        max_err=max_err,
        tokens_budget=tokens_budget,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        store=store,
    )
    return sieve, digest


# =========================================================================
# 1. DataFrame / Dict-of-Lists Adapter
# =========================================================================


def analyze_dataframe(
    df: Any,
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    top_k: int = 5,
    timestamp_col: str | None = None,
    columns: Sequence[str] | None = None,
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
    lookback: int = 0,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_partial_r2: float = 0.02,
    min_pcorr: float = 0.25,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    store: StreamStore | None = None,
) -> tuple[CausalSieve | ScreenedSieve, CausalDigest]:
    """Run CausalSieve directly on a Pandas DataFrame, Polars DataFrame, or dict-of-lists.

    Zero hard external dependencies (100% duck-typed). Automatically skips
    timestamp and non-numeric columns, causally forward-fills isolated NaNs
    (zero look-ahead), and automatically switches to Tier-2 :class:`ScreenedSieve`
    when ``D >= screened_threshold`` (default ``128``).

    Returns ``(sieve, digest)``.
    """
    col_names, cols, _n_rows = _extract_tabular_columns(df)
    numeric_cols, _detected_ts = _select_columns(
        col_names,
        cols,
        timestamp_col=timestamp_col,
        columns=columns,
    )
    flat, n_rows = _pack_causal_ffill_matrix([cols[c] for c in numeric_cols])
    return _run_matrix_analysis(
        flat,
        n_rows,
        numeric_cols,
        lag=lag,
        forgetting=forgetting,
        top_k=top_k,
        ridge=ridge,
        precision_ridge=precision_ridge,
        shock_threshold=shock_threshold,
        e_alpha=e_alpha,
        warmup_ticks=warmup_ticks,
        refresh_interval=refresh_interval,
        nonlinear_hinge=nonlinear_hinge,
        raw_ring_capacity=raw_ring_capacity,
        normalize_scale=normalize_scale,
        screened_threshold=screened_threshold,
        use_screened=use_screened,
        active_m=active_m,
        lookback=lookback,
        max_edges=max_edges,
        min_granger_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        min_pcorr=min_pcorr,
        max_err=max_err,
        tokens_budget=tokens_budget,
        stream_id=stream_id,
        store=store,
    )


# =========================================================================
# 2. Prometheus HTTP API `/api/v1/query_range` Adapter
# =========================================================================


def _format_prometheus_metric_name(metric: Mapping[str, Any], idx: int) -> str:
    """Format a Prometheus metric label dict as ``metric_name{k1=v1,k2=v2}``."""
    base = str(metric.get("__name__", "")).strip()
    label_parts = [
        f"{k}={v}"
        for k, v in sorted(metric.items(), key=lambda kv: str(kv[0]))
        if str(k) != "__name__"
    ]
    if base and label_parts:
        return f"{base}{{{','.join(label_parts)}}}"
    if base:
        return base
    if label_parts:
        return f"{{{','.join(label_parts)}}}"
    return f"series_{idx}"


def analyze_prometheus(
    payload: Mapping[str, Any] | str | bytes,
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    top_k: int = 5,
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
    lookback: int = 0,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_partial_r2: float = 0.02,
    min_pcorr: float = 0.25,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    store: StreamStore | None = None,
) -> tuple[CausalSieve | ScreenedSieve, CausalDigest]:
    """Parse a Prometheus ``/api/v1/query_range`` JSON matrix payload and run CausalSieve.

    Accepts either a parsed JSON dict or raw JSON ``str`` / ``bytes`` matching
    Prometheus HTTP API format::

        {
          "status": "success",
          "data": {
            "resultType": "matrix",
            "result": [
              {"metric": {"__name__": "cpu", "service": "db"}, "values": [[1700000000, "0.42"], ...]}
            ]
          }
        }

    Timestamps are aligned across all series onto the sorted union timeline with
    strict causal forward-fill (never look-ahead imputed). Automatically uses
    Tier-2 :class:`ScreenedSieve` when ``D >= screened_threshold``.

    Returns ``(sieve, digest)``.
    """
    raw_bytes: int | None = None
    parsed: Any
    if isinstance(payload, (bytes, bytearray)):
        raw_bytes = len(payload)
        parsed = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        raw_bytes = len(payload.encode("utf-8"))
        parsed = json.loads(payload)
    elif isinstance(payload, Mapping):
        parsed = payload
    else:
        raise TypeError(
            f"Expected Prometheus JSON mapping, str, or bytes; got {type(payload).__name__}"
        )

    if not isinstance(parsed, Mapping):
        raise ValueError("Prometheus payload must decode to a JSON object")

    status = parsed.get("status")
    if status is not None and status != "success":
        err_type = parsed.get("errorType", "")
        err_msg = parsed.get("error", "query failed")
        raise ValueError(f"Prometheus response status={status!r} ({err_type}): {err_msg}")

    data_obj = parsed.get("data", parsed)
    if not isinstance(data_obj, Mapping):
        raise ValueError("Prometheus payload 'data' field must be an object")

    result_type = data_obj.get("resultType")
    if result_type is not None and result_type != "matrix":
        raise ValueError(
            f"Expected Prometheus query_range resultType='matrix', got {result_type!r}"
        )

    series_list = data_obj.get("result")
    if not isinstance(series_list, Sequence) or isinstance(series_list, (str, bytes)):
        raise ValueError("Prometheus payload missing 'data.result' list of series")
    if not series_list:
        raise ValueError("Prometheus payload contains zero series")

    all_timestamps: set[float] = set()
    parsed_series: list[tuple[str, dict[float, float]]] = []
    name_counts: dict[str, int] = {}

    for idx, item in enumerate(series_list):
        if not isinstance(item, Mapping):
            raise ValueError(f"Prometheus series at index {idx} must be an object")
        metric_map = item.get("metric")
        metric_dict: Mapping[str, Any] = metric_map if isinstance(metric_map, Mapping) else {}
        base_name = _format_prometheus_metric_name(metric_dict, idx)
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        chan_name = base_name if count == 1 else f"{base_name}#{count}"

        values = item.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue

        ts_to_val: dict[float, float] = {}
        for pair in values:
            if not isinstance(pair, Sequence) or len(pair) < 2:
                continue
            ts = _parse_prom_float(pair[0])
            if ts is None:
                continue
            all_timestamps.add(ts)
            val = _parse_prom_float(pair[1])
            if val is not None:
                ts_to_val[ts] = val

        if ts_to_val:
            parsed_series.append((chan_name, ts_to_val))

    if not parsed_series or not all_timestamps:
        raise ValueError("Prometheus payload contains no finite numeric time-series observations")

    timeline = sorted(all_timestamps)
    channel_names = [name for name, _ in parsed_series]
    dim = len(channel_names)
    n_rows = len(timeline)

    flat = array.array("d")
    last_valid = [0.0] * dim
    for ts in timeline:
        for j in range(dim):
            sample = parsed_series[j][1].get(ts)
            if sample is not None:
                last_valid[j] = sample
                flat.append(sample)
            else:
                flat.append(last_valid[j])

    return _run_matrix_analysis(
        flat,
        n_rows,
        channel_names,
        lag=lag,
        forgetting=forgetting,
        top_k=top_k,
        ridge=ridge,
        precision_ridge=precision_ridge,
        shock_threshold=shock_threshold,
        e_alpha=e_alpha,
        warmup_ticks=warmup_ticks,
        refresh_interval=refresh_interval,
        nonlinear_hinge=nonlinear_hinge,
        raw_ring_capacity=raw_ring_capacity,
        normalize_scale=normalize_scale,
        screened_threshold=screened_threshold,
        use_screened=use_screened,
        active_m=active_m,
        lookback=lookback,
        max_edges=max_edges,
        min_granger_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        min_pcorr=min_pcorr,
        max_err=max_err,
        tokens_budget=tokens_budget,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        store=store,
    )


# =========================================================================
# 3. Drop-In RCAEval Root-Cause Localization Adapter
# =========================================================================


def _default_service_parser(metric_col: str) -> str:
    """Parse ``<service>_<metric>`` into ``<service>`` (standard RCAEval schema)."""
    return metric_col.split("_", 1)[0]


def _resolve_inject_index(
    timestamps: list[Any] | None,
    inject_time: Any,
    n_rows: int,
) -> int | None:
    """Find the first chronological index where ``timestamp >= inject_time``."""
    if inject_time is None or n_rows <= 0:
        return None

    if timestamps is None:
        if isinstance(inject_time, (int, float)) and not isinstance(inject_time, bool):
            idx = int(inject_time)
            if 0 <= idx < n_rows:
                return idx
        return None

    inj_num = _parse_prom_float(inject_time)
    if inj_num is not None:
        for idx, ts in enumerate(timestamps):
            ts_num = _parse_prom_float(ts)
            if ts_num is not None and ts_num >= inj_num:
                return idx
        return None

    # Fallback direct ordering comparison (e.g., datetime / Pandas Timestamp / ISO strings)
    for idx, ts in enumerate(timestamps):
        if ts is None:
            continue
        try:
            if ts >= inject_time:
                return idx
        except TypeError:
            if str(ts) >= str(inject_time):
                return idx
    return None


def _sort_row_order_by_timestamps(timestamps: list[Any]) -> list[int]:
    """Stable chronological sort indices for numeric or comparable timestamp lists."""
    n = len(timestamps)
    numeric_ts = [_parse_prom_float(t) for t in timestamps]
    if all(v is not None for v in numeric_ts):
        vals = [v if v is not None else 0.0 for v in numeric_ts]
        return sorted(range(n), key=lambda i: vals[i])
    try:
        return sorted(range(n), key=lambda i: timestamps[i])
    except TypeError:
        return sorted(range(n), key=lambda i: str(timestamps[i]))


def _aggregate_to_services(
    channel_scores: Sequence[float],
    services: Sequence[str],
    *,
    onset_ticks: Sequence[int] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Aggregate metric root scores to service level via max-over-metrics.

    Tie-breaks deterministically by earliest valid onset tick, then service name.
    """
    svc_best: dict[str, float] = {}
    svc_onset: dict[str, int] = {}
    for ch, svc in enumerate(services):
        sc = float(channel_scores[ch])
        if not math.isfinite(sc):
            sc = 0.0
        onset = int(onset_ticks[ch]) if onset_ticks is not None else -1
        eff_onset = onset if onset >= 0 else 10**9
        if svc not in svc_best or sc > svc_best[svc] or (
            sc == svc_best[svc] and eff_onset < svc_onset.get(svc, 10**9)
        ):
            svc_best[svc] = sc
            svc_onset[svc] = eff_onset

    ranked = sorted(
        svc_best.keys(),
        key=lambda s: (-svc_best[s], svc_onset.get(s, 10**9), s),
    )
    return ranked, {svc: svc_best[svc] for svc in ranked}


def rcaeval_rca(
    data: Any,
    inject_time: Any = None,
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    top_k: int = 5,
    timestamp_col: str | None = None,
    service_parser: Callable[[str], str] | None = None,
    ridge: float = 1e-2,
    precision_ridge: float = 1e-2,
    shock_threshold: float = 4.0,
    e_alpha: float = 0.01,
    refresh_interval: int = 0,
    nonlinear_hinge: bool = False,
    normalize_scale: bool = True,
    screened_threshold: int = 128,
    use_screened: bool | None = None,
    active_m: int = 32,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    store: StreamStore | None = None,
) -> dict[str, Any]:
    """Drop-in ``RCAEval`` microservice root-cause localization adapter.

    Accepts a Pandas DataFrame, Polars DataFrame, or dict-of-lists with an
    optional ``"time"`` column and ``"<service>_<metric>"`` columns.

    - When ``inject_time`` is provided, uses the symmetric baseline/incident
      split (``warmup_ticks = inject_index``, ``lookback = T - inject_index``).
    - When ``inject_time is None``, runs in **blind mode** (``warmup_ticks = 0``,
      ``lookback = 0`` autonomous onset detection).

    Returns::

        {
            "ranks": [svc1, svc2, ...],
            "scores": {svc1: score1, ...},
            "metric_top_k": [...],
            "digest": CausalDigest,
        }
    """
    col_names, cols, n_rows = _extract_tabular_columns(data)
    ts_col_arg = timestamp_col if timestamp_col is not None else ("time" if "time" in cols else None)
    metric_cols, detected_ts = _select_columns(col_names, cols, timestamp_col=ts_col_arg)

    row_order: list[int] | None = None
    sorted_timestamps: list[Any] | None = None
    if detected_ts is not None:
        raw_ts = cols[detected_ts]
        row_order = _sort_row_order_by_timestamps(raw_ts)
        sorted_timestamps = [raw_ts[i] for i in row_order]

    flat, n_valid = _pack_causal_ffill_matrix([cols[c] for c in metric_cols], row_order=row_order)

    inject_idx = _resolve_inject_index(sorted_timestamps, inject_time, n_valid)
    if inject_idx is not None and 0 < inject_idx < n_valid:
        warmup_ticks = inject_idx
        lookback = max(1, n_valid - inject_idx)
    elif inject_idx == 0:
        warmup_ticks = 0
        lookback = n_valid
    else:
        warmup_ticks = 0
        lookback = 0

    parser = service_parser if service_parser is not None else _default_service_parser
    services = [parser(c) for c in metric_cols]

    dim = len(metric_cols)
    should_screen = (
        use_screened if use_screened is not None else (dim >= int(screened_threshold))
    )

    if should_screen:
        eff_m = min(int(active_m), dim)
        screened = ScreenedSieve(
            dim=dim,
            active_m=eff_m,
            lag=lag,
            channel_names=metric_cols,
            forgetting=forgetting,
            ridge=ridge,
        )
        batch_res = screened.push_batch(flat, top_k=dim, return_all_z=False)
        ch_scores = [0.0] * dim
        for ch, sc in zip(batch_res.top_channels, batch_res.top_scores, strict=False):
            ch_scores[ch] = float(sc)
        ranked_services, svc_scores = _aggregate_to_services(ch_scores, services)
        eff_k = max(1, min(int(top_k), len(batch_res.top_channels)))
        metric_top_k: list[dict[str, Any]] = [
            {
                "rank": idx + 1,
                "channel": batch_res.top_channels[idx],
                "name": batch_res.top_names[idx],
                "service": services[batch_res.top_channels[idx]],
                "root_score": float(batch_res.top_scores[idx]),
                "peak_exo_z": float(batch_res.top_scores[idx]),
                "raw_z": float(batch_res.top_scores[idx]),
                "onset_tick": -1,
            }
            for idx in range(eff_k)
        ]
        digest = _build_screened_digest(
            screened,
            batch_res,
            top_k=top_k,
            stream_id=stream_id,
        )
        screened.close()
        return {
            "ranks": ranked_services,
            "scores": svc_scores,
            "metric_top_k": metric_top_k,
            "digest": digest,
        }

    ring_cap = max(8192, n_valid + 16)
    sieve = CausalSieve(
        dim=dim,
        lag=lag,
        channel_names=metric_cols,
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
    attr = sieve.root_attribution(lookback=lookback)
    ranked_services, svc_scores = _aggregate_to_services(
        attr.root_scores,
        services,
        onset_ticks=attr.onset_tick,
    )
    eff_k = max(1, min(int(top_k), len(attr.items)))
    metric_top_k = [
        {
            **item.to_dict(),
            "service": services[item.channel],
        }
        for item in attr.items[:eff_k]
    ]
    digest = build_digest(
        sieve,
        top_k=top_k,
        lookback=lookback,
        max_err=max_err,
        tokens_budget=tokens_budget,
        stream_id=stream_id,
        store=store,
    )
    return {
        "ranks": ranked_services,
        "scores": svc_scores,
        "metric_top_k": metric_top_k,
        "digest": digest,
    }


# =========================================================================
# 4. OpenTelemetry OTLP/JSON Metrics Adapter
# =========================================================================


def _otlp_attr_value(raw_val: Any) -> str:
    """Extract a concise string representation from an OTLP ``AnyValue`` object."""
    if isinstance(raw_val, Mapping):
        for key in (
            "stringValue",
            "string_value",
            "intValue",
            "int_value",
            "doubleValue",
            "double_value",
            "boolValue",
            "bool_value",
            "bytesValue",
            "bytes_value",
        ):
            if key in raw_val:
                return str(raw_val[key])
        return json.dumps(dict(raw_val), sort_keys=True, separators=(",", ":"))
    return str(raw_val)


def _otlp_extract_attrs(raw_attrs: Any) -> dict[str, str]:
    """Convert an OTLP attributes array ``[{"key": ..., "value": ...}]`` into a flat dict."""
    out: dict[str, str] = {}
    if isinstance(raw_attrs, Mapping):
        for k, v in raw_attrs.items():
            key = str(k).strip()
            if key:
                out[key] = _otlp_attr_value(v)
        return out
    if not isinstance(raw_attrs, Sequence) or isinstance(raw_attrs, (str, bytes, bytearray)):
        return out
    for item in raw_attrs:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        out[key] = _otlp_attr_value(item.get("value", ""))
    return out


def _parse_otlp_timestamp(dp: Mapping[str, Any]) -> int | float | None:
    """Parse OTLP ``timeUnixNano`` (preserving exact 64-bit integer precision)."""
    raw_ts: Any = None
    for key in ("timeUnixNano", "time_unix_nano", "startTimeUnixNano", "start_time_unix_nano"):
        if key in dp and dp[key] is not None:
            raw_ts = dp[key]
            break
    if raw_ts is None or isinstance(raw_ts, bool):
        return None
    if isinstance(raw_ts, int):
        return raw_ts
    if isinstance(raw_ts, float):
        return raw_ts if math.isfinite(raw_ts) else None
    if isinstance(raw_ts, (str, bytes, bytearray)):
        s = (
            raw_ts.decode("utf-8", errors="ignore").strip()
            if not isinstance(raw_ts, str)
            else raw_ts.strip()
        )
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
            except ValueError:
                return None
            return f if math.isfinite(f) else None
    return None


def _extract_otlp_datapoint_value(dp: Mapping[str, Any], kind: str) -> float | None:
    """Extract a finite numeric scalar from an OTLP gauge, sum, or histogram datapoint."""
    if kind in ("gauge", "sum"):
        for key in ("asDouble", "as_double", "asInt", "as_int"):
            if key in dp and dp[key] is not None:
                return _parse_prom_float(dp[key])
        return None

    if kind in ("histogram", "exponentialHistogram", "exponential_histogram"):
        sum_val = _parse_prom_float(dp.get("sum")) if "sum" in dp else None
        cnt_val = _parse_prom_float(dp.get("count")) if "count" in dp else None
        if sum_val is not None and cnt_val is not None and cnt_val > 0.0:
            ratio = sum_val / cnt_val
            return ratio if math.isfinite(ratio) else None
        if sum_val is not None:
            return sum_val
        if cnt_val is not None:
            return cnt_val
        return None

    return None


def analyze_otlp_metrics(
    payload: Mapping[str, Any] | str | bytes,
    *,
    lag: int = 2,
    forgetting: float = 0.995,
    warmup_ticks: int = 0,
    top_k: int = 5,
    token_budget: int | None = 400,
    ridge: float = 1e-2,
    precision_ridge: float = 1e-2,
    shock_threshold: float = 4.0,
    e_alpha: float = 0.01,
    refresh_interval: int = 0,
    nonlinear_hinge: bool = False,
    raw_ring_capacity: int = 0,
    normalize_scale: bool = True,
    screened_threshold: int = 128,
    use_screened: bool | None = None,
    active_m: int = 32,
    lookback: int = 0,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_partial_r2: float = 0.02,
    min_pcorr: float = 0.25,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    store: StreamStore | None = None,
) -> tuple[CausalSieve | ScreenedSieve, CausalDigest]:
    """Parse an OpenTelemetry OTLP/JSON ``ExportMetricsServiceRequest`` and run CausalSieve.

    Supports ``resourceMetrics[].scopeMetrics[].metrics[]`` containing ``gauge``,
    ``sum``, ``histogram`` (``sum/count`` when ``count > 0``, else ``sum``), or
    ``exponentialHistogram`` datapoints with ``timeUnixNano`` and
    ``asDouble``/``asInt``.

    Channels are named ``<service.name>_<metric.name>`` (using ``service.name``
    from ``resource.attributes``), appending sorted datapoint attributes
    ``{k=v,...}`` only when needed to disambiguate duplicate series. Timestamps
    across series are aligned onto the sorted union timeline using strict causal
    forward-fill (zero look-ahead).

    Returns ``(sieve, digest)``.
    """
    raw_bytes: int | None = None
    parsed: Any
    if isinstance(payload, (bytes, bytearray)):
        raw_bytes = len(payload)
        parsed = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        raw_bytes = len(payload.encode("utf-8"))
        parsed = json.loads(payload)
    elif isinstance(payload, Mapping):
        parsed = payload
    else:
        raise TypeError(
            f"Expected OTLP JSON mapping, str, or bytes; got {type(payload).__name__}"
        )

    if not isinstance(parsed, Mapping):
        raise ValueError("OTLP metrics payload must decode to a JSON object")

    resource_metrics = parsed.get("resourceMetrics")
    if resource_metrics is None:
        resource_metrics = parsed.get("resource_metrics")
    if not isinstance(resource_metrics, Sequence) or isinstance(
        resource_metrics, (str, bytes, bytearray)
    ):
        raise ValueError("OTLP payload missing 'resourceMetrics' list")
    if not resource_metrics:
        raise ValueError("OTLP payload contains zero resourceMetrics entries")

    all_timestamps: set[int | float] = set()
    # Each entry: (base_name, dp_attrs_tuple, extra_res_attrs_tuple, ts_to_val)
    raw_series: list[
        tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], dict[int | float, float]]
    ] = []
    metric_counter = 0

    for rm in resource_metrics:
        if not isinstance(rm, Mapping):
            continue
        resource = rm.get("resource")
        res_attrs = (
            _otlp_extract_attrs(resource.get("attributes"))
            if isinstance(resource, Mapping)
            else {}
        )
        service_name = res_attrs.get("service.name", "").strip()
        extra_res_attrs = tuple(
            sorted((k, v) for k, v in res_attrs.items() if k != "service.name")
        )

        scope_list = (
            rm.get("scopeMetrics")
            or rm.get("scope_metrics")
            or rm.get("instrumentationLibraryMetrics")
            or ()
        )
        if not isinstance(scope_list, Sequence) or isinstance(
            scope_list, (str, bytes, bytearray)
        ):
            continue

        for sm in scope_list:
            if not isinstance(sm, Mapping):
                continue
            metrics_list = sm.get("metrics") or ()
            if not isinstance(metrics_list, Sequence) or isinstance(
                metrics_list, (str, bytes, bytearray)
            ):
                continue

            for metric in metrics_list:
                if not isinstance(metric, Mapping):
                    continue
                metric_name = str(metric.get("name", "")).strip()
                if service_name and metric_name:
                    base_name = f"{service_name}_{metric_name}"
                elif metric_name:
                    base_name = metric_name
                elif service_name:
                    base_name = f"{service_name}_metric_{metric_counter}"
                else:
                    base_name = f"metric_{metric_counter}"
                metric_counter += 1

                for kind in (
                    "gauge",
                    "sum",
                    "histogram",
                    "exponentialHistogram",
                    "exponential_histogram",
                ):
                    container = metric.get(kind)
                    if not isinstance(container, Mapping):
                        continue
                    data_points = container.get("dataPoints")
                    if data_points is None:
                        data_points = container.get("data_points")
                    if not isinstance(data_points, Sequence) or isinstance(
                        data_points, (str, bytes, bytearray)
                    ):
                        continue

                    grouped: dict[tuple[tuple[str, str], ...], dict[int | float, float]] = {}
                    group_order: list[tuple[tuple[str, str], ...]] = []
                    for dp in data_points:
                        if not isinstance(dp, Mapping):
                            continue
                        ts = _parse_otlp_timestamp(dp)
                        if ts is None:
                            continue
                        all_timestamps.add(ts)
                        dp_attrs = tuple(sorted(_otlp_extract_attrs(dp.get("attributes")).items()))
                        val = _extract_otlp_datapoint_value(dp, kind)
                        if dp_attrs not in grouped:
                            grouped[dp_attrs] = {}
                            group_order.append(dp_attrs)
                        if val is not None:
                            grouped[dp_attrs][ts] = val

                    for dp_attrs in group_order:
                        ts_to_val = grouped[dp_attrs]
                        if ts_to_val:
                            raw_series.append((base_name, dp_attrs, extra_res_attrs, ts_to_val))

    if not raw_series or not all_timestamps:
        raise ValueError("OTLP payload contains no finite numeric metric datapoints")

    base_counts: dict[str, int] = {}
    for base_name, _, _, _ in raw_series:
        base_counts[base_name] = base_counts.get(base_name, 0) + 1

    channel_names: list[str] = []
    seen_names: dict[str, int] = {}
    for base_name, dp_attrs, extra_res_attrs, _ in raw_series:
        if base_counts[base_name] == 1:
            candidate = base_name
        elif dp_attrs:
            attr_str = ",".join(f"{k}={v}" for k, v in dp_attrs)
            candidate = f"{base_name}{{{attr_str}}}"
        elif extra_res_attrs:
            attr_str = ",".join(f"{k}={v}" for k, v in extra_res_attrs)
            candidate = f"{base_name}{{{attr_str}}}"
        else:
            candidate = base_name

        occ = seen_names.get(candidate, 0) + 1
        seen_names[candidate] = occ
        channel_names.append(candidate if occ == 1 else f"{candidate}#{occ}")

    timeline = sorted(all_timestamps)
    dim = len(channel_names)
    n_rows = len(timeline)

    flat = array.array("d")
    last_valid = [0.0] * dim
    for ts in timeline:
        for j in range(dim):
            sample = raw_series[j][3].get(ts)
            if sample is not None:
                last_valid[j] = sample
                flat.append(sample)
            else:
                flat.append(last_valid[j])

    eff_budget = tokens_budget if tokens_budget is not None else token_budget
    return _run_matrix_analysis(
        flat,
        n_rows,
        channel_names,
        lag=lag,
        forgetting=forgetting,
        top_k=top_k,
        ridge=ridge,
        precision_ridge=precision_ridge,
        shock_threshold=shock_threshold,
        e_alpha=e_alpha,
        warmup_ticks=warmup_ticks,
        refresh_interval=refresh_interval,
        nonlinear_hinge=nonlinear_hinge,
        raw_ring_capacity=raw_ring_capacity,
        normalize_scale=normalize_scale,
        screened_threshold=screened_threshold,
        use_screened=use_screened,
        active_m=active_m,
        lookback=lookback,
        max_edges=max_edges,
        min_granger_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        min_pcorr=min_pcorr,
        max_err=max_err,
        tokens_budget=eff_budget,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        store=store,
    )


# =========================================================================
# 5. NDJSON / JSONL Metric Stream Adapter
# =========================================================================


def _iter_jsonl_records(
    source: str | Path | bytes | bytearray | Iterable[str | Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int | None]:
    """Load JSONL records from a file path, raw JSONL text, or iterable of lines/dicts."""
    raw_bytes: int | None = None
    items: Iterable[str | bytes | bytearray | Mapping[str, Any]]

    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"JSONL file not found: {source}")
        raw_bytes = source.stat().st_size
        text = source.read_text(encoding="utf-8")
        items = text.splitlines()
    elif isinstance(source, (bytes, bytearray)):
        raw_bytes = len(source)
        items = source.decode("utf-8").splitlines()
    elif isinstance(source, str):
        stripped = source.lstrip()
        if not stripped:
            raise ValueError("JSONL source contains zero records")
        if "\n" in source or "\r" in source or stripped.startswith("{"):
            raw_bytes = len(source.encode("utf-8"))
            items = source.splitlines()
        else:
            candidate = Path(source)
            if not candidate.is_file():
                raise FileNotFoundError(f"JSONL file not found: {source!r}")
            raw_bytes = candidate.stat().st_size
            items = candidate.read_text(encoding="utf-8").splitlines()
    elif isinstance(source, Mapping):
        raise TypeError(
            "Expected a file path, JSONL string, or iterable of JSON lines/dicts; got a single Mapping"
        )
    elif isinstance(source, Iterable):
        items = source
    else:
        raise TypeError(
            f"Expected a file path, JSONL string, or iterable of JSON lines/dicts; got {type(source).__name__}"
        )

    records: list[Mapping[str, Any]] = []
    byte_total = 0
    for idx, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            records.append(item)
            continue
        if isinstance(item, (str, bytes, bytearray)):
            line = item.decode("utf-8") if not isinstance(item, str) else item
            s = line.strip()
            if not s:
                continue
            byte_total += len(line.encode("utf-8")) + 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {idx}: {exc.msg}") from exc
            if not isinstance(obj, Mapping):
                raise ValueError(
                    f"JSONL line {idx} must decode to a JSON object, got {type(obj).__name__}"
                )
            records.append(obj)
            continue
        raise TypeError(
            f"JSONL item {idx} must be a JSON string or dict mapping, got {type(item).__name__}"
        )

    if not records:
        raise ValueError("JSONL source contains zero records")
    if raw_bytes is None and byte_total > 0:
        raw_bytes = byte_total
    return records, raw_bytes


def analyze_jsonl(
    source: str | Path | bytes | bytearray | Iterable[str | Mapping[str, Any]],
    *,
    time_col: str | None = None,
    lag: int = 2,
    forgetting: float = 0.995,
    warmup_ticks: int = 0,
    top_k: int = 5,
    token_budget: int | None = 400,
    columns: Sequence[str] | None = None,
    ridge: float = 1e-2,
    precision_ridge: float = 1e-2,
    shock_threshold: float = 4.0,
    e_alpha: float = 0.01,
    refresh_interval: int = 0,
    nonlinear_hinge: bool = False,
    raw_ring_capacity: int = 0,
    normalize_scale: bool = True,
    screened_threshold: int = 128,
    use_screened: bool | None = None,
    active_m: int = 32,
    lookback: int = 0,
    max_edges: int = 8,
    min_granger_f: float = 3.0,
    min_partial_r2: float = 0.02,
    min_pcorr: float = 0.25,
    max_err: float = 0.05,
    tokens_budget: int | None = None,
    stream_id: str | None = None,
    store: StreamStore | None = None,
) -> tuple[CausalSieve | ScreenedSieve, CausalDigest]:
    """Parse an NDJSON / JSONL metric stream and run CausalSieve.

    Accepts a file path (``str`` or ``Path``), a raw multiline JSONL string or
    bytes payload, or any iterable of JSON strings / record dicts. Automatically
    discovers numeric metric columns (excluding timestamps, booleans, strings,
    and nested objects), enforces finiteness via strict causal forward-fill with
    zero look-ahead, and switches to Tier-2 :class:`ScreenedSieve` when
    ``D >= screened_threshold``.

    Returns ``(sieve, digest)``.
    """
    records, raw_bytes = _iter_jsonl_records(source)

    col_names: list[str] = []
    seen_cols: set[str] = set()
    for rec in records:
        for raw_key in rec:
            name = str(raw_key)
            if name not in seen_cols:
                seen_cols.add(name)
                col_names.append(name)

    if not col_names:
        raise ValueError("JSONL records contain zero fields")

    cols: dict[str, list[Any]] = {name: [] for name in col_names}
    for rec in records:
        for name in col_names:
            cols[name].append(rec.get(name, None))

    detected_ts = time_col
    if detected_ts is None and columns is None:
        # Also auto-skip timestamp columns placed after metric keys in 2-column JSONL objects.
        for name in col_names:
            if name.strip().lower() in _TIMESTAMP_HEADER_NAMES:
                vals = cols[name]
                if not _column_is_numeric(vals):
                    detected_ts = name
                    break
                sample_floats = [
                    f for f in (_coerce_finite_float(v) for v in vals[:32]) if f is not None
                ]
                has_other_num = any(
                    other != name and _column_is_numeric(cols[other]) for other in col_names
                )
                if has_other_num and _is_monotonic_nondecreasing(sample_floats):
                    detected_ts = name
                    break

    numeric_cols, _ = _select_columns(
        col_names,
        cols,
        timestamp_col=detected_ts,
        columns=columns,
    )
    flat, n_rows = _pack_causal_ffill_matrix([cols[c] for c in numeric_cols])

    eff_budget = tokens_budget if tokens_budget is not None else token_budget
    return _run_matrix_analysis(
        flat,
        n_rows,
        numeric_cols,
        lag=lag,
        forgetting=forgetting,
        top_k=top_k,
        ridge=ridge,
        precision_ridge=precision_ridge,
        shock_threshold=shock_threshold,
        e_alpha=e_alpha,
        warmup_ticks=warmup_ticks,
        refresh_interval=refresh_interval,
        nonlinear_hinge=nonlinear_hinge,
        raw_ring_capacity=raw_ring_capacity,
        normalize_scale=normalize_scale,
        screened_threshold=screened_threshold,
        use_screened=use_screened,
        active_m=active_m,
        lookback=lookback,
        max_edges=max_edges,
        min_granger_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        min_pcorr=min_pcorr,
        max_err=max_err,
        tokens_budget=eff_budget,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        store=store,
    )

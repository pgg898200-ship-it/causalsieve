"""Unit and integration tests for CausalSieve ecosystem adapters and live streaming CLI.

Covers:
  1. ``analyze_dataframe`` on real ``pandas.DataFrame``, dict-of-lists, and Polars-like
     duck-typed tables (auto-skipping timestamp/string/bool columns, strict causal
     forward-fill NaN/Inf handling with zero look-ahead, Tier-1 causal root-cause
     localization, and Tier-2 automatic screening switch).
  2. ``analyze_prometheus`` on realistic Prometheus ``/api/v1/query_range`` JSON matrix
     payloads (dict/str/bytes, string floats, ``"NaN"``/``"+Inf"`` handling, misaligned
     timestamps aligned onto sorted union timeline with strict causal ffill, duplicate
     label disambiguation, and root-cause localization).
  3. ``rcaeval_rca`` drop-in contract on a real cached RCAEval Kubernetes DataFrame
     (symmetric ``inject_time`` mode, blind ``inject_time=None`` mode, out-of-order
     timestamp sorting invariance, custom ``service_parser``, and Tier-2 screened mode).
  4. ``analyze_otlp_metrics`` on OpenTelemetry OTLP/JSON ``ExportMetricsServiceRequest``
     payloads (gauge, sum, histogram ``sum/count`` and ``sum``-only fallback,
     ``<service.name>_<metric.name>`` naming, duplicate datapoint attribute
     disambiguation, missing-tick causal forward-fill, and root-cause localization).
  5. ``analyze_jsonl`` on NDJSON/JSONL files, raw strings, and iterables of lines/dicts
     (auto-skipping timestamps/strings/bools/objects, explicit ``time_col``, finite
     causal forward-fill, and exact parity).
  6. ``causalsieve watch`` CLI subprocess streaming live CSV and JSONL via stdin and
     files (real-time pre-EOF JSONL alerts, ``--every`` status events, Markdown format,
     and final EOF summary).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import causalsieve  # noqa: E402
from causalsieve import (  # noqa: E402
    CausalDigest,
    CausalSieve,
    ScreenedSieve,
    StreamStore,
    analyze_dataframe,
    analyze_jsonl,
    analyze_otlp_metrics,
    analyze_prometheus,
    rcaeval_rca,
    recover_handle,
)


def _make_amplified_cascade(
    T: int = 700,
    D: int = 8,
    shock_start: int = 560,
    seed: int = 202601,
) -> np.ndarray:
    """Generate a 3-hop amplified causal cascade (ch0 -> ch1 -> ch2) + noisy distractors.

    By construction, downstream leaf ch2 has larger raw Z-score than root ch0,
    so univariate scorers pick ch2 while CausalSieve localizes ch0.
    """
    rng = np.random.default_rng(seed)
    burn = 80
    X = np.zeros((T + burn, D), dtype=np.float64)
    eps = rng.standard_normal((T + burn, D))
    eps[:, 1] *= 0.35
    eps[:, 2] *= 0.30
    for t in range(2, T + burn):
        X[t] = 0.25 * X[t - 1] - 0.06 * X[t - 2] + eps[t]
        u = 5.5 if (t - burn) >= shock_start else 0.0
        X[t, 0] = 0.25 * X[t - 1, 0] + eps[t, 0] + u
        X[t, 1] = 0.65 * X[t - 1, 1] + 0.70 * X[t - 1, 0] + eps[t, 1]
        X[t, 2] = 0.70 * X[t - 1, 2] + 0.72 * X[t - 1, 1] + eps[t, 2]
    return np.ascontiguousarray(X[burn:])


# =========================================================================
# 1. analyze_dataframe tests
# =========================================================================


def test_analyze_dataframe_pandas_and_dict_parity_and_causal_ffill() -> None:
    """Verify pandas.DataFrame + dict-of-lists parity, auto-skip, strict causal ffill, and RCA."""
    X = _make_amplified_cascade(T=700, D=6, shock_start=560, seed=101)
    metric_names = ["db_cpu", "api_lat", "gw_lat", "cache_hit", "worker_q", "bg_io"]
    T = X.shape[0]

    # Inject leading NaN, isolated interior NaNs, and +Inf to verify strict causal forward-fill.
    X_dirty = X.copy()
    X_dirty[0, 0] = np.nan
    X_dirty[1, 0] = np.nan
    X_dirty[120, 1] = np.nan
    X_dirty[121, 1] = np.inf
    X_dirty[300, 2] = -np.inf

    expected_ch0_0_2 = [0.0, 0.0, float(X[2, 0])]
    expected_ch1_119_122 = [
        float(X[119, 1]),
        float(X[119, 1]),
        float(X[119, 1]),
        float(X[122, 1]),
    ]
    expected_ch2_299_301 = [float(X[299, 2]), float(X[299, 2]), float(X[301, 2])]

    timestamps = [1_700_000_000 + i for i in range(T)]
    dict_payload: dict[str, Any] = {
        "timestamp": timestamps,
        "region": ["us-east-1"] * T,
        "is_canary": [bool(i % 2) for i in range(T)],
        "pod_label": [f"pod-{i % 4}" for i in range(T)],
    }
    for j, name in enumerate(metric_names):
        col_list = X_dirty[:, j].tolist()
        if j == 3:
            col_list[50] = None  # Python None null in dict-of-lists
        dict_payload[name] = col_list

    df = pd.DataFrame(dict_payload)
    # Add a Pandas datetime column too; must be auto-skipped.
    df.insert(1, "created_at", pd.date_range("2026-01-01", periods=T, freq="1s"))

    sieve_pd, digest_pd = analyze_dataframe(df, lag=2, top_k=5)
    try:
        assert isinstance(sieve_pd, CausalSieve)
        assert isinstance(digest_pd, CausalDigest)
        assert sieve_pd.channel_names == metric_names
        assert sieve_pd.ticks == T
        assert digest_pd.dim == len(metric_names)
        assert digest_pd.alarm is True
        assert digest_pd.root_causes[0].name == "db_cpu"
        assert digest_pd.reduction_pct >= 90.0

        # Strictly causal imputation check: leading NaNs -> 0.0 (never backward-filled),
        # interior NaN/Inf -> prior tick value (never next tick value).
        sl0 = sieve_pd.get_raw_slice("db_cpu", 0, 3)
        assert sl0.values == pytest.approx(expected_ch0_0_2, abs=0.0)
        sl1 = sieve_pd.get_raw_slice("api_lat", 119, 123)
        assert sl1.values == pytest.approx(expected_ch1_119_122, abs=0.0)
        sl2 = sieve_pd.get_raw_slice("gw_lat", 299, 302)
        assert sl2.values == pytest.approx(expected_ch2_299_301, abs=0.0)
        sl3 = sieve_pd.get_raw_slice("cache_hit", 49, 52)
        assert sl3.values == pytest.approx(
            [float(X[49, 3]), float(X[49, 3]), float(X[51, 3])], abs=0.0
        )
    finally:
        sieve_pd.close()

    # Exact parity on plain dict-of-lists (zero pandas dependency path).
    sieve_dict, digest_dict = analyze_dataframe(dict_payload, lag=2, top_k=5)
    try:
        assert isinstance(sieve_dict, CausalSieve)
        assert sieve_dict.channel_names == metric_names
        assert [r.name for r in digest_dict.root_causes] == [
            r.name for r in digest_pd.root_causes
        ]
        assert digest_dict.root_causes[0].root_score == pytest.approx(
            digest_pd.root_causes[0].root_score, rel=1e-12, abs=1e-12
        )
    finally:
        sieve_dict.close()


def test_analyze_dataframe_polars_duck_typing_columns_and_screened_switch() -> None:
    """Verify duck-typed Polars to_dict(as_series=False), column subsetting, and Tier-2 switch."""
    X = _make_amplified_cascade(T=400, D=10, shock_start=320, seed=102)
    names = [f"m_{i}" for i in range(10)]

    class _FakePolarsDF:
        def __init__(self, cols: dict[str, list[Any]]) -> None:
            self._cols = cols
            self.columns = list(cols.keys())

        def to_dict(self, *, as_series: bool = True) -> dict[str, list[Any]]:
            if as_series:
                raise AssertionError("Expected as_series=False fast path")
            return self._cols

        def __getitem__(self, key: str) -> list[Any]:
            return self._cols[key]

    fake_pl = _FakePolarsDF(
        {
            "ts": list(range(400)),
            "service_tag": ["svc"] * 400,
            **{names[j]: X[:, j].tolist() for j in range(10)},
        }
    )

    # Subset of columns + explicit timestamp_col
    sieve_sub, dig_sub = analyze_dataframe(
        fake_pl,
        timestamp_col="ts",
        columns=["m_0", "m_1", "m_2", "m_3"],
        lag=2,
    )
    try:
        assert isinstance(sieve_sub, CausalSieve)
        assert sieve_sub.channel_names == ["m_0", "m_1", "m_2", "m_3"]
        assert dig_sub.root_causes[0].name == "m_0"
    finally:
        sieve_sub.close()

    # Automatic Tier-2 ScreenedSieve selection when D >= screened_threshold
    sieve_t2, dig_t2 = analyze_dataframe(
        fake_pl,
        screened_threshold=8,
        active_m=6,
        lag=2,
        top_k=5,
    )
    try:
        assert isinstance(sieve_t2, ScreenedSieve)
        assert isinstance(dig_t2, CausalDigest)
        assert sieve_t2.dim == 10
        assert sieve_t2.active_m == 6
        assert dig_t2.root_causes[0].name == "m_0"
        assert len(dig_t2.root_causes) == 5
    finally:
        sieve_t2.close()


def test_analyze_dataframe_validation_errors() -> None:
    """Verify clear errors on invalid tabular inputs."""
    with pytest.raises(TypeError):
        analyze_dataframe(12345)
    with pytest.raises(ValueError, match="zero columns"):
        analyze_dataframe({})
    with pytest.raises(ValueError, match="zero rows"):
        analyze_dataframe({"a": [], "b": []})
    with pytest.raises(ValueError, match="mismatched lengths"):
        analyze_dataframe({"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(ValueError, match="No numeric metric columns"):
        analyze_dataframe({"time": [1, 2, 3], "label": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="not numeric"):
        analyze_dataframe({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]}, columns=["s"])


# =========================================================================
# 2. analyze_prometheus tests
# =========================================================================


def test_analyze_prometheus_query_range_misaligned_nan_and_rca() -> None:
    """Test realistic Prometheus /api/v1/query_range JSON matrix with string floats, NaNs, and misalignment."""
    T = 650
    shock_start = 520
    X = _make_amplified_cascade(T=T, D=6, shock_start=shock_start, seed=202)
    t0 = 1_710_000_000.0
    step = 15.0

    series_meta = [
        {"__name__": "db_cpu", "service": "db", "env": "prod"},
        {"__name__": "api_latency", "service": "api", "env": "prod"},
        {"__name__": "gateway_latency", "service": "gateway", "env": "prod"},
        {"__name__": "cache_ops", "service": "redis", "env": "prod"},
        {"__name__": "queue_depth", "service": "worker", "env": "prod"},
        {"__name__": "disk_iops", "service": "storage", "env": "prod"},
    ]

    result_entries: list[dict[str, Any]] = []
    for j, meta in enumerate(series_meta):
        values: list[list[Any]] = []
        for idx in range(T):
            ts = t0 + idx * step
            # Misalign timestamps realistically: drop scattered ticks on non-root series
            # and inject an extra intermediate timestamp on series 3.
            if j == 1 and idx in (10, 11, 200):
                continue
            if j == 2 and idx in (0, 1, 150):
                continue
            if j == 0 and idx == 80:
                values.append([ts, "NaN"])
                continue
            if j == 4 and idx == 120:
                values.append([ts, "+Inf"])
                continue
            if j == 5 and idx == 0:
                values.append([ts, "NaN"])
                continue
            values.append([ts, f"{X[idx, j]:.8f}"])
        if j == 3:
            # Extra off-grid timestamp between tick 40 and 41
            values.append([t0 + 40.5 * step, f"{X[40, j]:.8f}"])
            values.sort(key=lambda pair: pair[0])
        result_entries.append({"metric": meta, "values": values})

    payload_dict: dict[str, Any] = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": result_entries,
        },
    }
    payload_json = json.dumps(payload_dict)

    # Test dict payload
    sieve_d, digest_d = analyze_prometheus(payload_dict, lag=2, top_k=5)
    try:
        assert isinstance(sieve_d, CausalSieve)
        assert isinstance(digest_d, CausalDigest)
        # Union timeline has T + 1 timestamps because of the extra 40.5*step tick on series 3.
        assert sieve_d.ticks == T + 1
        assert sieve_d.dim == 6
        assert sieve_d.channel_names[0] == "db_cpu{env=prod,service=db}"
        assert sieve_d.channel_names[1] == "api_latency{env=prod,service=api}"
        assert sieve_d.channel_names[2] == "gateway_latency{env=prod,service=gateway}"

        # Root cause localization must identify db_cpu as #1 despite gateway_latency amplification.
        assert digest_d.alarm is True
        assert digest_d.root_causes[0].name == "db_cpu{env=prod,service=db}"
        assert digest_d.reduction_pct >= 90.0

        # Verify strict causal forward-fill at leading missing/NaN ticks (must be 0.0, never bfilled)
        gw_head = sieve_d.get_raw_slice("gateway_latency{env=prod,service=gateway}", 0, 3)
        assert gw_head.values[0] == 0.0
        assert gw_head.values[1] == 0.0
        assert gw_head.values[2] == pytest.approx(float(X[2, 2]), abs=1e-7)

        disk_head = sieve_d.get_raw_slice("disk_iops{env=prod,service=storage}", 0, 2)
        assert disk_head.values[0] == 0.0
        assert disk_head.values[1] == pytest.approx(float(X[1, 5]), abs=1e-7)

        # Verify interior "NaN" at tick 80 (shifted to index 81 due to 40.5 extra tick) forward-fills tick 79
        db_nan_win = sieve_d.get_raw_slice("db_cpu{env=prod,service=db}", 80, 83)
        assert db_nan_win.values[0] == pytest.approx(float(X[79, 0]), abs=1e-7)
        assert db_nan_win.values[1] == pytest.approx(float(X[79, 0]), abs=1e-7)
        assert db_nan_win.values[2] == pytest.approx(float(X[81, 0]), abs=1e-7)
    finally:
        sieve_d.close()

    # Test raw JSON str AND bytes payloads yield bit-identical results
    sieve_s, digest_s = analyze_prometheus(payload_json, lag=2, top_k=5)
    sieve_b, digest_b = analyze_prometheus(payload_json.encode("utf-8"), lag=2, top_k=5)
    try:
        assert [r.name for r in digest_s.root_causes] == [r.name for r in digest_d.root_causes]
        assert [r.name for r in digest_b.root_causes] == [r.name for r in digest_d.root_causes]
        assert digest_s.root_causes[0].root_score == pytest.approx(
            digest_d.root_causes[0].root_score, rel=1e-12
        )
    finally:
        sieve_s.close()
        sieve_b.close()


def test_analyze_prometheus_duplicate_labels_and_errors() -> None:
    """Verify duplicate metric label disambiguation and Prometheus error payload handling."""
    dup_payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "cpu"},
                    "values": [[1000 + i, str(float(i % 5))] for i in range(60)],
                },
                {
                    "metric": {"__name__": "cpu"},
                    "values": [[1000 + i, str(float((i + 1) % 7))] for i in range(60)],
                },
            ],
        },
    }
    sieve, _ = analyze_prometheus(dup_payload, lag=1)
    try:
        assert sieve.channel_names == ["cpu", "cpu#2"]
    finally:
        sieve.close()

    with pytest.raises(ValueError, match="status='error'"):
        analyze_prometheus({"status": "error", "errorType": "bad_data", "error": "parse error"})
    with pytest.raises(ValueError, match="resultType='matrix'"):
        analyze_prometheus({"status": "success", "data": {"resultType": "vector", "result": []}})
    with pytest.raises(ValueError, match="zero series"):
        analyze_prometheus({"status": "success", "data": {"resultType": "matrix", "result": []}})
    with pytest.raises(TypeError):
        analyze_prometheus([1, 2, 3])  # type: ignore[arg-type]


# =========================================================================
# 3. rcaeval_rca drop-in contract tests (real cached RCAEval DataFrame)
# =========================================================================


def _find_cached_rcaeval_case() -> tuple[Path, Path, str] | None:
    """Locate a cached Online Boutique RCAEval case for real-data integration testing."""
    preferred = REPO_ROOT / "benchmarks" / "data" / "rcaeval" / "re1ob_adservice_cpu_1"
    pq = preferred / "metrics.parquet"
    inj = preferred / "inject_time.txt"
    if pq.exists() and inj.exists():
        return pq, inj, "adservice"
    base = REPO_ROOT / "benchmarks" / "data" / "rcaeval"
    if base.exists():
        for case_dir in sorted(base.iterdir()):
            pq_c = case_dir / "metrics.parquet"
            inj_c = case_dir / "inject_time.txt"
            if pq_c.exists() and inj_c.exists():
                parts = case_dir.name.split("_")
                if len(parts) >= 3:
                    return pq_c, inj_c, parts[1]
    return None


def test_rcaeval_rca_real_cached_dataframe_symmetric_and_blind() -> None:
    """Verify rcaeval_rca drop-in contract on a real cached RCAEval Kubernetes DataFrame."""
    found = _find_cached_rcaeval_case()
    if found is None:
        pytest.skip("No cached RCAEval parquet found under benchmarks/data/rcaeval/")
    pq_path, inj_path, true_service = found

    df = pd.read_parquet(pq_path)
    inject_time = int(inj_path.read_text(encoding="utf-8").strip())
    expected_services = {c.split("_", 1)[0] for c in df.columns if c != "time"}

    store = StreamStore()
    res_sym = rcaeval_rca(df, inject_time=inject_time, top_k=5, store=store)

    # Contract structure check
    assert set(res_sym.keys()) == {"ranks", "scores", "metric_top_k", "digest"}
    ranks = res_sym["ranks"]
    scores = res_sym["scores"]
    metric_top_k = res_sym["metric_top_k"]
    digest = res_sym["digest"]

    assert isinstance(ranks, list) and len(ranks) == len(expected_services)
    assert set(ranks) == expected_services
    assert list(scores.keys()) == ranks
    assert all(isinstance(v, float) and math.isfinite(v) for v in scores.values())
    # Scores must be non-increasing along ranks
    score_vals = [scores[s] for s in ranks]
    assert all(score_vals[i] >= score_vals[i + 1] for i in range(len(score_vals) - 1))

    assert isinstance(metric_top_k, list) and len(metric_top_k) == 5
    for item in metric_top_k:
        assert {"rank", "channel", "name", "service", "root_score"}.issubset(item.keys())
        assert item["service"] in expected_services

    assert isinstance(digest, CausalDigest)
    assert digest.reduction_pct >= 95.0
    assert len(digest.recovery_handles) >= 1
    # Verify lossless recovery handle registered in StreamStore works end-to-end
    recovered = recover_handle(digest.recovery_handles[0], store=store)
    assert len(recovered.values) > 0

    # Real Kubernetes incident localization check (on re1ob_adservice_cpu_1, true_service is top-3)
    assert true_service in ranks[:3], (
        f"Expected {true_service!r} in top-3 symmetric ranks, got {ranks[:5]}"
    )

    # Shuffled row order must produce identical ranks & scores (chronological sorting invariance)
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    res_shuf = rcaeval_rca(df_shuffled, inject_time=inject_time, top_k=5)
    assert res_shuf["ranks"] == ranks
    assert [res_shuf["scores"][s] for s in ranks] == pytest.approx(score_vals, rel=1e-12, abs=1e-12)

    # Blind mode (inject_time=None) must satisfy exact same contract
    res_blind = rcaeval_rca(df, inject_time=None, top_k=5)
    assert set(res_blind.keys()) == {"ranks", "scores", "metric_top_k", "digest"}
    assert set(res_blind["ranks"]) == expected_services
    assert isinstance(res_blind["digest"], CausalDigest)
    assert true_service in res_blind["ranks"][:5], (
        f"Expected {true_service!r} in top-5 blind ranks, got {res_blind['ranks'][:5]}"
    )


def test_rcaeval_rca_screened_mode_and_custom_service_parser() -> None:
    """Verify rcaeval_rca Tier-2 screened contract and custom service_parser hook."""
    X = _make_amplified_cascade(T=350, D=12, shock_start=280, seed=303)
    # Use custom colon-delimited service::metric schema
    cols = {
        "time": list(range(1_700_000_000, 1_700_000_000 + 350)),
        "svc-alpha::cpu": X[:, 0].tolist(),
        "svc-beta::lat": X[:, 1].tolist(),
        "svc-gamma::lat": X[:, 2].tolist(),
        **{f"svc-bg{i}::cpu": X[:, i].tolist() for i in range(3, 12)},
    }

    res_t1 = rcaeval_rca(
        cols,
        inject_time=1_700_000_280,
        service_parser=lambda col: col.split("::", 1)[0],
        use_screened=False,
    )
    assert res_t1["ranks"][0] == "svc-alpha"
    assert len(res_t1["ranks"]) == 12

    res_t2 = rcaeval_rca(
        cols,
        inject_time=None,
        service_parser=lambda col: col.split("::", 1)[0],
        use_screened=True,
        active_m=6,
        top_k=5,
    )
    assert set(res_t2.keys()) == {"ranks", "scores", "metric_top_k", "digest"}
    assert len(res_t2["ranks"]) == 12
    assert isinstance(res_t2["digest"], CausalDigest)
    assert res_t2["ranks"][0] == "svc-alpha"


# =========================================================================
# 4. analyze_otlp_metrics tests (OpenTelemetry OTLP/JSON)
# =========================================================================


def test_analyze_otlp_metrics_gauge_sum_histogram_ffill_and_rca() -> None:
    """Verify OTLP/JSON gauge, sum, histogram (sum/count & sum-only), naming, causal ffill, and RCA."""
    T = 650
    shock_start = 520
    X = _make_amplified_cascade(T=T, D=6, shock_start=shock_start, seed=404)
    base_ns = 1_710_000_000_000_000_000
    step_ns = 1_000_000_000  # 1s in exact 64-bit nanoseconds

    # Build realistic OTLP ExportMetricsServiceRequest:
    # ch0: db / cpu.utilization (gauge asDouble) -> root cause
    # ch1: api / http.server.duration (histogram sum/count, missing interior ticks 10, 11)
    # ch2: gateway / request.latency (histogram sum-only without count, missing leading ticks 0, 1)
    # ch3: cache / ops.total (sum asInt string-encoded per protobuf JSON spec)
    # ch4 & ch5: worker / queue.depth (gauge duplicated across 2 datapoint attribute sets)
    db_dps: list[dict[str, Any]] = []
    api_dps: list[dict[str, Any]] = []
    gw_dps: list[dict[str, Any]] = []
    cache_dps: list[dict[str, Any]] = []
    worker_dps: list[dict[str, Any]] = []

    for idx in range(T):
        ts_str = str(base_ns + idx * step_ns)
        db_dps.append({"timeUnixNano": ts_str, "asDouble": float(X[idx, 0])})

        if idx not in (10, 11):
            cnt = 4
            api_dps.append(
                {
                    "timeUnixNano": ts_str,
                    "count": str(cnt),
                    "sum": float(X[idx, 1] * cnt),
                }
            )

        if idx not in (0, 1):
            # Histogram without count -> falls back cleanly to sum
            gw_dps.append({"timeUnixNano": ts_str, "sum": float(X[idx, 2])})

        # Protobuf JSON encodes int64 as string on odd ticks, int on even ticks
        int_val = int(round(X[idx, 3] * 100.0))
        cache_dps.append(
            {
                "timeUnixNano": ts_str,
                "asInt": str(int_val) if idx % 2 else int_val,
            }
        )

        worker_dps.append(
            {
                "timeUnixNano": ts_str,
                "asDouble": float(X[idx, 4]),
                "attributes": [{"key": "queue", "value": {"stringValue": "fast"}}],
            }
        )
        worker_dps.append(
            {
                "timeUnixNano": ts_str,
                "asDouble": float(X[idx, 5]),
                "attributes": [{"key": "queue", "value": {"stringValue": "slow"}}],
            }
        )

    otlp_payload: dict[str, Any] = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "db"}}]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "otel.runtime"},
                        "metrics": [
                            {"name": "cpu.utilization", "gauge": {"dataPoints": db_dps}}
                        ],
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "api"}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {"name": "http.server.duration", "histogram": {"dataPoints": api_dps}}
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "gateway"}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {"name": "request.latency", "histogram": {"dataPoints": gw_dps}}
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "cache"}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {"name": "ops.total", "sum": {"dataPoints": cache_dps}}
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "worker"}}]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {"name": "queue.depth", "gauge": {"dataPoints": worker_dps}}
                        ]
                    }
                ],
            },
        ]
    }
    otlp_json = json.dumps(otlp_payload)

    sieve_d, digest_d = analyze_otlp_metrics(otlp_payload, lag=2, top_k=5, token_budget=400)
    try:
        assert isinstance(sieve_d, CausalSieve)
        assert isinstance(digest_d, CausalDigest)
        assert sieve_d.ticks == T
        assert sieve_d.channel_names == [
            "db_cpu.utilization",
            "api_http.server.duration",
            "gateway_request.latency",
            "cache_ops.total",
            "worker_queue.depth{queue=fast}",
            "worker_queue.depth{queue=slow}",
        ]
        assert digest_d.alarm is True
        assert digest_d.root_causes[0].name == "db_cpu.utilization"

        # Verify histogram sum/count exact arithmetic and interior causal forward-fill (ticks 10, 11)
        api_sl = sieve_d.get_raw_slice("api_http.server.duration", 9, 13)
        assert api_sl.values == pytest.approx(
            [float(X[9, 1]), float(X[9, 1]), float(X[9, 1]), float(X[12, 1])],
            rel=1e-12,
            abs=1e-12,
        )

        # Verify histogram sum-only fallback and leading missing ticks -> 0.0 (never backward-filled!)
        gw_sl = sieve_d.get_raw_slice("gateway_request.latency", 0, 3)
        assert gw_sl.values == pytest.approx([0.0, 0.0, float(X[2, 2])], abs=1e-12)

        # Verify asInt string/int parsing on sum metric
        cache_sl = sieve_d.get_raw_slice("cache_ops.total", 0, 4)
        assert cache_sl.values == pytest.approx(
            [float(int(round(X[i, 3] * 100.0))) for i in range(4)],
            abs=0.0,
        )
    finally:
        sieve_d.close()

    # Exact parity on JSON string and UTF-8 bytes
    sieve_s, digest_s = analyze_otlp_metrics(otlp_json, lag=2, top_k=5, token_budget=400)
    sieve_b, digest_b = analyze_otlp_metrics(
        otlp_json.encode("utf-8"), lag=2, top_k=5, token_budget=400
    )
    try:
        assert [r.name for r in digest_s.root_causes] == [r.name for r in digest_d.root_causes]
        assert [r.name for r in digest_b.root_causes] == [r.name for r in digest_d.root_causes]
        assert digest_s.root_causes[0].root_score == pytest.approx(
            digest_d.root_causes[0].root_score, rel=1e-12
        )
    finally:
        sieve_s.close()
        sieve_b.close()


def test_analyze_otlp_metrics_validation_errors() -> None:
    """Verify clear validation errors on malformed OTLP payloads."""
    with pytest.raises(TypeError):
        analyze_otlp_metrics(12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resourceMetrics"):
        analyze_otlp_metrics({"invalid": []})
    with pytest.raises(ValueError, match="zero resourceMetrics"):
        analyze_otlp_metrics({"resourceMetrics": []})
    with pytest.raises(ValueError, match="no finite numeric"):
        analyze_otlp_metrics(
            {
                "resourceMetrics": [
                    {
                        "resource": {"attributes": []},
                        "scopeMetrics": [
                            {
                                "metrics": [
                                    {
                                        "name": "empty",
                                        "gauge": {
                                            "dataPoints": [
                                                {"timeUnixNano": "100", "asDouble": "NaN"}
                                            ]
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        )


# =========================================================================
# 5. analyze_jsonl tests (NDJSON / JSONL streams)
# =========================================================================


def test_analyze_jsonl_file_string_iterable_and_timestamp_exclusion(tmp_path: Path) -> None:
    """Verify analyze_jsonl across file path, raw JSONL string, and iterables with causal ffill."""
    T = 650
    shock_start = 520
    X = _make_amplified_cascade(T=T, D=6, shock_start=shock_start, seed=505)
    names = ["db_cpu", "api_lat", "gw_lat", "cache_hit", "worker_q", "bg_io"]

    dict_records: list[dict[str, Any]] = []
    for t in range(T):
        rec: dict[str, Any] = {
            "ts_ms": 1_710_000_000_000 + t * 1000,
            "env": "prod",
            "canary": bool(t % 2),
            "meta": {"host": f"node-{t % 3}"},
        }
        for j, col in enumerate(names):
            if j == 1 and t == 50:
                rec[col] = None
            elif j == 2 and t == 0:
                rec[col] = float("nan")
            else:
                rec[col] = float(X[t, j])
        dict_records.append(rec)

    lines = [json.dumps(r) for r in dict_records]
    raw_jsonl = "\n".join(lines) + "\n"
    jsonl_file = tmp_path / "metrics.jsonl"
    jsonl_file.write_text(raw_jsonl, encoding="utf-8")

    # 1. File path (Path) with explicit time_col
    sieve_f, dig_f = analyze_jsonl(jsonl_file, time_col="ts_ms", lag=2, top_k=5)
    try:
        assert isinstance(sieve_f, CausalSieve)
        assert isinstance(dig_f, CausalDigest)
        assert sieve_f.channel_names == names
        assert sieve_f.ticks == T
        assert dig_f.alarm is True
        assert dig_f.root_causes[0].name == "db_cpu"

        # Strict causal forward-fill checks on null and leading NaN
        sl_api = sieve_f.get_raw_slice("api_lat", 49, 52)
        assert sl_api.values == pytest.approx(
            [float(X[49, 1]), float(X[49, 1]), float(X[51, 1])], abs=1e-12
        )
        sl_gw = sieve_f.get_raw_slice("gw_lat", 0, 2)
        assert sl_gw.values == pytest.approx([0.0, float(X[1, 2])], abs=1e-12)
    finally:
        sieve_f.close()

    # 2. Raw multiline string, str file path, iterable of lines, and iterable of dicts
    sieve_s, dig_s = analyze_jsonl(raw_jsonl, time_col="ts_ms", lag=2, top_k=5)
    sieve_p, dig_p = analyze_jsonl(str(jsonl_file), time_col="ts_ms", lag=2, top_k=5)
    sieve_iter_str, dig_iter_str = analyze_jsonl(iter(lines), time_col="ts_ms", lag=2, top_k=5)
    sieve_iter_dict, dig_iter_dict = analyze_jsonl(
        iter(dict_records), time_col="ts_ms", lag=2, top_k=5
    )
    try:
        for dig in (dig_s, dig_p, dig_iter_str, dig_iter_dict):
            assert [r.name for r in dig.root_causes] == [r.name for r in dig_f.root_causes]
            assert dig.root_causes[0].root_score == pytest.approx(
                dig_f.root_causes[0].root_score, rel=1e-12
            )
    finally:
        sieve_s.close()
        sieve_p.close()
        sieve_iter_str.close()
        sieve_iter_dict.close()

    # 3. Auto-skipping standard timestamp key without explicit time_col
    auto_records = [{"timestamp": 1000 + t, "db_cpu": float(X[t, 0]), "api_lat": float(X[t, 1])} for t in range(120)]
    sieve_auto, _ = analyze_jsonl(auto_records, lag=2)
    try:
        assert sieve_auto.channel_names == ["db_cpu", "api_lat"]
    finally:
        sieve_auto.close()


def test_analyze_jsonl_validation_errors(tmp_path: Path) -> None:
    """Verify clear validation errors on malformed or empty JSONL sources."""
    with pytest.raises(ValueError, match="zero records"):
        analyze_jsonl("   \n\n")
    with pytest.raises(FileNotFoundError):
        analyze_jsonl(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="Malformed JSON"):
        analyze_jsonl('{"a": 1.0}\n{bad_json}\n')
    with pytest.raises(ValueError, match="JSON object"):
        analyze_jsonl("[1, 2, 3]\n")
    with pytest.raises(TypeError):
        analyze_jsonl({"a": 1.0})  # type: ignore[arg-type]


# =========================================================================
# 6. Live streaming `causalsieve watch` CLI tests
# =========================================================================


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PYTHON_DIR}{os.pathsep}{existing}" if existing else str(PYTHON_DIR)
    return env


def test_cli_watch_live_csv_stdin_streaming_alerts_and_summary() -> None:
    """Pipe live CSV rows into `causalsieve watch -` and verify pre-EOF alerts + final summary."""
    T = 620
    shock_start = 480
    X = _make_amplified_cascade(T=T, D=6, shock_start=shock_start, seed=606)
    cols = ["db_cpu", "api_lat", "gw_lat", "cache_hit", "worker_q", "bg_io"]

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "causalsieve",
            "watch",
            "-",
            "--input-format",
            "csv",
            "--lag",
            "2",
            "--alpha",
            "0.01",
            "--top-k",
            "4",
            "--format",
            "jsonl",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write("timestamp," + ",".join(cols) + "\n")
        for t in range(T):
            row_vals = ",".join(f"{X[t, j]:.6f}" for j in range(len(cols)))
            proc.stdin.write(f"{1_710_000_000 + t},{row_vals}\n")
        proc.stdin.flush()

        # Verify real-time streaming emission BEFORE stdin EOF!
        first_alert_line = proc.stdout.readline()
        assert first_alert_line, "Expected at least one real-time alarm line before EOF"
        first_alert = json.loads(first_alert_line)
        assert first_alert["event"] == "alarm"
        for required_key in (
            "tick",
            "e_value",
            "max_exo_z",
            "top_root_channel",
            "top_root_name",
            "ranked_top_k",
        ):
            assert required_key in first_alert
        assert first_alert["tick"] >= shock_start - 5
        assert first_alert["e_value"] >= 100.0 or first_alert["max_exo_z"] >= 4.5
        assert len(first_alert["ranked_top_k"]) == 4

        # Close stdin and consume remaining events through final summary
        proc.stdin.close()
        remaining = [json.loads(line) for line in proc.stdout if line.strip()]
        stderr_txt = proc.stderr.read() if proc.stderr is not None else ""
        rc = proc.wait(timeout=15)
        assert rc == 0, f"causalsieve watch failed (rc={rc}): {stderr_txt}"

        events = [first_alert, *remaining]
        summary = events[-1]
        assert summary["event"] == "summary"
        assert summary["ticks"] == T
        assert summary["dim"] == len(cols)
        assert summary["alarm"] is True
        assert summary["alarm_count"] >= 1
        assert summary["max_e_value"] >= 100.0
        assert summary["top_root_name"] == "db_cpu"
        assert summary["ranked_top_k"][0]["name"] == "db_cpu"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_cli_watch_live_jsonl_stdin_every_status_and_md_file(tmp_path: Path) -> None:
    """Verify `causalsieve watch` on piped JSONL stdin (--every status) and file Markdown output."""
    T = 620
    shock_start = 480
    X = _make_amplified_cascade(T=T, D=6, shock_start=shock_start, seed=707)
    cols = ["db_cpu", "api_lat", "gw_lat", "cache_hit", "worker_q", "bg_io"]

    jsonl_lines = [
        json.dumps(
            {
                "ts_ms": 1_710_000_000_000 + t * 1000,
                "service_tag": "prod",
                **{cols[j]: round(float(X[t, j]), 6) for j in range(len(cols))},
            }
        )
        for t in range(T)
    ]
    jsonl_text = "\n".join(jsonl_lines) + "\n"

    # 1. Pipe JSONL into stdin with auto format detection, --time-col, and --every 150
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "causalsieve",
            "watch",
            "--input-format",
            "auto",
            "--time-col",
            "ts_ms",
            "--every",
            "150",
            "--lag",
            "2",
            "--top-k",
            "3",
            "--format",
            "jsonl",
        ],
        input=jsonl_text,
        capture_output=True,
        text=True,
        timeout=20,
        env=_cli_env(),
        check=False,
    )
    assert proc.returncode == 0, f"watch JSONL failed: {proc.stderr}"
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    statuses = [e for e in events if e["event"] == "status"]
    alarms = [e for e in events if e["event"] == "alarm"]
    summary = events[-1]

    assert len(statuses) == T // 150
    assert len(alarms) >= 1
    assert summary["event"] == "summary"
    assert summary["ticks"] == T
    assert summary["dim"] == len(cols)
    assert summary["top_root_name"] == "db_cpu"
    assert len(summary["ranked_top_k"]) == 3

    # 2. File mode with Markdown output format
    jsonl_path = tmp_path / "stream.jsonl"
    jsonl_path.write_text(jsonl_text, encoding="utf-8")
    proc_md = subprocess.run(
        [
            sys.executable,
            "-m",
            "causalsieve",
            "watch",
            str(jsonl_path),
            "--time-col",
            "ts_ms",
            "--format",
            "md",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=_cli_env(),
        check=False,
    )
    assert proc_md.returncode == 0, f"watch md failed: {proc_md.stderr}"
    assert "**[ALARM tick=" in proc_md.stdout
    assert "## Watch Summary (ALARM" in proc_md.stdout
    assert "`db_cpu`" in proc_md.stdout


def test_top_level_lazy_exports() -> None:
    """Verify adapter functions are exposed via causalsieve top-level __all__."""
    for name in (
        "analyze_dataframe",
        "analyze_jsonl",
        "analyze_otlp_metrics",
        "analyze_prometheus",
        "rcaeval_rca",
    ):
        assert name in causalsieve.__all__
        assert callable(getattr(causalsieve, name))

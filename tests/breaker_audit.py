"""Hostile Independent Adversarial Breaker Audit for CausalSieve.

Actively attempts to break CausalSieve across 7 independent attack surfaces:
  1. Frozen Provenance Integrity (live binary SHA-256 vs all 5 benchmark JSONs +
     RCAEval DEV/HOLDOUT split isolation and summary arithmetic verification).
  2. Zero-Runtime-Dependency Proof (bare system python3 with confirmed absent NumPy
     exercising full Python API, CLI, Digest, Blackboard, CSV, and MCP stdio).
  3. Hostile Numerical & Edge-Case Torture (D=1/p=1, D=2/p=8, nonlinear hinge,
     1e12 DC + 1e-12 micro-fluctuation, exact duplicates + zero columns + sudden
     1e-4 / 1e6 spikes, max_abs_err=0.0 lossless PLA, reset() bit-exact determinism,
     and alternating +-1e7 bursts).
  4. Hostile Causality Red-Team (assert_causal with +-1e100 future poison and
     atomic NaN/Inf rejection leaving subsequent ticks bit-identical).
  5. Hostile Native CLI (`build/causalsieve-bin`) Fuzzing (valid CSV, stdin pipe,
     empty, header-only, 1-row, malformed tokens, NaN rows, --compact, invalid flags).
  6. Hostile MCP Security & Protocol Red-Team (bare system python3, --allowed-root
     path traversal, symlink escape, malformed JSON, unknown method, missing params,
     invalid channel indices, NaN payloads, huge horizon, post-attack liveness).
  7. Clean-Room Anonymization & ASan/UBSan Gate (test_core_cpp_asan zero-finding
     execution + full repo scan for leaked personal/workspace paths, tokens, keys).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
BUILD_DIR = REPO_ROOT / "build"
BENCH_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_bare_system_python() -> str:
    """Locate a real python3 executable that does NOT have NumPy installed."""
    candidates: list[str] = []
    for p in ("/usr/bin/python3", "/bin/python3", shutil.which("python3")):
        if p and Path(p).is_file():
            resolved = str(Path(p).resolve())
            if ".bench_venv" not in resolved and p not in candidates:
                candidates.append(p)

    for cand in candidates:
        probe = subprocess.run(
            [cand, "-c", "import numpy"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0 and "ModuleNotFoundError" in probe.stderr:
            return cand

    # Fallback for runners where distro /usr/bin/python3 preinstalls python3-numpy:
    # create an isolated stdlib-only venv and verify NumPy is genuinely absent.
    bare_venv = Path("/tmp/causalsieve_bare_audit_venv")
    bare_py = bare_venv / "bin" / "python"
    if not bare_py.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(bare_venv)],
            check=True,
            timeout=30,
        )
    probe = subprocess.run(
        [str(bare_py), "-c", "import numpy"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0 and "ModuleNotFoundError" in probe.stderr:
        return str(bare_py)

    pytest.fail(
        f"Could not find a bare python3 without NumPy. Candidates checked: {candidates}"
    )


def _bare_python_env() -> dict[str, str]:
    """Construct a clean environment with repo python/ on PYTHONPATH and no venv leakage."""
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(PYTHON_DIR),
        "PYTHONNOUSERSITE": "1",
    }
    return env


def _make_clean_var_rows(
    n_ticks: int, dim: int, seed: int = 1337, shock_at: int | None = None
) -> list[list[float]]:
    """Pure-stdlib deterministic multivariate VAR(2) generator (zero NumPy dependency)."""
    # Simple LCG + Box-Muller for exact reproducibility without numpy
    state = seed & 0xFFFFFFFFFFFFFFFF

    def rand_u() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((state >> 11) + 0.5) / float(1 << 53)

    def rand_n() -> float:
        u1 = max(1e-15, rand_u())
        u2 = rand_u()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    rows: list[list[float]] = []
    prev1 = [0.0] * dim
    prev2 = [0.0] * dim
    for t in range(n_ticks):
        cur = [0.0] * dim
        for j in range(dim):
            eps = rand_n() * (0.4 if j > 0 and j < 4 else 1.0)
            val = 0.30 * prev1[j] - 0.08 * prev2[j] + eps
            if j == 0 and shock_at is not None and t >= shock_at:
                val += 7.0
            if j in (1, 2, 3) and j - 1 < dim:
                val += 0.65 * prev1[j - 1]
            cur[j] = val
        rows.append(cur)
        prev2 = prev1
        prev1 = cur
    return rows


def _linear_interp_at_int_ticks(
    knots: list[tuple[int, float]], start_tick: int, end_tick: int
) -> list[float]:
    """Evaluate piecewise linear interpolation of sorted PLA knots at every integer tick."""
    assert len(knots) >= 1
    out: list[float] = []
    k_idx = 0
    n_k = len(knots)
    for t in range(start_tick, end_tick):
        while k_idx + 1 < n_k and knots[k_idx + 1][0] < t:
            k_idx += 1
        t0, v0 = knots[k_idx]
        if t == t0 or k_idx + 1 >= n_k:
            out.append(v0)
        else:
            t1, v1 = knots[k_idx + 1]
            if t == t1:
                out.append(v1)
            else:
                slope = (v1 - v0) / float(t1 - t0)
                out.append(v0 + slope * float(t - t0))
    return out


def _assert_all_finite_nested(obj: Any, label: str) -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float in {label}: {obj!r}"
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_all_finite_nested(item, f"{label}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite_nested(v, f"{label}.{k}")


# =========================================================================
# 1. FROZEN PROVENANCE INTEGRITY
# =========================================================================


class TestFrozenProvenanceIntegrity:
    """Verify every frozen benchmark JSON matches the exact compiled binaries on disk."""

    REQUIRED_JSONS = (
        "speed.json",
        "optimization_log.json",
        "synthetic_rca.json",
        "sift_scale.json",
        "rcaeval.json",
        "rcaeval_trainticket.json",
        "rcaeval_suite.json",
        "agent_tokens.json",
    )

    def test_live_binary_sha256_matches_all_benchmark_jsons(self) -> None:
        lib_so = BUILD_DIR / "libcausalsieve.so"
        lib_scalar_so = BUILD_DIR / "libcausalsieve_scalar.so"
        assert lib_so.is_file(), f"Missing native library: {lib_so}"
        assert lib_scalar_so.is_file(), f"Missing scalar native library: {lib_scalar_so}"

        live_sha = _sha256_file(lib_so)
        live_scalar_sha = _sha256_file(lib_scalar_so)
        assert len(live_sha) == 64 and len(live_scalar_sha) == 64
        assert live_sha != live_scalar_sha, "AVX2 and scalar .so unexpectedly identical"

        for fname in self.REQUIRED_JSONS:
            jpath = BENCH_RESULTS_DIR / fname
            assert jpath.is_file(), f"Missing frozen benchmark JSON: {jpath}"
            data = json.loads(jpath.read_text(encoding="utf-8"))
            sys_meta = data.get("system", {})
            assert sys_meta.get("lib_sha256") == live_sha, (
                f"STALE BENCHMARK in {fname}: recorded lib_sha256="
                f"{sys_meta.get('lib_sha256')} != live {live_sha}"
            )
            assert sys_meta.get("lib_scalar_sha256") == live_scalar_sha, (
                f"STALE BENCHMARK in {fname}: recorded lib_scalar_sha256="
                f"{sys_meta.get('lib_scalar_sha256')} != live {live_scalar_sha}"
            )
            if fname == "rcaeval.json":
                proto_sha = data.get("protocol", {}).get("lib_sha256")
                assert proto_sha == live_sha, (
                    f"STALE PROTOCOL SHA in rcaeval.json: {proto_sha} != live {live_sha}"
                )

    def test_rcaeval_dev_holdout_sha256_isolation_and_metric_integrity(self) -> None:
        rpath = BENCH_RESULTS_DIR / "rcaeval.json"
        data = json.loads(rpath.read_text(encoding="utf-8"))
        splits = data["splits"]
        dev = splits["dev"]
        holdout = splits["holdout"]

        assert dev["n_cases"] == 10 and len(dev["cases"]) == 10
        assert holdout["n_cases"] == 40 and len(holdout["cases"]) == 40

        hex64 = re.compile(r"^[0-9a-f]{64}$")

        def collect_split_hashes(cases: list[dict[str, Any]], expected_split: str) -> tuple[set[str], set[str], set[str]]:
            cids: set[str] = set()
            pq_hashes: set[str] = set()
            inj_hashes: set[str] = set()
            for c in cases:
                assert c["split"] == expected_split, f"Misplaced split label in {c['case_id']}"
                cid = c["case_id"]
                pq = c.get("parquet_sha256", "")
                inj = c.get("inject_sha256", "")
                assert hex64.match(pq), f"Invalid parquet_sha256 in {cid}: {pq!r}"
                assert hex64.match(inj), f"Invalid inject_sha256 in {cid}: {inj!r}"
                assert cid not in cids, f"Duplicate case_id within {expected_split}: {cid}"
                assert pq not in pq_hashes, f"Duplicate parquet_sha256 within {expected_split}: {cid}"
                assert inj not in inj_hashes, f"Duplicate inject_sha256 within {expected_split}: {cid}"
                cids.add(cid)
                pq_hashes.add(pq)
                inj_hashes.add(inj)
            return cids, pq_hashes, inj_hashes

        dev_ids, dev_pq, dev_inj = collect_split_hashes(dev["cases"], "dev")
        hold_ids, hold_pq, hold_inj = collect_split_hashes(holdout["cases"], "holdout")

        assert dev_ids.isdisjoint(hold_ids), f"Case ID leak between DEV & HOLDOUT: {dev_ids & hold_ids}"
        assert dev_pq.isdisjoint(hold_pq), f"Parquet SHA-256 leak between DEV & HOLDOUT: {dev_pq & hold_pq}"
        assert dev_inj.isdisjoint(hold_inj), f"Inject SHA-256 leak between DEV & HOLDOUT: {dev_inj & hold_inj}"

        # Independently recompute summary metrics from raw per-case ranks to verify zero fabrication
        for split_name, split_obj in (("dev", dev), ("holdout", holdout)):
            cases = split_obj["cases"]
            n = len(cases)
            for method, reported in split_obj["summary"].items():
                ranks = [int(c["predictions"][method]["rank"]) for c in cases]
                ac1 = sum(1 for r in ranks if r <= 1) / n
                ac3 = sum(1 for r in ranks if r <= 3) / n
                ac5 = sum(1 for r in ranks if r <= 5) / n
                avg5 = sum((sum(1 for r in ranks if r <= k) / n) for k in range(1, 6)) / 5.0
                mrr = sum(1.0 / r for r in ranks) / n
                mean_rank = sum(ranks) / n
                assert reported["n"] == n
                assert abs(reported["ac1"] - ac1) < 1e-9, f"{split_name}.{method}.ac1 mismatch"
                assert abs(reported["ac3"] - ac3) < 1e-9, f"{split_name}.{method}.ac3 mismatch"
                assert abs(reported["ac5"] - ac5) < 1e-9, f"{split_name}.{method}.ac5 mismatch"
                assert abs(reported["avg5"] - avg5) < 1e-9, f"{split_name}.{method}.avg5 mismatch"
                assert abs(reported["mrr"] - mrr) < 1e-9, f"{split_name}.{method}.mrr mismatch"
                assert abs(reported["mean_rank"] - mean_rank) < 1e-9, (
                    f"{split_name}.{method}.mean_rank mismatch"
                )


# =========================================================================
# 2. ZERO-RUNTIME-DEPENDENCY PROOF (BARE SYSTEM PYTHON3)
# =========================================================================


class TestZeroRuntimeDependencyProof:
    """Prove causalsieve works end-to-end under bare system python3 with NO NumPy."""

    def test_bare_system_python_full_api_cli_and_mcp(self, tmp_path: Path) -> None:
        bare_py = _find_bare_system_python()
        env = _bare_python_env()

        # Prepare a small cascade CSV inside tmp_path
        csv_path = tmp_path / "zero_dep_stream.csv"
        rows = _make_clean_var_rows(260, 5, seed=9091, shock_at=200)
        names = ["db_cpu", "api_lat", "web_lat", "cache_hit", "disk_io"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tick", *names])
            for t, row in enumerate(rows):
                w.writerow([t, *[f"{v:.17g}" for v in row]])

        # 1. Verify CLI `causalsieve info` directly via bare system python3
        cli_proc = subprocess.run(
            [bare_py, "-m", "causalsieve", "info", "--json"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert cli_proc.returncode == 0, f"CLI info failed: {cli_proc.stderr}"
        info_json = json.loads(cli_proc.stdout)
        assert info_json["package"] == "causalsieve"
        assert info_json["status"] == "ready"
        assert info_json["simd_isa"] in ("avx2+fma", "scalar", "neon")

        # 2. Run comprehensive zero-dependency script inside bare system python3
        script = f"""
import json, math, subprocess, sys

# Hard proof: NumPy MUST NOT be importable in this interpreter!
try:
    import numpy  # noqa: F401
    raise AssertionError("NumPy unexpectedly present in bare system python!")
except ModuleNotFoundError:
    pass

assert ".bench_venv" not in sys.executable, f"Running inside venv: {{sys.executable}}"

import causalsieve
from causalsieve import (
    CausalBlackboard,
    CausalDigest,
    CausalSieve,
    analyze_csv,
    build_digest,
    recover_handle,
)
from causalsieve.cli import main as cli_main

assert "numpy" not in sys.modules
assert "scipy" not in sys.modules
assert "pandas" not in sys.modules

# Exercise causalsieve info via CLI main
assert cli_main(["info", "--json"]) == 0

# Deterministic pure-Python stream
csv_path = {str(csv_path)!r}
with analyze_csv(csv_path, lag=2, use_screened=False) as csv_res:
    assert csv_res.n_rows == 260
    assert csv_res.channel_names == {names!r}
    dig = csv_res.digest(top_k=5, max_err=0.05)
    assert isinstance(dig, CausalDigest)
    assert dig.root_causes[0].name == "db_cpu", f"Expected db_cpu root, got {{dig.root_causes[0]}}"
    md = dig.to_markdown()
    assert "# CausalDigest" in md
    assert len(dig.recovery_handles) >= 1
    sl_h = recover_handle(dig.recovery_handles[0])
    assert len(sl_h.values) > 0

# Direct CausalSieve full method surface
with CausalSieve(dim=5, lag=2, channel_names={names!r}, raw_ring_capacity=1024) as eng:
    r0 = eng.push([0.1, 0.2, 0.3, 0.4, 0.5])
    assert r0.tick == 0 and r0.exo_z is not None and r0.innov is not None
    # Push remaining rows from CSV via push_batch
    batch = []
    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        rd = _csv.reader(fh)
        next(rd)
        for r in rd:
            batch.append([float(x) for x in r[1:]])
    res_list = eng.push_batch(batch, return_results=True, return_exo_z=True, return_innov=True)
    assert res_list is not None and len(res_list) == len(batch)

    gr = eng.granger_matrix()
    pr = eng.precision_matrix()
    ori = eng.contemporaneous_orientation()
    attr = eng.root_attribution()
    cf = eng.counterfactual_do("db_cpu", 3.0, horizon=8)
    pla = eng.compress_pla("db_cpu", 10, 250, max_abs_err=0.05)
    raw = eng.get_raw_slice("db_cpu", 1, 20)
    st = eng.inspect_state()

    assert len(gr.f_stat) == 5 and len(pr.theta) == 5 and len(ori) == 5
    assert attr.top_name == "db_cpu"
    assert len(cf) == 8 and abs(cf[0][0] - 3.0) < 1e-9
    assert len(pla.knots) >= 2
    assert len(raw.values) == 19
    assert math.isfinite(st.freivalds_rel_err)

# CausalBlackboard end-to-end
bb = CausalBlackboard()
try:
    sid = bb.create_stream("bare-stream", dim=5, lag=2, channel_names={names!r})
    bb.push_batch(sid, batch, return_results=False)
    snap = bb.snapshot(sid)
    assert snap.as_of_tick == len(batch)
    h1 = bb.verify_hypothesis(sid, "agent-a", "db_cpu", delta=4.0, horizon=6, confidence=0.9)
    h2 = bb.verify_hypothesis(sid, "agent-b", "web_lat", delta=4.0, horizon=6, confidence=0.9)
    assert h1.objective_score > h2.objective_score
    cons = bb.consensus(sid)
    assert cons.winner_name == "db_cpu"
finally:
    bb.close()

# Full MCP stdio JSON-RPC session spawned using the SAME bare system python3!
mcp_proc = subprocess.Popen(
    [sys.executable, "-m", "causalsieve.mcp_server", "--allowed-root", {str(tmp_path)!r}],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd={str(REPO_ROOT)!r},
)

def rpc(mid, method, params=None):
    msg = {{"jsonrpc": "2.0", "id": mid, "method": method}}
    if params is not None:
        msg["params"] = params
    mcp_proc.stdin.write(json.dumps(msg) + "\\n")
    mcp_proc.stdin.flush()
    line = mcp_proc.stdout.readline()
    assert line, f"MCP EOF; stderr={{mcp_proc.stderr.read()}}"
    return json.loads(line)

try:
    r_init = rpc(1, "initialize", {{"protocolVersion": "2024-11-05"}})
    assert r_init["result"]["serverInfo"]["name"] == "causalsieve"
    mcp_proc.stdin.write(json.dumps({{"jsonrpc": "2.0", "method": "notifications/initialized"}}) + "\\n")
    mcp_proc.stdin.flush()

    r_list = rpc(2, "tools/list", {{}})
    assert len(r_list["result"]["tools"]) == 6

    r_ing = rpc(3, "tools/call", {{
        "name": "sieve_ingest_csv",
        "arguments": {{"path": csv_path, "stream_id": "bare-mcp", "format": "json"}},
    }})
    assert not r_ing["result"]["isError"], r_ing
    ing_data = json.loads(r_ing["result"]["content"][0]["text"])
    assert ing_data["root_causes"][0]["name"] == "db_cpu"

    r_cf = rpc(4, "tools/call", {{
        "name": "sieve_counterfactual",
        "arguments": {{"stream_id": "bare-mcp", "channel": "db_cpu", "delta": 2.0, "horizon": 5}},
    }})
    assert not r_cf["result"]["isError"], r_cf

    r_sl = rpc(5, "tools/call", {{
        "name": "sieve_recover_slice",
        "arguments": {{"stream_id": "bare-mcp", "channel": "db_cpu", "start_tick": 0, "end_tick": 10}},
    }})
    assert not r_sl["result"]["isError"], r_sl
    assert len(json.loads(r_sl["result"]["content"][0]["text"])["values"]) == 10
finally:
    mcp_proc.stdin.close()
    mcp_proc.wait(timeout=10)
    assert mcp_proc.returncode == 0

assert "numpy" not in sys.modules
print("ZERO_DEP_OK")
"""
        proc = subprocess.run(
            [bare_py, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, (
            f"Bare-python zero-dep test failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "ZERO_DEP_OK" in proc.stdout


# =========================================================================
# 3. HOSTILE NUMERICAL & EDGE-CASE TORTURE
# =========================================================================


class TestHostileNumericalTorture:
    """Push the C++20 SIMD numerical core to extreme degenerate corners."""

    @pytest.mark.parametrize(
        "dim,lag,hinge",
        [
            (1, 1, False),
            (2, 8, False),
            (3, 1, True),
        ],
    )
    def test_boundary_dimensions_lags_and_hinge(self, dim: int, lag: int, hinge: bool) -> None:
        from causalsieve import CausalSieve

        rows = _make_clean_var_rows(220, dim, seed=3001 + dim * 10 + lag, shock_at=170)
        with CausalSieve(dim=dim, lag=lag, nonlinear_hinge=hinge, normalize_scale=True) as eng:
            expected_k = 1 + lag * dim + (dim if hinge else 0)
            assert eng.feature_dim == expected_k

            res = eng.push_batch(rows, return_results=True, return_exo_z=True, return_innov=True)
            assert res is not None and len(res) == len(rows)
            for r in res:
                assert math.isfinite(r.mahalanobis_sq) and r.mahalanobis_sq >= 0.0
                assert math.isfinite(r.e_value) and r.e_value >= 0.0
                assert math.isfinite(r.shock_weight) and 0.0 <= r.shock_weight <= 1.0
                assert math.isfinite(r.max_exo_z)
                _assert_all_finite_nested(r.exo_z, "exo_z")
                _assert_all_finite_nested(r.innov, "innov")

            gr = eng.granger_matrix()
            pr = eng.precision_matrix()
            ori = eng.contemporaneous_orientation()
            attr = eng.root_attribution()
            cf = eng.counterfactual_do(0, 2.5, horizon=6)
            st = eng.inspect_state()
            pla = eng.compress_pla(0, 0, len(rows), max_abs_err=0.1)
            raw = eng.get_raw_slice(0, 0, len(rows))

            _assert_all_finite_nested(gr.f_stat, "granger.f_stat")
            _assert_all_finite_nested(gr.partial_r2, "granger.partial_r2")
            _assert_all_finite_nested(pr.theta, "precision.theta")
            _assert_all_finite_nested(pr.pcorr, "precision.pcorr")
            _assert_all_finite_nested(ori, "orientation")
            _assert_all_finite_nested(attr.root_scores, "root_scores")
            _assert_all_finite_nested(cf, "counterfactual")
            _assert_all_finite_nested(st.B, "B")
            assert math.isfinite(st.freivalds_rel_err) and st.freivalds_rel_err < 1e-5
            assert len(pla.knots) >= 2
            assert raw.values == [row[0] for row in rows]

    def test_1e12_dc_offset_and_1e_minus_12_micro_fluctuation_simultaneous(self) -> None:
        from causalsieve import CausalSieve

        T, D = 350, 5
        base = _make_clean_var_rows(T, D, seed=3100, shock_at=280)
        rows: list[list[float]] = []
        for t in range(T):
            r = [
                1e12 + base[t][0],          # Huge +1e12 DC offset + O(1) signal
                1e-12 * base[t][1],         # Tiny 1e-12 micro-fluctuation
                -1e12 + 1e-12 * base[t][2], # Both -1e12 DC offset AND 1e-12 fluctuation!
                base[t][3],                 # Normal O(1)
                1e6 * base[t][4] + 1e9,     # Mid-huge scale + offset
            ]
            rows.append(r)

        with CausalSieve(dim=D, lag=2, normalize_scale=True) as eng:
            res = eng.push_batch(rows, return_results=True, return_exo_z=True, return_innov=True)
            assert res is not None
            for r in res:
                assert math.isfinite(r.mahalanobis_sq)
                assert math.isfinite(r.e_value)
                assert math.isfinite(r.max_exo_z)
                _assert_all_finite_nested(r.exo_z, "exo_z")
                _assert_all_finite_nested(r.innov, "innov")

            st = eng.inspect_state()
            gr = eng.granger_matrix()
            pr = eng.precision_matrix()
            ori = eng.contemporaneous_orientation()
            attr = eng.root_attribution()

            _assert_all_finite_nested(st.B, "B")
            assert math.isfinite(st.freivalds_rel_err) and st.freivalds_rel_err < 1e-4
            _assert_all_finite_nested(gr.f_stat, "granger.f_stat")
            _assert_all_finite_nested(gr.partial_r2, "granger.partial_r2")
            _assert_all_finite_nested(pr.theta, "precision.theta")
            _assert_all_finite_nested(pr.pcorr, "precision.pcorr")
            _assert_all_finite_nested(ori, "orientation")
            _assert_all_finite_nested(attr.root_scores, "root_scores")
            _assert_all_finite_nested(attr.peak_exo_z, "peak_exo_z")

    def test_exact_duplicates_all_zeros_and_sudden_micro_and_mega_spikes(self) -> None:
        from causalsieve import CausalSieve

        T, D = 380, 8
        base = _make_clean_var_rows(T, D, seed=3200)
        rows: list[list[float]] = []
        for t in range(T):
            c = base[t][0]
            # c0, c1, c2 are exact triple duplicates; c3 permanently 0;
            # c4: 0 -> sudden 1e-4 spike at t=260..280;
            # c5: 0 -> sudden 1e6 spike at t=310..330;
            # c6: permanently constant 42e6; c7: normal
            s_small = 1e-4 if 260 <= t < 280 else 0.0
            s_huge = 1e6 if 310 <= t < 330 else 0.0
            rows.append([c, c, c, 0.0, s_small, s_huge, 42e6, base[t][7]])

        with CausalSieve(dim=D, lag=2, normalize_scale=True) as eng:
            res = eng.push_batch(rows, return_results=True, return_exo_z=True, return_innov=True)
            assert res is not None
            for t, r in enumerate(res):
                assert math.isfinite(r.mahalanobis_sq) and r.mahalanobis_sq >= 0.0, f"tick {t}"
                assert math.isfinite(r.e_value) and r.e_value >= 0.0, f"tick {t}"
                assert math.isfinite(r.max_exo_z), f"tick {t}"
                _assert_all_finite_nested(r.exo_z, f"exo_z@{t}")
                _assert_all_finite_nested(r.innov, f"innov@{t}")

            st = eng.inspect_state()
            gr = eng.granger_matrix()
            pr = eng.precision_matrix()
            ori = eng.contemporaneous_orientation()
            attr = eng.root_attribution(lookback=80)

            _assert_all_finite_nested(st.B, "B")
            assert math.isfinite(st.freivalds_rel_err) and st.freivalds_rel_err < 1e-4
            _assert_all_finite_nested(gr.f_stat, "granger.f_stat")
            _assert_all_finite_nested(gr.partial_r2, "granger.partial_r2")
            _assert_all_finite_nested(pr.theta, "precision.theta")
            _assert_all_finite_nested(pr.pcorr, "precision.pcorr")
            _assert_all_finite_nested(ori, "orientation")
            _assert_all_finite_nested(attr.root_scores, "root_scores")

            # Huge 0 -> 1e6 spike on channel 5 must trigger strong exogenous response & top attribution
            assert abs(res[310].exo_z[5]) > 3.0  # type: ignore[index]
            assert attr.top_channel == 5, f"Expected channel 5 top root, got {attr.ranked_channels[:3]}"

    def test_exact_lossless_pla_max_abs_err_zero(self) -> None:
        from causalsieve import CausalSieve

        T = 200
        # ch0: nonlinear sine + step + micro jitter; ch1: exact linear ramp; ch2: constant
        rows: list[list[float]] = []
        for t in range(T):
            v0 = math.sin(0.17 * t) + (3.25 if t >= 90 else 0.0) + 1e-7 * ((t * 7) % 13)
            v1 = 10.0 + 0.5 * t
            v2 = -7.125
            rows.append([v0, v1, v2])

        with CausalSieve(dim=3, lag=1, raw_ring_capacity=1024) as eng:
            eng.push_batch(rows, return_results=False)
            for ch in range(3):
                raw = eng.get_raw_slice(ch, 0, T)
                pla = eng.compress_pla(ch, 0, T, max_abs_err=0.0)
                assert len(pla.knots) >= 2
                assert pla.knots[0][0] == 0 and pla.knots[-1][0] == T - 1
                # Strictly increasing knot ticks
                for i in range(1, len(pla.knots)):
                    assert pla.knots[i][0] > pla.knots[i - 1][0]

                interp = _linear_interp_at_int_ticks(pla.knots, 0, T)
                assert len(interp) == T
                max_err = max(abs(a - b) for a, b in zip(interp, raw.values, strict=True))
                assert max_err == 0.0, (
                    f"max_abs_err=0.0 PLA violated exact lossless guarantee on ch={ch}: "
                    f"max_err={max_err:.3e}, n_knots={len(pla.knots)}"
                )

    @pytest.mark.parametrize("normalize_scale", [False, True])
    def test_reset_bit_exact_determinism(self, normalize_scale: bool) -> None:
        from causalsieve import CausalSieve

        rows = _make_clean_var_rows(320, 6, seed=3400, shock_at=240)
        with CausalSieve(
            dim=6,
            lag=2,
            forgetting=0.99,
            normalize_scale=normalize_scale,
            refresh_interval=50,
            raw_ring_capacity=1024,
        ) as eng:
            def run_once() -> dict[str, Any]:
                res = eng.push_batch(rows, return_results=True, return_exo_z=True, return_innov=True)
                assert res is not None
                st = eng.inspect_state()
                gr = eng.granger_matrix()
                pr = eng.precision_matrix()
                ori = eng.contemporaneous_orientation()
                attr = eng.root_attribution()
                cf = eng.counterfactual_do(0, 3.0, horizon=8)
                sl = eng.get_raw_slice(0, 0, len(rows))
                pla = eng.compress_pla(0, 0, len(rows), max_abs_err=0.05)
                return {
                    "ticks": [r.tick for r in res],
                    "maha": [r.mahalanobis_sq for r in res],
                    "e_val": [r.e_value for r in res],
                    "weight": [r.shock_weight for r in res],
                    "max_exo": [r.max_exo_z for r in res],
                    "top_ch": [r.top_root_channel for r in res],
                    "alarm": [r.alarm for r in res],
                    "refresh": [r.refresh_count for r in res],
                    "exo_z": [r.exo_z for r in res],
                    "innov": [r.innov for r in res],
                    "B": st.B,
                    "freiv": st.freivalds_rel_err,
                    "gr_f": gr.f_stat,
                    "gr_r2": gr.partial_r2,
                    "theta": pr.theta,
                    "pcorr": pr.pcorr,
                    "ori": ori,
                    "root_scores": attr.root_scores,
                    "peak_exo": attr.peak_exo_z,
                    "raw_z": attr.raw_z,
                    "onset": attr.onset_tick,
                    "ranked": attr.ranked_channels,
                    "cf": cf,
                    "slice": sl.values,
                    "pla": pla.knots,
                }

            run1 = run_once()
            eng.reset()
            assert eng.ticks == 0
            assert eng.last_result is None
            assert eng.any_alarm is False
            run2 = run_once()

            for key in run1:
                assert run1[key] == run2[key], f"reset() determinism broken on output key={key!r}"

    @pytest.mark.parametrize("normalize_scale", [True, False])
    def test_alternating_plus_minus_1e7_burst_and_recovery(self, normalize_scale: bool) -> None:
        from causalsieve import CausalSieve

        D = 6
        rows = _make_clean_var_rows(450, D, seed=3500)
        # Inject alternating +-1e7 burst across all channels on ticks 180..229, then normal 230..449
        for t in range(180, 230):
            sign = 1.0 if (t % 2 == 0) else -1.0
            rows[t] = [sign * 1e7 * (1.0 + 0.1 * j) for j in range(D)]

        with CausalSieve(dim=D, lag=2, forgetting=0.99, normalize_scale=normalize_scale) as eng:
            res = eng.push_batch(rows, return_results=True, return_exo_z=True, return_innov=True)
            assert res is not None
            for t, r in enumerate(res):
                assert math.isfinite(r.mahalanobis_sq) and r.mahalanobis_sq >= 0.0, f"tick {t}"
                assert math.isfinite(r.e_value) and r.e_value >= 0.0, f"tick {t}"
                assert math.isfinite(r.shock_weight) and 0.0 <= r.shock_weight <= 1.0, f"tick {t}"
                assert math.isfinite(r.max_exo_z), f"tick {t}"
                _assert_all_finite_nested(r.exo_z, f"exo_z@{t}")
                _assert_all_finite_nested(r.innov, f"innov@{t}")

            st = eng.inspect_state()
            gr = eng.granger_matrix()
            pr = eng.precision_matrix()
            ori = eng.contemporaneous_orientation()
            attr = eng.root_attribution()

            _assert_all_finite_nested(st.B, "B")
            assert math.isfinite(st.freivalds_rel_err), (
                f"Non-finite Freivalds residual after +-1e7 burst: {st.freivalds_rel_err}"
            )
            assert st.freivalds_rel_err < 1e-3, (
                f"Large Freivalds drift after +-1e7 burst: {st.freivalds_rel_err:.3e}"
            )
            _assert_all_finite_nested(gr.f_stat, "granger.f_stat")
            _assert_all_finite_nested(gr.partial_r2, "granger.partial_r2")
            _assert_all_finite_nested(pr.theta, "precision.theta")
            _assert_all_finite_nested(pr.pcorr, "precision.pcorr")
            _assert_all_finite_nested(ori, "orientation")
            _assert_all_finite_nested(attr.root_scores, "root_scores")


# =========================================================================
# 4. HOSTILE CAUSALITY RED-TEAM
# =========================================================================


class TestHostileCausalityRedTeam:
    """Hostile assert_causal tests with +-1e100 future poison and NaN rejection atomicity."""

    @pytest.mark.parametrize("normalize_scale", [True, False])
    @pytest.mark.parametrize("nonlinear_hinge", [False, True])
    @pytest.mark.parametrize("poison_val", [1e100, -1e100])
    def test_assert_causal_plus_minus_1e100_future_poison(
        self, normalize_scale: bool, nonlinear_hinge: bool, poison_val: float
    ) -> None:
        from causalsieve import CausalSieve

        T, D, lag = 180, 5, 2
        clean = _make_clean_var_rows(T, D, seed=4100)

        for probe in (0, 2, 35, 110, T - 2):
            poisoned = [list(r) for r in clean]
            for t in range(probe + 1, T):
                poisoned[t] = [poison_val * (1.0 if (j % 2 == 0) else -1.0) for j in range(D)]

            with CausalSieve(
                dim=D, lag=lag, normalize_scale=normalize_scale, nonlinear_hinge=nonlinear_hinge
            ) as e_clean, CausalSieve(
                dim=D, lag=lag, normalize_scale=normalize_scale, nonlinear_hinge=nonlinear_hinge
            ) as e_poison:
                # Batch path
                r_c = e_clean.push_batch(clean, return_results=True, return_exo_z=True, return_innov=True)
                r_p = e_poison.push_batch(poisoned, return_results=True, return_exo_z=True, return_innov=True)
                assert r_c is not None and r_p is not None
                for t in range(probe + 1):
                    assert r_c[t].tick == r_p[t].tick
                    assert r_c[t].mahalanobis_sq == r_p[t].mahalanobis_sq
                    assert r_c[t].e_value == r_p[t].e_value
                    assert r_c[t].shock_weight == r_p[t].shock_weight
                    assert r_c[t].max_exo_z == r_p[t].max_exo_z
                    assert r_c[t].top_root_channel == r_p[t].top_root_channel
                    assert r_c[t].alarm == r_p[t].alarm
                    assert r_c[t].exo_z == r_p[t].exo_z
                    assert r_c[t].innov == r_p[t].innov

    def test_nan_rejection_at_tick_t_leaves_tick_t_plus_1_bit_identical(self) -> None:
        from causalsieve import CausalSieve

        D, lag = 6, 2
        warm = _make_clean_var_rows(110, D, seed=4200)
        tail = _make_clean_var_rows(20, D, seed=4201)

        with CausalSieve(dim=D, lag=lag, normalize_scale=True) as attacked, CausalSieve(
            dim=D, lag=lag, normalize_scale=True
        ) as twin:
            attacked.push_batch(warm, return_results=False)
            twin.push_batch(warm, return_results=False)
            assert attacked.ticks == 110 and twin.ticks == 110

            # Attack single-tick push with NaN, +Inf, -Inf at tick t
            for bad_val in (float("nan"), float("inf"), float("-inf")):
                bad_row = list(tail[0])
                bad_row[2] = bad_val
                with pytest.raises(ValueError):
                    attacked.push(bad_row)
                assert attacked.ticks == 110

            # Also attack push_batch with NaN at future tick inside batch
            bad_batch = [list(r) for r in tail[:10]]
            bad_batch[7][4] = float("nan")
            with pytest.raises(ValueError):
                attacked.push_batch(bad_batch, return_results=True)
            assert attacked.ticks == 110

            # Tick t+1 (and next 19 ticks) must be BIT-IDENTICAL to untouched twin!
            for row in tail:
                ra = attacked.push(row, return_vectors=True)
                rt = twin.push(row, return_vectors=True)
                assert ra == rt

            assert attacked.inspect_state().B == twin.inspect_state().B
            assert attacked.granger_matrix().f_stat == twin.granger_matrix().f_stat
            assert attacked.precision_matrix().theta == twin.precision_matrix().theta
            assert attacked.contemporaneous_orientation() == twin.contemporaneous_orientation()
            assert attacked.root_attribution() == twin.root_attribution()


# =========================================================================
# 5. HOSTILE NATIVE CLI (`build/causalsieve-bin`) FUZZING
# =========================================================================


class TestHostileNativeCLIFuzzing:
    """Fuzz build/causalsieve-bin across valid, piped, edge-case, and malformed inputs."""

    CLI_BIN = BUILD_DIR / "causalsieve-bin"

    def _run_cli(
        self, args: list[str], *, stdin_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert self.CLI_BIN.is_file(), f"Missing native CLI binary: {self.CLI_BIN}"
        proc = subprocess.run(
            [str(self.CLI_BIN), *args],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        # Never crash by signal (SIGSEGV=-11, SIGABRT=-6, SIGFPE=-8, etc.)
        assert proc.returncode >= 0, (
            f"Native CLI crashed with signal {-proc.returncode}! args={args}, stderr={proc.stderr}"
        )
        return proc

    def test_valid_csv_stdin_and_compact(self, tmp_path: Path) -> None:
        rows = _make_clean_var_rows(160, 4, seed=5100, shock_at=120)
        lines = ["tick,db_cpu,api_lat,web_lat,cache_hit"]
        for t, r in enumerate(rows):
            lines.append(f"{t}," + ",".join(f"{v:.8f}" for v in r))
        csv_text = "\n".join(lines) + "\n"
        csv_file = tmp_path / "valid.csv"
        csv_file.write_text(csv_text, encoding="utf-8")

        # File mode (JSON)
        p_file = self._run_cli([str(csv_file)])
        assert p_file.returncode == 0, p_file.stderr
        parsed = json.loads(p_file.stdout)
        assert parsed["dim"] == 4 and parsed["ticks"] == 160
        assert parsed["top_root_causes"][0]["name"] == "db_cpu"

        # Stdin pipe mode with explicit `-`
        p_stdin = self._run_cli(["-"], stdin_text=csv_text)
        assert p_stdin.returncode == 0, p_stdin.stderr
        parsed_stdin = json.loads(p_stdin.stdout)
        assert parsed_stdin["ticks"] == 160
        assert parsed_stdin["top_root_causes"][0]["name"] == "db_cpu"

        # Compact mode
        p_compact = self._run_cli(["--compact", str(csv_file)])
        assert p_compact.returncode == 0, p_compact.stderr
        out_lines = [ln for ln in p_compact.stdout.splitlines() if ln.strip()]
        assert len(out_lines) == 1
        assert "ROOT1=db_cpu(" in out_lines[0]

    def test_empty_header_only_and_one_row_csvs(self, tmp_path: Path) -> None:
        # Empty file
        empty_f = tmp_path / "empty.csv"
        empty_f.write_text("", encoding="utf-8")
        p_empty = self._run_cli([str(empty_f)])
        assert p_empty.returncode == 1

        # Header-only CSV
        hdr_f = tmp_path / "header_only.csv"
        hdr_f.write_text("tick,a,b,c\n", encoding="utf-8")
        p_hdr = self._run_cli([str(hdr_f)])
        assert p_hdr.returncode == 1

        # 1-row CSV (with header) — JSON and --compact
        one_f = tmp_path / "one_row.csv"
        one_f.write_text("tick,a,b,c\n0,1.5,2.5,3.5\n", encoding="utf-8")
        p_one = self._run_cli([str(one_f)])
        assert p_one.returncode == 0, p_one.stderr
        j_one = json.loads(p_one.stdout)
        assert j_one["dim"] == 3 and j_one["ticks"] == 1

        p_one_c = self._run_cli(["--compact", str(one_f)])
        assert p_one_c.returncode == 0, p_one_c.stderr
        assert "T=1" in p_one_c.stdout

        # 1-row CSV (no header)
        one_nh = tmp_path / "one_no_header.csv"
        one_nh.write_text("1.0,2.0,3.0\n", encoding="utf-8")
        p_one_nh = self._run_cli(["--no-header", str(one_nh)])
        assert p_one_nh.returncode == 0, p_one_nh.stderr
        assert json.loads(p_one_nh.stdout)["ticks"] == 1

    def test_malformed_tokens_nan_rows_and_hostile_flags(self, tmp_path: Path) -> None:
        # Mix of valid rows, malformed non-numeric tokens, short rows, NaN, and Inf
        mixed_f = tmp_path / "mixed_dirty.csv"
        mixed_lines = [
            "tick,x0,x1,x2",
            "0,1.0,2.0,3.0",
            "1,not_a_num,2.1,3.1",
            "2,NaN,2.2,3.2",
            "3,1.3,inf,3.3",
            "4,1.4,-inf,3.4",
            "5,1.5",
            "6,1.6,2.6,3.6",
            "7,1.7,2.7,3.7",
            "8,1.8,2.8,3.8",
        ]
        mixed_f.write_text("\n".join(mixed_lines) + "\n", encoding="utf-8")
        p_mixed = self._run_cli([str(mixed_f)])
        assert p_mixed.returncode == 0, p_mixed.stderr
        # strict JSON parse rejects bare NaN/Infinity in JSON!
        j_mixed = json.loads(p_mixed.stdout, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        assert j_mixed["ticks"] == 4

        # All-NaN / all-garbage CSV -> clean exit 1, zero crash
        all_nan = tmp_path / "all_nan.csv"
        all_nan.write_text("a,b\nNaN,NaN\nfoo,bar\ninf,-inf\n", encoding="utf-8")
        p_nan = self._run_cli([str(all_nan)])
        assert p_nan.returncode == 1

        # Unknown option, missing file, invalid lag
        assert self._run_cli(["--nonexistent-flag"]).returncode == 2
        assert self._run_cli([str(tmp_path / "does_not_exist.csv")]).returncode == 1
        assert self._run_cli(["--lag", "999", str(mixed_f)]).returncode == 1
        assert self._run_cli(["--lag", "0", str(mixed_f)]).returncode == 1

    def test_tier2_high_dim_top_k_zero_compact_no_segfault(self, tmp_path: Path) -> None:
        """Tier-2 (D=130 > 128) with --top-k 0 --compact must NOT dereference OOB / SIGSEGV."""
        wide_f = tmp_path / "wide_130.csv"
        hdr = ",".join(f"c{i}" for i in range(130))
        row = ",".join("1.0" for _ in range(130))
        wide_f.write_text(hdr + "\n" + "\n".join(row for _ in range(10)) + "\n", encoding="utf-8")
        # _run_cli asserts returncode >= 0 (no SIGSEGV / SIGABRT)
        proc = self._run_cli(["--top-k", "0", "--compact", str(wide_f)])
        assert proc.returncode in (0, 1, 2)


# =========================================================================
# 6. HOSTILE MCP SECURITY & PROTOCOL RED-TEAM
# =========================================================================


class TestHostileMCPSecurityRedTeam:
    """Attack the live bare-python MCP stdio server with security & protocol payloads."""

    def test_mcp_security_path_traversal_symlink_and_protocol_fuzz(self, tmp_path: Path) -> None:
        bare_py = _find_bare_system_python()
        env = _bare_python_env()

        allowed_dir = tmp_path / "allowed_root"
        outside_dir = tmp_path / "secret_outside"
        allowed_dir.mkdir()
        outside_dir.mkdir()

        # Valid CSV inside allowed root
        valid_csv = allowed_dir / "legit.csv"
        rows = _make_clean_var_rows(120, 4, seed=6100, shock_at=90)
        with valid_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tick", "db", "api", "web", "cache"])
            for t, r in enumerate(rows):
                w.writerow([t, *[f"{v:.8f}" for v in r]])

        # Secret file outside allowed root + symlink inside allowed root pointing outside
        outside_csv = outside_dir / "secret.csv"
        outside_csv.write_text("tick,secret_metric\n0,999.0\n1,999.0\n", encoding="utf-8")

        symlink_outside = allowed_dir / "escape_link.csv"
        symlink_outside.symlink_to(outside_csv)

        symlink_etc = allowed_dir / "etc_passwd_link.csv"
        symlink_etc.symlink_to(Path("/etc/passwd"))

        proc = subprocess.Popen(
            [bare_py, "-m", "causalsieve.mcp_server", "--allowed-root", str(allowed_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )

        def send_raw(raw_line: str) -> dict[str, Any]:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(raw_line + "\n")
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            assert resp_line, (
                f"MCP server crashed/EOF on input {raw_line!r}; "
                f"stderr={proc.stderr.read() if proc.stderr else ''}"
            )
            return json.loads(resp_line)

        def rpc(msg_id: int, method: str, params: Any = None) -> dict[str, Any]:
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
            if params is not None:
                payload["params"] = params
            return send_raw(json.dumps(payload))

        def assert_rejected(resp: dict[str, Any], desc: str, forbidden_substrings: tuple[str, ...] = ()) -> None:
            is_rpc_err = "error" in resp
            is_tool_err = "result" in resp and resp["result"].get("isError") is True
            assert is_rpc_err or is_tool_err, f"Attack NOT rejected ({desc}): {resp}"
            serialized = json.dumps(resp)
            for s in forbidden_substrings:
                assert s not in serialized, f"Sensitive leak ({s!r}) in response for {desc}: {serialized}"

        try:
            # Handshake
            init_r = rpc(1, "initialize", {"protocolVersion": "2024-11-05"})
            assert init_r["result"]["serverInfo"]["name"] == "causalsieve"

            # Seed one valid live stream so channel/horizon/NaN attacks target an active stream
            seed_r = rpc(
                2,
                "tools/call",
                {
                    "name": "sieve_push_ticks",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "channel_names": ["db", "api", "web", "cache"],
                        "rows": rows[:40],
                    },
                },
            )
            assert not seed_r["result"]["isError"], seed_r

            # Attack 1: Path traversal ../../etc/passwd & relative escape from allowed_dir
            for idx, bad_path in enumerate(
                (
                    "../../etc/passwd",
                    str(allowed_dir / ".." / ".." / ".." / ".." / "etc" / "passwd"),
                    str(allowed_dir / ".." / "secret_outside" / "secret.csv"),
                    str(outside_csv),
                    str(allowed_dir / "legit.csv\x00.txt"),
                    str(allowed_dir),  # directory, not regular file
                ),
                start=10,
            ):
                r = rpc(
                    idx,
                    "tools/call",
                    {"name": "sieve_ingest_csv", "arguments": {"path": bad_path}},
                )
                assert_rejected(r, f"path_traversal:{bad_path!r}", ("root:x:0:0", "secret_metric"))

            # Attack 2: Symlink inside --allowed-root pointing outside
            for idx, sym in enumerate((symlink_outside, symlink_etc), start=20):
                r = rpc(
                    idx,
                    "tools/call",
                    {"name": "sieve_ingest_csv", "arguments": {"path": str(sym)}},
                )
                assert_rejected(r, f"symlink_escape:{sym}", ("root:x:0:0", "secret_metric"))

            # Attack 3: Malformed JSON -> JSON-RPC -32700
            for bad_json in ("{not valid json", "[1, 2,", '{"jsonrpc":"2.0"'):
                r = send_raw(bad_json)
                assert "error" in r and r["error"]["code"] == -32700, f"Bad parse resp: {r}"

            # Attack 4: Unknown JSON-RPC method -> -32601
            r_unk = rpc(30, "system/exec_arbitrary_cmd", {"cmd": "id"})
            assert "error" in r_unk and r_unk["error"]["code"] == -32601

            # Attack 5: Missing required params across tools
            missing_cases = [
                ("sieve_ingest_csv", {}),
                ("sieve_push_ticks", {"stream_id": "victim-stream"}),
                ("sieve_causal_graph", {}),
                ("sieve_counterfactual", {"stream_id": "victim-stream", "channel": 0}),
                ("sieve_recover_slice", {"stream_id": "victim-stream"}),
                ("sieve_swarm_propose", {"stream_id": "victim-stream", "agent_id": "a1"}),
            ]
            for idx, (tname, targs) in enumerate(missing_cases, start=40):
                r = rpc(idx, "tools/call", {"name": tname, "arguments": targs})
                assert_rejected(r, f"missing_params:{tname}")

            # Attack 6: Invalid channel index -1 and 99999
            for idx, bad_ch in enumerate((-1, 99999, "nonexistent_ch"), start=50):
                r_cf = rpc(
                    idx,
                    "tools/call",
                    {
                        "name": "sieve_counterfactual",
                        "arguments": {
                            "stream_id": "victim-stream",
                            "channel": bad_ch,
                            "delta": 1.0,
                        },
                    },
                )
                assert_rejected(r_cf, f"invalid_channel_cf:{bad_ch}")

                r_sl = rpc(
                    idx + 10,
                    "tools/call",
                    {
                        "name": "sieve_recover_slice",
                        "arguments": {
                            "stream_id": "victim-stream",
                            "channel": bad_ch,
                            "start_tick": 0,
                            "end_tick": 5,
                        },
                    },
                )
                assert_rejected(r_sl, f"invalid_channel_slice:{bad_ch}")

            # Attack 7: NaN / Infinity in sieve_push_ticks (both literal JSON NaN & string "NaN")
            raw_nan_msg = (
                '{"jsonrpc":"2.0","id":70,"method":"tools/call","params":{'
                '"name":"sieve_push_ticks","arguments":{"stream_id":"victim-stream",'
                '"rows":[[1.0, NaN, 2.0, 3.0]]}}}'
            )
            assert_rejected(send_raw(raw_nan_msg), "literal_NaN_push_ticks")

            raw_inf_msg = (
                '{"jsonrpc":"2.0","id":71,"method":"tools/call","params":{'
                '"name":"sieve_push_ticks","arguments":{"stream_id":"victim-stream",'
                '"rows":[[1.0, Infinity, 2.0, 3.0]]}}}'
            )
            assert_rejected(send_raw(raw_inf_msg), "literal_Inf_push_ticks")

            r_str_nan = rpc(
                72,
                "tools/call",
                {
                    "name": "sieve_push_ticks",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "rows": [[1.0, "NaN", 2.0, 3.0]],
                    },
                },
            )
            assert_rejected(r_str_nan, "string_NaN_push_ticks")

            # Attack 8: Huge horizon=100000 (on counterfactual and swarm_propose)
            r_huge_cf = rpc(
                80,
                "tools/call",
                {
                    "name": "sieve_counterfactual",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "channel": 0,
                        "delta": 1.0,
                        "horizon": 100000,
                    },
                },
            )
            assert_rejected(r_huge_cf, "huge_horizon_counterfactual")

            r_huge_sw = rpc(
                81,
                "tools/call",
                {
                    "name": "sieve_swarm_propose",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "agent_id": "attacker",
                        "candidate_channel": 0,
                        "horizon": 100000,
                    },
                },
            )
            assert_rejected(r_huge_sw, "huge_horizon_swarm")

            # Attack 9: Special device / stream_id traversal / huge slice / bool row / non-object RPC
            r_dev = rpc(
                82,
                "tools/call",
                {"name": "sieve_ingest_csv", "arguments": {"path": "/dev/null"}},
            )
            assert_rejected(r_dev, "device_file_ingest")

            r_sid = rpc(
                83,
                "tools/call",
                {
                    "name": "sieve_push_ticks",
                    "arguments": {"stream_id": "../escape", "rows": [[1.0, 2.0, 3.0, 4.0]]},
                },
            )
            assert_rejected(r_sid, "stream_id_traversal")

            r_big_sl = rpc(
                84,
                "tools/call",
                {
                    "name": "sieve_recover_slice",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "channel": 0,
                        "start_tick": 0,
                        "end_tick": 500000,
                    },
                },
            )
            assert_rejected(r_big_sl, "huge_slice_cap")

            r_bool = rpc(
                85,
                "tools/call",
                {
                    "name": "sieve_push_ticks",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "rows": [[True, False, 1.0, 2.0]],
                    },
                },
            )
            assert_rejected(r_bool, "bool_in_numeric_rows")

            r_nonobj = send_raw("12345")
            assert "error" in r_nonobj and r_nonobj["error"]["code"] == -32600

            # Post-attack liveness & integrity proof: valid operations MUST still succeed cleanly!
            r_valid_ing = rpc(
                90,
                "tools/call",
                {
                    "name": "sieve_ingest_csv",
                    "arguments": {
                        "path": str(valid_csv),
                        "stream_id": "post-attack-ok",
                        "format": "json",
                    },
                },
            )
            assert not r_valid_ing["result"]["isError"], r_valid_ing
            ok_data = json.loads(r_valid_ing["result"]["content"][0]["text"])
            assert ok_data["stream_id"] == "post-attack-ok"
            assert ok_data["root_causes"][0]["name"] == "db"

            # And victim-stream watermark must still be exact 40 (zero NaN contamination)
            r_post_push = rpc(
                91,
                "tools/call",
                {
                    "name": "sieve_push_ticks",
                    "arguments": {
                        "stream_id": "victim-stream",
                        "rows": rows[40:50],
                    },
                },
            )
            assert not r_post_push["result"]["isError"], r_post_push
            assert json.loads(r_post_push["result"]["content"][0]["text"])["watermark"] == 50
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
            assert proc.returncode == 0, f"MCP server exited non-zero: {proc.returncode}"


# =========================================================================
# 7. CLEAN-ROOM ANONYMIZATION & ASAN/UBSAN GATE
# =========================================================================


class TestCleanRoomAndSanitizerGate:
    """Verify ASan/UBSan binary passes cleanly and zero personal/private leaks exist."""

    def test_asan_ubsan_cpp_suite_zero_findings(self) -> None:
        asan_bin = BUILD_DIR / "test_core_cpp_asan"
        assert asan_bin.is_file(), f"Missing ASan/UBSan test binary: {asan_bin}"
        proc = subprocess.run(
            [str(asan_bin)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode == 0, f"ASan/UBSan suite failed (rc={proc.returncode}):\n{combined}"
        for bad_marker in (
            "AddressSanitizer",
            "LeakSanitizer",
            "UndefinedBehaviorSanitizer",
            "runtime error:",
            "ERROR: ",
        ):
            assert bad_marker not in combined, (
                f"Sanitizer marker {bad_marker!r} found in output:\n{combined}"
            )
        assert "ALL TESTS PASSED (0 failures)" in proc.stdout

    def test_clean_room_anonymization_scan(self) -> None:
        """Scan all source/doc files (excl. benchmarks/data/ & .git/) for leaked paths/secrets."""
        # Build sensitive tokens dynamically so this audit script itself stays clean if scanned.
        private_terms = [
            "millstlabs/" + "Diego",
            "millstlabs/" + "streamcompiler",
            "legion" + "edge",
            r"\b" + "Die" + "go" + r"\b",
            r"\b" + "stream" + "compiler" + r"\b",
            r"\b" + "anti" + "gravity" + r"\b",
        ]
        patterns: dict[str, re.Pattern[str]] = {
            "macos_user_path": re.compile(r"/Users/[A-Za-z0-9._-]+"),
            "linux_home_path": re.compile(r"/home/[A-Za-z0-9._-]+"),
            "windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+"),
            "personal_email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "secret_token": re.compile(
                r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9]{20,}|"
                r"AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY)"
            ),
            "private_workspace_ref": re.compile("|".join(private_terms), re.IGNORECASE),
        }

        skip_dirs = {
            ".git",
            "data",
            "build",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }
        self_path = Path(__file__).resolve()
        findings: list[str] = []

        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.resolve() == self_path:
                    continue
                if p.suffix in {".so", ".o", ".a", ".pyc", ".png", ".jpg", ".parquet", ".gz"}:
                    continue
                try:
                    text = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                rel = p.relative_to(REPO_ROOT)
                for lineno, line in enumerate(text.splitlines(), 1):
                    for pname, rx in patterns.items():
                        for m in rx.finditer(line):
                            findings.append(
                                f"[{pname}] {rel}:{lineno}: matched {m.group(0)!r} -> {line.strip()[:120]}"
                            )

        assert not findings, (
            f"Clean-room anonymization scan found {len(findings)} leak(s):\n"
            + "\n".join(findings)
        )

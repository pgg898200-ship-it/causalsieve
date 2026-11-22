"""Caveman Causal Digest, Swing-Filter PLA, lossless recovery, MCP stdio server, and CausalBlackboard tests.

Covers Hypothesis H5 and agent-native multi-agent coordination guarantees:
  1. >= 95% token reduction vs raw CSV/JSON telemetry.
  2. 0.0 FP64 bit-exact slice recovery via recover_slice() and slice:// handles.
  3. Exact L_inf <= max_abs_err guarantee at EVERY interpolated tick of Swing-Filter PLA knots.
  4. Live subprocess MCP JSON-RPC 2.0 stdio server across initialize, tools/list,
     sieve_ingest_csv, sieve_causal_graph, sieve_counterfactual, sieve_recover_slice,
     and sieve_swarm_propose.
  5. Multi-threaded CausalBlackboard concurrent live ingest, versioned snapshots,
     append-only non-destructive hypothesis scoring, and anti-ballot-stuffing consensus.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _make_cascade_dataset(
    T: int = 1200, D: int = 16, shock_start: int = 1050, seed: int = 40401
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    names = [f"svc_{i}" for i in range(D)]
    burn = 100
    X = np.zeros((T + burn, D), dtype=np.float64)
    eps = rng.standard_normal((T + burn, D))
    eps[:, 1] *= 0.35
    eps[:, 2] *= 0.35
    eps[:, 3] *= 0.30
    for t in range(2, T + burn):
        X[t] = 0.25 * X[t - 1] - 0.08 * X[t - 2] + eps[t]
        u = 5.5 if (t - burn) >= shock_start else 0.0
        X[t, 0] = 0.25 * X[t - 1, 0] + eps[t, 0] + u
        X[t, 1] = 0.65 * X[t - 1, 1] + 0.68 * X[t - 1, 0] + eps[t, 1]
        X[t, 2] = 0.68 * X[t - 1, 2] + 0.68 * X[t - 1, 1] + eps[t, 2]
        X[t, 3] = 0.72 * X[t - 1, 3] + 0.72 * X[t - 1, 2] + eps[t, 3]
    return np.ascontiguousarray(X[burn:]), names


def _write_csv(path: Path, X: np.ndarray, names: list[str]) -> int:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tick", *names])
        for t in range(X.shape[0]):
            w.writerow([t, *[f"{v:.17g}" for v in X[t]]])
    return path.stat().st_size


def test_caveman_digest_token_reduction_lossless_recovery_and_pla_bound(tmp_path: Path) -> None:
    """Verify >= 95% token reduction, 0.0 FP64 slice error, and exact L_inf PLA knot error bound."""
    from causalsieve._native import CausalSieve
    from causalsieve.digest import (
        StreamStore,
        build_digest,
        estimate_tokens,
        recover_handle,
        recover_slice,
    )

    T, D = 1200, 16
    X, names = _make_cascade_dataset(T=T, D=D, shock_start=1050, seed=40401)
    csv_path = tmp_path / "telemetry.csv"
    raw_csv_bytes = _write_csv(csv_path, X, names)
    raw_csv_text = csv_path.read_text(encoding="utf-8")
    raw_json_text = json.dumps({"columns": names, "rows": X.tolist()}, separators=(",", ":"))

    store = StreamStore()
    with CausalSieve(dim=D, lag=2, channel_names=names, raw_ring_capacity=4096) as sieve:
        sieve.push_batch(X, return_results=True)
        digest = build_digest(
            sieve,
            top_k=5,
            max_edges=8,
            max_err=0.05,
            raw_bytes=raw_csv_bytes,
            store=store,
        )

        md = digest.to_markdown()
        js = digest.to_json()

        raw_csv_tokens = estimate_tokens(raw_csv_text)
        raw_json_tokens = estimate_tokens(raw_json_text)
        md_tokens = estimate_tokens(md)
        js_tokens = estimate_tokens(js)

        red_vs_csv = 100.0 * (1.0 - md_tokens / raw_csv_tokens)
        red_vs_json = 100.0 * (1.0 - md_tokens / raw_json_tokens)
        assert digest.reduction_pct >= 95.0, f"Expected >= 95% reduction, got {digest.reduction_pct}%"
        assert red_vs_csv >= 95.0, f"Markdown reduction vs raw CSV {red_vs_csv:.2f}% < 95%"
        assert red_vs_json >= 95.0, f"Markdown reduction vs raw JSON {red_vs_json:.2f}% < 95%"
        assert js_tokens < 0.05 * raw_json_tokens, "JSON digest exceeded 5% of raw JSON tokens"
        assert digest.root_causes[0].name == "svc_0"

        # 100% lossless FP64 exact recovery via recover_slice() AND slice:// handle
        for ch_idx, ch_name in ((0, "svc_0"), (3, "svc_3"), (15, "svc_15")):
            sl = recover_slice(digest.stream_id, ch_name, 100, 1150, store=store)
            assert sl.actual_start == 100
            assert len(sl.values) == 1050
            np.testing.assert_array_equal(
                np.asarray(sl.values, dtype=np.float64), X[100:1150, ch_idx]
            )

        assert len(digest.recovery_handles) >= 1
        h_slice = recover_handle(digest.recovery_handles[0], store=store)
        ch_idx = sieve.resolve_channel(h_slice.channel_name)
        expected = X[h_slice.actual_start : h_slice.actual_start + len(h_slice.values), ch_idx]
        np.testing.assert_array_equal(np.asarray(h_slice.values, dtype=np.float64), expected)

        # Exact L_inf Swing-Filter PLA error guarantee across multiple tolerances
        for max_err in (0.01, 0.05, 0.25):
            pla = sieve.compress_pla(0, 900, 1200, max_abs_err=max_err)
            assert len(pla.knots) >= 2
            knot_ticks = np.array([k[0] for k in pla.knots], dtype=np.float64)
            knot_vals = np.array([k[1] for k in pla.knots], dtype=np.float64)
            assert knot_ticks[0] == 900 and knot_ticks[-1] == 1199
            assert np.all(np.diff(knot_ticks) > 0), "PLA knot ticks must be strictly increasing"

            eval_ticks = np.arange(900, 1200, dtype=np.float64)
            interp_vals = np.interp(eval_ticks, knot_ticks, knot_vals)
            raw_vals = X[900:1200, 0]
            max_observed_err = float(np.max(np.abs(interp_vals - raw_vals)))
            assert max_observed_err <= max_err + 1e-12, (
                f"Swing-Filter PLA violated L_inf bound: observed={max_observed_err:.6g} > {max_err}"
            )
            if max_err >= 0.25:
                assert len(pla.knots) < 300, "Expected PLA to compress 300 ticks into fewer knots"


def test_mcp_stdio_server_full_lifecycle(tmp_path: Path) -> None:
    """Spawn python -m causalsieve.mcp_server and verify full JSON-RPC 2.0 tool workflow."""
    X, names = _make_cascade_dataset(T=600, D=10, shock_start=500, seed=40402)
    csv_path = tmp_path / "incident.csv"
    _write_csv(csv_path, X, names)

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PYTHON_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(PYTHON_DIR)

    proc = subprocess.Popen(
        [sys.executable, "-m", "causalsieve.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    def rpc(req_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, f"MCP server closed stdout unexpectedly; stderr={proc.stderr.read() if proc.stderr else ''}"
        resp = json.loads(line)
        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == req_id
        return resp

    def notify(method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    try:
        # 1. initialize + initialized notification
        init_resp = rpc(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest-verifier", "version": "1.0"},
            },
        )
        assert "result" in init_resp
        assert init_resp["result"]["serverInfo"]["name"] == "causalsieve"
        notify("notifications/initialized")

        # 2. tools/list
        list_resp = rpc(2, "tools/list", {})
        tools = {t["name"] for t in list_resp["result"]["tools"]}
        expected_tools = {
            "sieve_ingest_csv",
            "sieve_push_ticks",
            "sieve_causal_graph",
            "sieve_counterfactual",
            "sieve_recover_slice",
            "sieve_swarm_propose",
        }
        assert expected_tools.issubset(tools), f"Missing MCP tools: {expected_tools - tools}"

        # 3. sieve_ingest_csv
        ing_resp = rpc(
            3,
            "tools/call",
            {
                "name": "sieve_ingest_csv",
                "arguments": {
                    "path": str(csv_path),
                    "stream_id": "mcp-inc-1",
                    "lag": 2,
                    "top_k": 5,
                    "format": "json",
                },
            },
        )
        assert not ing_resp["result"].get("isError", False), ing_resp
        ing_payload = json.loads(ing_resp["result"]["content"][0]["text"])
        assert ing_payload["stream_id"] == "mcp-inc-1"
        assert ing_payload["root_causes"][0]["name"] == "svc_0"
        assert ing_payload["token_stats"]["reduction_pct"] >= 90.0

        # 4. sieve_causal_graph
        g_resp = rpc(
            4,
            "tools/call",
            {
                "name": "sieve_causal_graph",
                "arguments": {"stream_id": "mcp-inc-1", "min_granger_f": 3.0},
            },
        )
        assert not g_resp["result"].get("isError", False), g_resp
        g_payload = json.loads(g_resp["result"]["content"][0]["text"])
        lagged_pairs = {(e["src"], e["dst"]) for e in g_payload["lagged_granger_edges"]}
        assert ("svc_0", "svc_1") in lagged_pairs, f"Expected svc_0->svc_1 in {lagged_pairs}"

        # 5. sieve_counterfactual
        cf_resp = rpc(
            5,
            "tools/call",
            {
                "name": "sieve_counterfactual",
                "arguments": {
                    "stream_id": "mcp-inc-1",
                    "channel": "svc_0",
                    "delta": 5.0,
                    "horizon": 8,
                },
            },
        )
        assert not cf_resp["result"].get("isError", False), cf_resp
        cf_payload = json.loads(cf_resp["result"]["content"][0]["text"])
        assert len(cf_payload["trajectory"]) == 8
        svc0_idx = cf_payload["channel_names"].index("svc_0")
        assert cf_payload["trajectory"][0][svc0_idx] == pytest.approx(5.0, abs=1e-6)

        # 6. sieve_recover_slice -> exact 0.0 FP64 error!
        rec_resp = rpc(
            6,
            "tools/call",
            {
                "name": "sieve_recover_slice",
                "arguments": {
                    "stream_id": "mcp-inc-1",
                    "channel": "svc_0",
                    "start_tick": 495,
                    "end_tick": 525,
                },
            },
        )
        assert not rec_resp["result"].get("isError", False), rec_resp
        rec_payload = json.loads(rec_resp["result"]["content"][0]["text"])
        recovered_vals = np.asarray(rec_payload["values"], dtype=np.float64)
        np.testing.assert_array_equal(recovered_vals, X[495:525, 0])

        # 7. sieve_swarm_propose (non-destructive multi-agent scoring)
        p1 = rpc(
            7,
            "tools/call",
            {
                "name": "sieve_swarm_propose",
                "arguments": {
                    "stream_id": "mcp-inc-1",
                    "agent_id": "agent-root",
                    "candidate_channel": "svc_0",
                    "delta": 5.0,
                    "horizon": 8,
                    "confidence": 0.9,
                    "rationale": "Earliest exogenous shock with strong outflow",
                },
            },
        )
        p2 = rpc(
            8,
            "tools/call",
            {
                "name": "sieve_swarm_propose",
                "arguments": {
                    "stream_id": "mcp-inc-1",
                    "agent_id": "agent-victim",
                    "candidate_channel": "svc_3",
                    "delta": 5.0,
                    "horizon": 8,
                    "confidence": 0.9,
                    "rationale": "Largest raw z-score",
                },
            },
        )
        assert not p1["result"].get("isError", False) and not p2["result"].get("isError", False)
        p1_data = json.loads(p1["result"]["content"][0]["text"])
        p2_data = json.loads(p2["result"]["content"][0]["text"])
        assert p1_data["hypothesis"]["objective_score"] > p2_data["hypothesis"]["objective_score"]
        assert p2_data["consensus"]["total_hypotheses"] == 2
        assert p2_data["consensus"]["winner_name"] == "svc_0"

        # 8. Unknown method returns JSON-RPC -32601 cleanly
        err_resp = rpc(9, "nonexistent/method", {})
        assert "error" in err_resp and err_resp["error"]["code"] == -32601
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=10)
        assert proc.returncode == 0, f"MCP server exited with code {proc.returncode}"


def test_causal_blackboard_concurrent_snapshots_and_nondestructive_scoring() -> None:
    """Concurrent live ingest + 4 parallel agents taking snapshots & submitting hypotheses."""
    from causalsieve.swarm import CausalBlackboard

    X, names = _make_cascade_dataset(T=650, D=12, shock_start=500, seed=40403)
    bb = CausalBlackboard()
    try:
        sid = bb.create_stream("live-prod", dim=12, lag=2, channel_names=names)
        # Seed baseline ticks before spawning concurrent swarm
        bb.push_batch(sid, X[:520], return_results=False)
        snap_before = bb.snapshot(sid)
        assert snap_before.as_of_tick == 520

        errors: list[BaseException] = []
        observed_watermarks: list[int] = []
        wm_lock = threading.Lock()

        def writer_worker() -> None:
            try:
                for start in range(520, 650, 10):
                    bb.push_batch(sid, X[start : start + 10], return_results=False)
                    bb.snapshot(sid)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def investigator_worker(agent_idx: int) -> None:
            try:
                agent_id = f"agent-{agent_idx}"
                prev_wm = 0
                for step in range(6):
                    wm = bb.watermark(sid)
                    assert wm >= prev_wm, f"Non-monotonic watermark: {wm} < {prev_wm}"
                    prev_wm = wm
                    with wm_lock:
                        observed_watermarks.append(wm)

                    snap = bb.snapshot(sid)
                    assert snap.as_of_tick >= 520
                    assert snap.stream_id == sid

                    # Agent 0 proposes true root; Agents 1..3 propose victims/decoys (ballot stuffing test)
                    cand = "svc_0" if agent_idx == 0 else f"svc_{(agent_idx % 3) + 1}"
                    rec = bb.verify_hypothesis(
                        sid,
                        agent_id=agent_id,
                        candidate_channel=cand,
                        delta=5.0,
                        horizon=8,
                        confidence=0.85,
                        rationale=f"step {step}",
                    )
                    assert 0.0 <= rec.objective_score <= 1.0
                    sl = bb.recover_slice(sid, "svc_0", 500, 520)
                    np.testing.assert_array_equal(np.asarray(sl.values, dtype=np.float64), X[500:520, 0])
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer_worker)]
        threads.extend(threading.Thread(target=investigator_worker, args=(i,)) for i in range(4))
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)
            assert not th.is_alive(), "Concurrent blackboard thread deadlocked"

        assert not errors, f"Concurrent blackboard raised exceptions: {errors}"
        assert bb.watermark(sid) == 650

        # Non-destructive append-only log: all 4 * 6 = 24 hypotheses must be preserved intact!
        all_hyps = bb.hypotheses(sid)
        assert len(all_hyps) == 24
        assert len({h.hypothesis_id for h in all_hyps}) == 24

        # Retained earlier snapshot at tick 520 must still be immutable and retrievable
        replayed_520 = bb.snapshot(sid, as_of_tick=520)
        assert replayed_520.as_of_tick == 520
        assert replayed_520.snapshot_version == snap_before.snapshot_version

        # Objective scoring + anti-ballot-stuffing consensus must still identify svc_0 as winner
        cons = bb.consensus(sid)
        assert cons.total_hypotheses == 24
        assert len(cons.participating_agents) == 4
        assert cons.winner_name == "svc_0", f"Consensus failed: {cons.to_dict()}"
        assert cons.agent_weights["agent-0"] > cons.agent_weights["agent-3"]
    finally:
        bb.close()

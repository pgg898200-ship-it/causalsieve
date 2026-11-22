"""Fast (<20 ms startup) zero-dependency CLI for CausalSieve.

All heavy modules (``ctypes``, ``_native``, ``digest``, ``mcp_server``) are
lazily imported only inside the invoked subcommand handler.
"""

from __future__ import annotations

import sys

_TYPE_CHECKING = False
if _TYPE_CHECKING:
    import argparse

_TOP_HELP = """usage: causalsieve [-h] {rca,digest,slice,counterfactual,watch,info,mcp} ...

Sub-millisecond SIMD online multivariate causal discovery & lossless-recoverable stream digests.

positional arguments:
  {rca,digest,slice,counterfactual,watch,info,mcp}
    rca                 Run causal root-cause analysis on a CSV telemetry file.
    digest              Emit a token-budgeted Caveman Causal Digest with lossless recovery handles.
    slice               Recover 100% lossless exact FP64 raw values for a channel tick range.
    counterfactual      Simulate exact h-step counterfactual do(channel += delta) propagation.
    watch               Stream CSV or JSONL telemetry line-by-line with real-time Ville E-process alerts.
    info                Show CausalSieve version, active SIMD ISA, and native library path.
    mcp                 Launch the zero-dependency JSON-RPC 2.0 MCP stdio server.

options:
  -h, --help            show this help message and exit
"""


def _cmd_rca(ns: argparse.Namespace) -> int:
    import json

    from causalsieve._native import CausalSieve, analyze_csv

    with analyze_csv(
        ns.file,
        lag=ns.lag,
        forgetting=ns.forgetting,
        ridge=ns.ridge,
        nonlinear_hinge=ns.hinge,
        normalize_scale=not ns.no_normalize,
        use_screened=False if ns.no_screened else None,
        top_k=ns.top_k,
    ) as res:
        if res.screened and res.screened_result is not None:
            top_rows: list[dict[str, object]] = [
                {
                    "rank": i + 1,
                    "channel": ch,
                    "name": name,
                    "score": round(sc, 4),
                }
                for i, (ch, name, sc) in enumerate(
                    zip(
                        res.screened_result.top_channels,
                        res.screened_result.top_names,
                        res.screened_result.top_scores,
                        strict=False,
                    )
                )
            ]
            payload: dict[str, object] = {
                "path": res.path,
                "mode": "screened_tier2",
                "ticks": res.n_rows,
                "dim": len(res.channel_names),
                "top_root_causes": top_rows,
            }
            if ns.format == "json":
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"# CausalSieve Tier-2 Screened RCA ({res.n_rows} ticks, {len(res.channel_names)} ch)"
                )
                print("| rank | channel | score |")
                print("|---:|---|---:|")
                for row in top_rows:
                    print(f"| {row['rank']} | {row['name']} | {row['score']:.4f} |")
            return 0

        assert isinstance(res.sieve, CausalSieve)
        dig = res.digest(
            top_k=ns.top_k,
            lookback=ns.lookback,
            max_err=ns.max_err,
        )
        if ns.format == "json":
            print(dig.to_json(indent=2))
        else:
            print(dig.to_markdown())
    return 0


def _cmd_digest(ns: argparse.Namespace) -> int:
    from causalsieve._native import analyze_csv

    with analyze_csv(
        ns.file,
        lag=ns.lag,
        forgetting=ns.forgetting,
        nonlinear_hinge=ns.hinge,
        use_screened=False,
    ) as res:
        dig = res.digest(
            top_k=ns.top_k,
            max_err=ns.max_err,
            tokens_budget=ns.tokens_budget,
        )
        if ns.format == "json":
            print(dig.to_json(indent=2))
        else:
            print(dig.to_markdown())
    return 0


def _cmd_slice(ns: argparse.Namespace) -> int:
    import json

    from causalsieve._native import CausalSieve, analyze_csv

    ring_cap = max(8192, ns.end + 64)
    with analyze_csv(
        ns.file,
        lag=1,
        raw_ring_capacity=ring_cap,
        use_screened=False,
    ) as res:
        assert isinstance(res.sieve, CausalSieve)
        sl = res.sieve.get_raw_slice(ns.channel, ns.start, ns.end)
        if ns.format == "json":
            print(json.dumps(sl.to_dict(), indent=2))
        elif ns.format == "csv":
            print(f"tick,{sl.channel_name}")
            for t, v in zip(sl.ticks, sl.values, strict=True):
                print(f"{t},{v!r}")
        else:
            print(
                f"channel={sl.channel_name} (idx={sl.channel}) range=[{sl.actual_start}..{sl.end_tick}) count={len(sl.values)}"
            )
            for t, v in zip(sl.ticks, sl.values, strict=True):
                print(f"  t={t}: {v!r}")
    return 0


def _cmd_counterfactual(ns: argparse.Namespace) -> int:
    import json

    from causalsieve._native import CausalSieve, analyze_csv

    with analyze_csv(
        ns.file,
        lag=ns.lag,
        nonlinear_hinge=ns.hinge,
        normalize_scale=not ns.no_normalize,
        use_screened=False,
    ) as res:
        assert isinstance(res.sieve, CausalSieve)
        ch_idx = res.sieve.resolve_channel(ns.channel)
        ch_name = res.sieve.channel_names[ch_idx]
        traj = res.sieve.counterfactual_do(ch_idx, ns.delta, horizon=ns.horizon)
        names = res.sieve.channel_names

        if ns.format == "json":
            print(
                json.dumps(
                    {
                        "channel": ch_name,
                        "channel_index": ch_idx,
                        "delta": ns.delta,
                        "horizon": ns.horizon,
                        "channel_names": names,
                        "trajectory": traj,
                    },
                    indent=2,
                )
            )
        else:
            print(f"# Counterfactual do({ch_name} += {ns.delta:g}) horizon={ns.horizon}")
            header = "| step | " + " | ".join(names) + " |"
            sep = "|---:|" + "|".join("---:" for _ in names) + "|"
            print(header)
            print(sep)
            for h, row in enumerate(traj):
                cells = " | ".join(f"{v:+.4f}" for v in row)
                print(f"| {h} | {cells} |")
    return 0


def _cmd_info(ns: argparse.Namespace) -> int:
    import json

    from causalsieve import __version__
    from causalsieve._native import (
        CAUSALSIEVE_ABI_VERSION,
        CausalSieveLibraryError,
        library_path,
        simd_isa,
    )

    isa: str
    lib_p: str | None
    status = "ready"
    try:
        isa = simd_isa()
        lib_p = library_path()
    except CausalSieveLibraryError as exc:
        isa = "unavailable"
        lib_p = None
        status = f"not_built ({exc.args[0].splitlines()[0]})"

    info: dict[str, object] = {
        "package": "causalsieve",
        "version": __version__,
        "abi_version": CAUSALSIEVE_ABI_VERSION,
        "python": sys.version.split()[0],
        "simd_isa": isa,
        "library_path": lib_p,
        "status": status,
    }
    if ns.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"causalsieve {info['version']} (ABI v{info['abi_version']})")
        print(f"  python:       {info['python']}")
        print(f"  simd_isa:     {info['simd_isa']}")
        print(f"  library_path: {info['library_path'] or '(not compiled yet)'}")
        print(f"  status:       {info['status']}")
    return 0


def _cmd_mcp(ns: argparse.Namespace) -> int:
    from causalsieve.mcp_server import MCPServer

    server = MCPServer(allowed_roots=ns.allowed_root, max_csv_bytes=ns.max_csv_bytes)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


def _cmd_watch(ns: argparse.Namespace) -> int:
    import contextlib
    import csv
    import json
    import math
    from typing import Any, TextIO

    from causalsieve._native import _TIMESTAMP_HEADER_NAMES, CausalSieve

    if ns.every < 0:
        raise ValueError(f"--every must be >= 0, got {ns.every}")
    if ns.top_k < 1:
        raise ValueError(f"--top-k must be >= 1, got {ns.top_k}")

    def _parse_finite_num(val: Any) -> float | None:
        if val is None or isinstance(val, (bool, dict, list, tuple, set)):
            return None
        if isinstance(val, (int, float)):
            f = float(val)
            return f if math.isfinite(f) else None
        return None

    def _parse_csv_float(cell: str) -> float | None:
        s = cell.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None

    def _ranked_top_k(sieve_obj: CausalSieve) -> list[dict[str, Any]]:
        attr = sieve_obj.root_attribution()
        eff_k = max(1, min(int(ns.top_k), len(attr.items)))
        return [
            {
                "rank": item.rank,
                "channel": item.channel,
                "name": item.name,
                "root_score": round(item.root_score, 4),
                "peak_exo_z": round(item.peak_exo_z, 4),
                "raw_z": round(item.raw_z, 4),
                "onset_tick": item.onset_tick,
            }
            for item in attr.items[:eff_k]
        ]

    use_stdin = ns.file in (None, "-")

    sieve: CausalSieve | None = None
    alarm_count = 0
    first_alarm_tick: int | None = None
    max_e_value = 1.0

    def _emit_event(
        kind: str,
        tick: int,
        e_value: float,
        max_exo_z: float,
        top_ch: int,
        top_name: str | None,
        ranked: list[dict[str, Any]],
        ts_val: Any = None,
    ) -> None:
        if ns.format == "jsonl":
            payload: dict[str, Any] = {
                "event": kind,
                "tick": tick,
                "e_value": round(e_value, 4),
                "max_exo_z": round(max_exo_z, 4),
                "top_root_channel": top_ch,
                "top_root_name": top_name,
                "ranked_top_k": ranked,
            }
            if ts_val is not None:
                payload["timestamp"] = ts_val
            print(json.dumps(payload), flush=True)
        else:
            tag = "ALARM" if kind == "alarm" else "STATUS"
            top_preview = ", ".join(
                f"`{r['name']}` ({r['root_score']:.2f})" for r in ranked[:3]
            )
            ts_part = f" ts={ts_val}" if ts_val is not None else ""
            print(
                f"**[{tag} tick={tick}{ts_part}]** `e_value={e_value:.2f}` "
                f"`max_exo_z={max_exo_z:.2f}` top_root=`{top_name}` (ch={top_ch}) | {top_preview}",
                flush=True,
            )

    def _push_tick(row_vals: list[float], ts_val: Any = None) -> None:
        nonlocal alarm_count, first_alarm_tick, max_e_value
        assert sieve is not None
        tick_res = sieve.push(row_vals)
        if tick_res.e_value > max_e_value:
            max_e_value = tick_res.e_value

        ranked_cache: list[dict[str, Any]] | None = None
        if tick_res.alarm:
            alarm_count += 1
            if first_alarm_tick is None:
                first_alarm_tick = tick_res.tick
            ranked_cache = _ranked_top_k(sieve)
            _emit_event(
                "alarm",
                tick_res.tick,
                tick_res.e_value,
                tick_res.max_exo_z,
                tick_res.top_root_channel,
                tick_res.top_root_name,
                ranked_cache,
                ts_val=ts_val,
            )
        if ns.every > 0 and sieve.ticks % ns.every == 0:
            if ranked_cache is None:
                ranked_cache = _ranked_top_k(sieve)
            _emit_event(
                "status",
                tick_res.tick,
                tick_res.e_value,
                tick_res.max_exo_z,
                tick_res.top_root_channel,
                tick_res.top_root_name,
                ranked_cache,
                ts_val=ts_val,
            )

    try:
        with (
            contextlib.nullcontext(sys.stdin)
            if use_stdin
            else open(ns.file, encoding="utf-8", newline="")
        ) as stream_raw:
            stream: TextIO = stream_raw
            line_iter = iter(stream)
            first_line: str | None = None
            for raw in line_iter:
                if raw.strip():
                    first_line = raw
                    break

            if first_line is None:
                raise ValueError("Input stream contained zero data rows")

            detected_fmt: str = ns.input_format
            if detected_fmt == "auto":
                detected_fmt = "jsonl" if first_line.lstrip().startswith("{") else "csv"

            if detected_fmt == "jsonl":
                first_obj = json.loads(first_line)
                if not isinstance(first_obj, dict):
                    raise ValueError(
                        f"JSONL line 1 must decode to a JSON object, got {type(first_obj).__name__}"
                    )

                ts_key: str | None = ns.time_col
                if ts_key is not None and ts_key not in first_obj:
                    raise ValueError(
                        f"Specified --time-col {ts_key!r} not found in first JSONL record"
                    )

                numeric_keys = [
                    str(k)
                    for k, v in first_obj.items()
                    if str(k) != ts_key and _parse_finite_num(v) is not None
                ]
                if ts_key is None:
                    for k in list(first_obj.keys()):
                        k_str = str(k)
                        if k_str.strip().lower() in _TIMESTAMP_HEADER_NAMES:
                            if k_str not in numeric_keys:
                                ts_key = k_str
                                break
                            if len(numeric_keys) > 1:
                                ts_key = k_str
                                numeric_keys.remove(k_str)
                                break

                if not numeric_keys:
                    raise ValueError("No finite numeric metric columns found in first JSONL record")

                channel_names = numeric_keys
                dim = len(channel_names)
                sieve = CausalSieve(
                    dim=dim,
                    lag=ns.lag,
                    channel_names=channel_names,
                    forgetting=ns.forgetting,
                    e_alpha=ns.alpha,
                    warmup_ticks=ns.warmup,
                )
                last_valid = [0.0] * dim

                def _extract_jsonl_row(obj: dict[str, Any]) -> tuple[list[float], Any]:
                    row: list[float] = []
                    for j, key in enumerate(channel_names):
                        val = _parse_finite_num(obj.get(key))
                        if val is not None:
                            last_valid[j] = val
                            row.append(val)
                        else:
                            row.append(last_valid[j])
                    return row, (obj.get(ts_key) if ts_key is not None else None)

                row0, ts0 = _extract_jsonl_row(first_obj)
                _push_tick(row0, ts_val=ts0)

                for line_no, raw in enumerate(line_iter, start=2):
                    s = raw.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON on line {line_no}: {exc.msg}") from exc
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"JSONL line {line_no} must decode to a JSON object, got {type(obj).__name__}"
                        )
                    row_vals, ts_val = _extract_jsonl_row(obj)
                    _push_tick(row_vals, ts_val=ts_val)
            else:
                first_cells = next(csv.reader([first_line]))
                if first_cells and first_cells[0].startswith("\ufeff"):
                    first_cells[0] = first_cells[0].lstrip("\ufeff")

                is_header = any(_parse_csv_float(cell) is None for cell in first_cells)
                ts_idx: int | None = None
                data_cells: list[str]
                headers: list[str]

                if is_header:
                    headers = [cell.strip() for cell in first_cells]
                    second_line: str | None = None
                    for raw in line_iter:
                        if raw.strip():
                            second_line = raw
                            break
                    if second_line is None:
                        raise ValueError("CSV stream contains header only and zero data rows")
                    data_cells = next(csv.reader([second_line]))

                    if ns.time_col is not None:
                        if ns.time_col not in headers:
                            raise ValueError(
                                f"Specified --time-col {ns.time_col!r} not found in CSV header: {headers}"
                            )
                        ts_idx = headers.index(ns.time_col)

                    numeric_indices = [
                        i
                        for i in range(len(headers))
                        if i != ts_idx
                        and i < len(data_cells)
                        and _parse_csv_float(data_cells[i]) is not None
                    ]
                    if ts_idx is None:
                        for i, h in enumerate(headers):
                            if h.strip().lower() in _TIMESTAMP_HEADER_NAMES:
                                if i not in numeric_indices:
                                    ts_idx = i
                                    break
                                if len(numeric_indices) > 1:
                                    ts_idx = i
                                    numeric_indices.remove(i)
                                    break

                    if not numeric_indices:
                        raise ValueError("No finite numeric metric columns found in CSV stream")
                    keep_indices = numeric_indices
                    channel_names = [headers[i] or f"ch_{i}" for i in keep_indices]
                else:
                    data_cells = first_cells
                    if ns.time_col is not None:
                        try:
                            ts_idx = int(ns.time_col)
                        except ValueError as exc:
                            raise ValueError(
                                f"CSV has no header; --time-col {ns.time_col!r} must be a 0-based column index"
                            ) from exc
                        if ts_idx < 0 or ts_idx >= len(data_cells):
                            raise ValueError(
                                f"--time-col index {ts_idx} out of range for {len(data_cells)} columns"
                            )
                    keep_indices = [i for i in range(len(data_cells)) if i != ts_idx]
                    if not keep_indices:
                        raise ValueError("No numeric metric columns remain in CSV stream")
                    channel_names = [f"ch_{i}" for i in keep_indices]

                dim = len(channel_names)
                sieve = CausalSieve(
                    dim=dim,
                    lag=ns.lag,
                    channel_names=channel_names,
                    forgetting=ns.forgetting,
                    e_alpha=ns.alpha,
                    warmup_ticks=ns.warmup,
                )
                last_valid = [0.0] * dim

                def _extract_csv_row(cells: list[str]) -> tuple[list[float], Any]:
                    row: list[float] = []
                    for j, col_i in enumerate(keep_indices):
                        val = _parse_csv_float(cells[col_i]) if col_i < len(cells) else None
                        if val is not None:
                            last_valid[j] = val
                            row.append(val)
                        else:
                            row.append(last_valid[j])
                    ts_v: Any = None
                    if ts_idx is not None and ts_idx < len(cells):
                        ts_v = cells[ts_idx].strip()
                    return row, ts_v

                row0, ts0 = _extract_csv_row(data_cells)
                _push_tick(row0, ts_val=ts0)

                for raw in line_iter:
                    if not raw.strip():
                        continue
                    cells = next(csv.reader([raw]))
                    row_vals, ts_val = _extract_csv_row(cells)
                    _push_tick(row_vals, ts_val=ts_val)
    except KeyboardInterrupt:
        pass

    if sieve is None:
        raise ValueError("Input stream contained zero data rows")

    try:
        final_ranked = _ranked_top_k(sieve)
        top_item = final_ranked[0] if final_ranked else None
        if ns.format == "jsonl":
            summary: dict[str, Any] = {
                "event": "summary",
                "ticks": sieve.ticks,
                "dim": sieve.dim,
                "lag": sieve.lag,
                "simd_isa": sieve.simd_isa,
                "alarm": alarm_count > 0,
                "alarm_count": alarm_count,
                "first_alarm_tick": first_alarm_tick,
                "max_e_value": round(max_e_value, 4),
                "top_root_channel": top_item["channel"] if top_item is not None else -1,
                "top_root_name": top_item["name"] if top_item is not None else None,
                "ranked_top_k": final_ranked,
            }
            print(json.dumps(summary), flush=True)
        else:
            status_word = "ALARM" if alarm_count > 0 else "NOMINAL"
            print(
                f"\n## Watch Summary ({status_word} | ticks={sieve.ticks} | dim={sieve.dim} | "
                f"alarms={alarm_count} | first_alarm={first_alarm_tick} | max_e={max_e_value:.2f})"
            )
            print("| rank | channel | root_score | peak_exo_z | raw_z | onset_tick |")
            print("|---:|---|---:|---:|---:|---:|")
            for item in final_ranked:
                print(
                    f"| {item['rank']} | `{item['name']}` | {item['root_score']:.4f} | "
                    f"{item['peak_exo_z']:+.4f} | {item['raw_z']:+.4f} | {item['onset_tick']} |"
                )
            sys.stdout.flush()
    finally:
        sieve.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    import argparse

    parser = argparse.ArgumentParser(
        prog="causalsieve",
        description="Sub-millisecond SIMD online multivariate causal discovery & lossless-recoverable stream digests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rca
    p_rca = sub.add_parser(
        "rca",
        help="Run causal root-cause analysis on a CSV telemetry file.",
    )
    p_rca.add_argument("file", help="Path to multivariate CSV telemetry file.")
    p_rca.add_argument("--lag", type=int, default=2, help="VAR lag order p (default: 2).")
    p_rca.add_argument("--top-k", type=int, default=5, help="Top-K root causes (default: 5).")
    p_rca.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format: md or json (default: md).",
    )
    p_rca.add_argument(
        "--hinge",
        action="store_true",
        help="Enable asymmetric hinge max(0, x-1) saturation basis.",
    )
    p_rca.add_argument(
        "--max-err",
        type=float,
        default=0.05,
        help="L_inf error bound for PLA trajectory knots (default: 0.05).",
    )
    p_rca.add_argument(
        "--forgetting",
        type=float,
        default=0.995,
        help="Forgetting factor lambda in (0, 1] (default: 0.995).",
    )
    p_rca.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="Ridge regularization alpha > 0 (default: 0.01).",
    )
    p_rca.add_argument(
        "--lookback",
        type=int,
        default=0,
        help="Root-cause lookback window in ticks; 0 = auto (default: 0).",
    )
    p_rca.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable causal baseline scale normalization (anchor-shift only).",
    )
    p_rca.add_argument(
        "--no-screened",
        action="store_true",
        help="Force Tier-1 full engine even for high-dimensional CSVs (>=128 cols).",
    )
    p_rca.set_defaults(handler=_cmd_rca)

    # digest
    p_dig = sub.add_parser(
        "digest",
        help="Emit a token-budgeted Caveman Causal Digest with lossless recovery handles.",
    )
    p_dig.add_argument("file", help="Path to multivariate CSV telemetry file.")
    p_dig.add_argument(
        "--tokens-budget",
        type=int,
        default=400,
        help="Target LLM token budget (default: 400).",
    )
    p_dig.add_argument("--lag", type=int, default=2, help="VAR lag order p (default: 2).")
    p_dig.add_argument("--top-k", type=int, default=5, help="Top-K root causes (default: 5).")
    p_dig.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format: md or json (default: md).",
    )
    p_dig.add_argument("--hinge", action="store_true", help="Enable nonlinear hinge basis.")
    p_dig.add_argument(
        "--max-err",
        type=float,
        default=0.05,
        help="PLA L_inf error bound (default: 0.05).",
    )
    p_dig.add_argument(
        "--forgetting",
        type=float,
        default=0.995,
        help="Forgetting factor lambda (default: 0.995).",
    )
    p_dig.set_defaults(handler=_cmd_digest)

    # slice
    p_sl = sub.add_parser(
        "slice",
        help="Recover 100%% lossless exact FP64 raw values for a channel tick range.",
    )
    p_sl.add_argument("file", help="Path to multivariate CSV telemetry file.")
    p_sl.add_argument("--channel", required=True, help="Channel name or 0-based index.")
    p_sl.add_argument("--start", type=int, required=True, help="Inclusive start tick.")
    p_sl.add_argument("--end", type=int, required=True, help="Exclusive end tick.")
    p_sl.add_argument(
        "--format",
        choices=("json", "csv", "table"),
        default="json",
        help="Output format (default: json).",
    )
    p_sl.set_defaults(handler=_cmd_slice)

    # counterfactual
    p_cf = sub.add_parser(
        "counterfactual",
        help="Simulate exact h-step counterfactual do(channel += delta) propagation.",
    )
    p_cf.add_argument("file", help="Path to multivariate CSV telemetry file.")
    p_cf.add_argument("--channel", required=True, help="Target channel name or 0-based index.")
    p_cf.add_argument("--delta", type=float, required=True, help="Intervention magnitude delta.")
    p_cf.add_argument("--horizon", type=int, default=10, help="Forward steps (default: 10).")
    p_cf.add_argument("--lag", type=int, default=2, help="VAR lag order p (default: 2).")
    p_cf.add_argument("--hinge", action="store_true", help="Enable nonlinear hinge basis.")
    p_cf.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable baseline scale normalization.",
    )
    p_cf.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format: md or json (default: md).",
    )
    p_cf.set_defaults(handler=_cmd_counterfactual)

    # watch
    p_watch = sub.add_parser(
        "watch",
        help="Stream CSV or JSONL telemetry line-by-line with real-time Ville E-process alerts.",
    )
    p_watch.add_argument(
        "file",
        nargs="?",
        default="-",
        help="Path to CSV/JSONL file, or '-' (default) to stream from stdin.",
    )
    p_watch.add_argument(
        "--input-format",
        choices=("auto", "csv", "jsonl"),
        default="auto",
        help="Input stream format: auto, csv, or jsonl (default: auto).",
    )
    p_watch.add_argument("--lag", type=int, default=2, help="VAR lag order p (default: 2).")
    p_watch.add_argument(
        "--forgetting",
        type=float,
        default=0.995,
        help="Forgetting factor lambda in (0, 1] (default: 0.995).",
    )
    p_watch.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Anytime-valid Ville E-process significance level alpha (default: 0.01).",
    )
    p_watch.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Baseline warmup ticks before alarm activation (default: 0).",
    )
    p_watch.add_argument("--top-k", type=int, default=5, help="Top-K root causes (default: 5).")
    p_watch.add_argument(
        "--every",
        type=int,
        default=0,
        help="Emit periodic status every N ticks; 0 = alerts + final summary only (default: 0).",
    )
    p_watch.add_argument(
        "--format",
        choices=("jsonl", "md"),
        default="jsonl",
        help="Output event format: jsonl or md (default: jsonl).",
    )
    p_watch.add_argument(
        "--time-col",
        default=None,
        help="Explicit timestamp column/key name (or 0-based index for headerless CSV) to exclude.",
    )
    p_watch.set_defaults(handler=_cmd_watch)

    # info
    p_info = sub.add_parser(
        "info",
        help="Show CausalSieve version, active SIMD ISA, and native library path.",
    )
    p_info.add_argument("--json", action="store_true", help="Print info as JSON.")
    p_info.set_defaults(handler=_cmd_info)

    # mcp
    p_mcp = sub.add_parser(
        "mcp",
        help="Launch the zero-dependency JSON-RPC 2.0 MCP stdio server.",
    )
    p_mcp.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="Restrict CSV ingestion to files under this directory (repeatable).",
    )
    p_mcp.add_argument(
        "--max-csv-bytes",
        type=int,
        default=None,
        help="Maximum allowed CSV file size in bytes.",
    )
    p_mcp.set_defaults(handler=_cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args in (["-h"], ["--help"]):
        sys.stdout.write(_TOP_HELP)
        return 0
    parser = build_parser()
    ns = parser.parse_args(args)
    try:
        return int(ns.handler(ns))
    except Exception as exc:
        print(f"causalsieve error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

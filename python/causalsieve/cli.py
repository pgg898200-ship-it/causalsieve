"""Fast (<20 ms startup) zero-dependency CLI for CausalSieve.

All heavy modules (``ctypes``, ``_native``, ``digest``, ``mcp_server``) are
lazily imported only inside the invoked subcommand handler.
"""

from __future__ import annotations

import sys

_TYPE_CHECKING = False
if _TYPE_CHECKING:
    import argparse

_TOP_HELP = """usage: causalsieve [-h] {rca,digest,slice,counterfactual,info,mcp} ...

Sub-millisecond SIMD online multivariate causal discovery & lossless-recoverable stream digests.

positional arguments:
  {rca,digest,slice,counterfactual,info,mcp}
    rca                 Run causal root-cause analysis on a CSV telemetry file.
    digest              Emit a token-budgeted Caveman Causal Digest with lossless recovery handles.
    slice               Recover 100% lossless exact FP64 raw values for a channel tick range.
    counterfactual      Simulate exact h-step counterfactual do(channel += delta) propagation.
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

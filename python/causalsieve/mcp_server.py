"""Zero-dependency JSON-RPC 2.0 MCP stdio server for CausalSieve.

Speaks the Model Context Protocol (MCP) over newline-delimited JSON-RPC 2.0
on stdin/stdout with strict input validation, path safety checks, and a shared
thread-safe :class:`~causalsieve.swarm.CausalBlackboard` across 6 tools:

  1. ``sieve_ingest_csv``
  2. ``sieve_push_ticks``
  3. ``sieve_causal_graph``
  4. ``sieve_counterfactual``
  5. ``sieve_recover_slice``
  6. ``sieve_swarm_propose``
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, TextIO

from causalsieve._native import CausalSieve, analyze_csv
from causalsieve.digest import build_digest, parse_slice_handle
from causalsieve.swarm import CausalBlackboard

SERVER_NAME = "causalsieve"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

DEFAULT_MAX_CSV_BYTES = 100 * 1024 * 1024  # 100 MiB
DEFAULT_MAX_SLICE_POINTS = 100_000
DEFAULT_MAX_PUSH_ROWS = 50_000

# JSON-RPC 2.0 standard error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "sieve_ingest_csv",
        "description": (
            "Stream and analyze a multivariate CSV telemetry file with CausalSieve in milliseconds. "
            "Returns a 95-99.5% token-compressed Causal Digest (exogenous root causes vs downstream "
            "victims, pruned conditional Granger + contemporaneous LiNGAM graph, L_inf-bounded PLA "
            "trajectories, and lossless slice:// recovery handles) plus the registered stream_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a local regular CSV telemetry file.",
                },
                "stream_id": {
                    "type": "string",
                    "description": "Optional custom stream_id (defaults to deterministic content hash).",
                },
                "lag": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "default": 2,
                    "description": "VAR lag order p (default 2).",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "default": 5,
                    "description": "Number of top root-cause channels to include (default 5).",
                },
                "tokens_budget": {
                    "type": "integer",
                    "minimum": 120,
                    "maximum": 8000,
                    "description": "Optional target LLM token budget for the digest.",
                },
                "nonlinear_hinge": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable asymmetric hinge max(0, x-1) saturation basis.",
                },
                "max_err": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.05,
                    "description": "L_inf absolute error bound for Swing-Filter PLA knots.",
                },
                "forgetting": {
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.995,
                    "description": "Exponential forgetting factor lambda in (0, 1].",
                },
                "format": {
                    "type": "string",
                    "enum": ["md", "json"],
                    "default": "md",
                    "description": "Digest output format: compact Markdown ('md') or JSON ('json').",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sieve_push_ticks",
        "description": (
            "Stream live multivariate numeric rows into a stateful CausalSieve stream on the shared "
            "blackboard (auto-creating the stream on first call) and return instant exogenous root-cause "
            "attribution and anytime-valid Ville E-value alarm state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Identifier for the live stream.",
                },
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "minItems": 1,
                    "description": "2D row-major array of finite float values, shape (n_ticks, D).",
                },
                "channel_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional length-D channel names (used when creating a new stream).",
                },
                "lag": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "default": 2,
                    "description": "VAR lag order p when creating a new stream.",
                },
                "forgetting": {
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.995,
                    "description": "Exponential forgetting factor lambda in (0, 1].",
                },
                "nonlinear_hinge": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable asymmetric hinge nonlinear basis when creating stream.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "default": 5,
                    "description": "Number of ranked root-cause channels to return.",
                },
            },
            "required": ["stream_id", "rows"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sieve_causal_graph",
        "description": (
            "Return statistically pruned conditional Granger (lagged) edges and contemporaneous "
            "innovation partial-correlation + Hyvarinen-Smith LiNGAM oriented edges for a stream."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Registered stream_id.",
                },
                "min_granger_f": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 3.0,
                    "description": "Minimum conditional Granger F-statistic threshold.",
                },
                "min_partial_r2": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.02,
                    "description": "Minimum partial R^2 threshold for lagged edges.",
                },
                "min_pcorr": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "Minimum |partial correlation| threshold for contemporaneous edges.",
                },
                "max_edges": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 20,
                    "description": "Maximum edges per edge class to return.",
                },
            },
            "required": ["stream_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sieve_counterfactual",
        "description": (
            "Simulate an exact h-step counterfactual do-intervention do(channel += delta) through the "
            "learned online VAR(p) dynamics to quantitatively verify whether a candidate root cause "
            "reproduces observed downstream propagation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Registered stream_id.",
                },
                "channel": {
                    "oneOf": [{"type": "string"}, {"type": "integer", "minimum": 0}],
                    "description": "Target channel name or 0-based index to intervene on.",
                },
                "delta": {
                    "type": "number",
                    "description": "Intervention magnitude delta injected at h=0.",
                },
                "horizon": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 10,
                    "description": "Number of forward steps to simulate (default 10).",
                },
            },
            "required": ["stream_id", "channel", "delta"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sieve_recover_slice",
        "description": (
            "Recover 100% lossless exact IEEE-754 FP64 raw telemetry values from the native ring "
            "buffer for any (stream_id, channel, start_tick, end_tick) or slice:// URI handle."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Optional slice://<stream_id>/<channel>/<start>..<end> handle.",
                },
                "stream_id": {
                    "type": "string",
                    "description": "Registered stream_id (required if handle is omitted).",
                },
                "channel": {
                    "oneOf": [{"type": "string"}, {"type": "integer", "minimum": 0}],
                    "description": "Channel name or 0-based index (required if handle is omitted).",
                },
                "start_tick": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Inclusive start tick (required if handle is omitted).",
                },
                "end_tick": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Exclusive end tick (required if handle is omitted).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sieve_swarm_propose",
        "description": (
            "Submit an append-only root-cause hypothesis on the shared multi-agent CausalBlackboard. "
            "Objectively scores the hypothesis via precision-whitened exogenous attribution and "
            "counterfactual do-intervention fit without overwriting peers, and returns updated "
            "proper-scoring-calibrated swarm consensus."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stream_id": {
                    "type": "string",
                    "description": "Registered stream_id on the shared blackboard.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Unique identifier for the proposing agent.",
                },
                "candidate_channel": {
                    "oneOf": [{"type": "string"}, {"type": "integer", "minimum": 0}],
                    "description": "Proposed root-cause channel name or 0-based index.",
                },
                "delta": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Counterfactual intervention delta for verification (default 1.0).",
                },
                "horizon": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 10,
                    "description": "Counterfactual propagation horizon (default 10).",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Agent's stated probability confidence in [0, 1] (scored via Brier/log loss).",
                },
                "rationale": {
                    "type": "string",
                    "default": "",
                    "description": "Optional brief explanation (logged immutably; does not affect objective score).",
                },
            },
            "required": ["stream_id", "agent_id", "candidate_channel"],
            "additionalProperties": False,
        },
    },
]


# =========================================================================
# Validation & Path Safety Helpers
# =========================================================================


class ValidationError(ValueError):
    """Raised when tool arguments fail strict schema or safety checks."""


def _require_str(args: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    val = args.get(key)
    if not isinstance(val, str):
        raise ValidationError(f"Parameter {key!r} must be a string")
    if not allow_empty and not val.strip():
        raise ValidationError(f"Parameter {key!r} must be a non-empty string")
    return val


def _validate_stream_id(sid: str) -> str:
    s = sid.strip()
    if not s or len(s) > 128:
        raise ValidationError("stream_id must be 1..128 characters")
    if any(ch in s for ch in ("/", "\\", "\x00", "\n", "\r", " ")):
        raise ValidationError("stream_id must not contain whitespace, slashes, or control characters")
    if ".." in s:
        raise ValidationError("stream_id must not contain '..'")
    return s


def _require_int(
    args: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if key not in args or args[key] is None:
        if default is not None:
            return default
        raise ValidationError(f"Missing required integer parameter {key!r}")
    val = args[key]
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValidationError(f"Parameter {key!r} must be an integer")
    if min_val is not None and val < min_val:
        raise ValidationError(f"Parameter {key!r} must be >= {min_val}, got {val}")
    if max_val is not None and val > max_val:
        raise ValidationError(f"Parameter {key!r} must be <= {max_val}, got {val}")
    return val


def _require_float(
    args: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
    min_val: float | None = None,
    exclusive_min: float | None = None,
    max_val: float | None = None,
) -> float:
    if key not in args or args[key] is None:
        if default is not None:
            return default
        raise ValidationError(f"Missing required numeric parameter {key!r}")
    val = args[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValidationError(f"Parameter {key!r} must be a finite number")
    f = float(val)
    if not math.isfinite(f):
        raise ValidationError(f"Parameter {key!r} must be finite, got {val}")
    if min_val is not None and f < min_val:
        raise ValidationError(f"Parameter {key!r} must be >= {min_val}, got {f}")
    if exclusive_min is not None and f <= exclusive_min:
        raise ValidationError(f"Parameter {key!r} must be > {exclusive_min}, got {f}")
    if max_val is not None and f > max_val:
        raise ValidationError(f"Parameter {key!r} must be <= {max_val}, got {f}")
    return f


def _require_bool(args: dict[str, Any], key: str, *, default: bool = False) -> bool:
    if key not in args or args[key] is None:
        return default
    val = args[key]
    if not isinstance(val, bool):
        raise ValidationError(f"Parameter {key!r} must be a boolean")
    return val


def _require_channel(args: dict[str, Any], key: str) -> int | str:
    if key not in args or args[key] is None:
        raise ValidationError(f"Missing required parameter {key!r}")
    val = args[key]
    if isinstance(val, bool):
        raise ValidationError(f"Parameter {key!r} must be a channel name or non-negative integer")
    if isinstance(val, int):
        if val < 0:
            raise ValidationError(f"Parameter {key!r} channel index must be >= 0")
        return val
    if isinstance(val, str) and val.strip():
        return val.strip()
    raise ValidationError(f"Parameter {key!r} must be a non-empty string or non-negative integer")


def _validate_csv_path(
    raw_path: str,
    allowed_roots: tuple[Path, ...],
    max_bytes: int,
) -> Path:
    if "\x00" in raw_path:
        raise ValidationError("Invalid path: contains null byte")
    try:
        resolved = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"CSV path does not exist or is inaccessible: {raw_path}") from exc

    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValidationError(f"CSV path must be a regular file (not directory/device): {resolved}")

    if allowed_roots and not any(resolved.is_relative_to(root) for root in allowed_roots):
        roots_s = ", ".join(str(r) for r in allowed_roots)
        raise ValidationError(
            f"Access denied: {resolved} is outside allowed roots [{roots_s}]"
        )

    if st.st_size > max_bytes:
        raise ValidationError(
            f"CSV file size ({st.st_size} bytes) exceeds limit ({max_bytes} bytes)"
        )
    return resolved


# =========================================================================
# MCP Server Core
# =========================================================================


class MCPServer:
    """Stateful, zero-dependency JSON-RPC 2.0 MCP server backed by :class:`CausalBlackboard`."""

    def __init__(
        self,
        *,
        allowed_roots: list[str | Path] | None = None,
        max_csv_bytes: int | None = None,
        blackboard: CausalBlackboard | None = None,
    ) -> None:
        roots_raw = list(allowed_roots) if allowed_roots else []
        env_roots = os.environ.get("CAUSALSIEVE_ALLOWED_ROOTS", "").strip()
        if env_roots and not roots_raw:
            roots_raw = [p for p in env_roots.split(os.pathsep) if p.strip()]
        self._allowed_roots: tuple[Path, ...] = tuple(
            Path(p).expanduser().resolve() for p in roots_raw
        )

        if max_csv_bytes is not None:
            self._max_csv_bytes = int(max_csv_bytes)
        else:
            env_max = os.environ.get("CAUSALSIEVE_MAX_CSV_BYTES", "").strip()
            self._max_csv_bytes = int(env_max) if env_max.isdigit() else DEFAULT_MAX_CSV_BYTES

        self._blackboard = blackboard if blackboard is not None else CausalBlackboard()
        self._tool_handlers = {
            "sieve_ingest_csv": self._tool_ingest_csv,
            "sieve_push_ticks": self._tool_push_ticks,
            "sieve_causal_graph": self._tool_causal_graph,
            "sieve_counterfactual": self._tool_counterfactual,
            "sieve_recover_slice": self._tool_recover_slice,
            "sieve_swarm_propose": self._tool_swarm_propose,
        }

    @property
    def blackboard(self) -> CausalBlackboard:
        return self._blackboard

    def close(self) -> None:
        self._blackboard.close()

    # ----- Tool implementations -----

    def _tool_ingest_csv(self, args: dict[str, Any]) -> str:
        raw_path = _require_str(args, "path")
        path = _validate_csv_path(raw_path, self._allowed_roots, self._max_csv_bytes)

        custom_sid = args.get("stream_id")
        if custom_sid is not None:
            if not isinstance(custom_sid, str):
                raise ValidationError("Parameter 'stream_id' must be a string")
            custom_sid = _validate_stream_id(custom_sid)

        lag = _require_int(args, "lag", default=2, min_val=1, max_val=16)
        top_k = _require_int(args, "top_k", default=5, min_val=1, max_val=64)
        tokens_budget = (
            _require_int(args, "tokens_budget", min_val=120, max_val=8000)
            if args.get("tokens_budget") is not None
            else None
        )
        nonlinear_hinge = _require_bool(args, "nonlinear_hinge", default=False)
        max_err = _require_float(args, "max_err", default=0.05, min_val=0.0)
        forgetting = _require_float(
            args, "forgetting", default=0.995, exclusive_min=0.0, max_val=1.0
        )
        fmt = args.get("format", "md")
        if fmt not in ("md", "json"):
            raise ValidationError("Parameter 'format' must be 'md' or 'json'")

        res = analyze_csv(
            path,
            lag=lag,
            forgetting=forgetting,
            nonlinear_hinge=nonlinear_hinge,
            use_screened=False,
        )
        assert isinstance(res.sieve, CausalSieve)
        sid = self._blackboard.register_sieve(
            custom_sid,
            res.sieve,
            raw_bytes=res.raw_bytes,
            overwrite=True,
        )
        digest = build_digest(
            res.sieve,
            top_k=top_k,
            max_err=max_err,
            tokens_budget=tokens_budget,
            stream_id=sid,
            raw_bytes=res.raw_bytes,
            store=self._blackboard.store,
        )
        if fmt == "json":
            return digest.to_json()
        return digest.to_markdown()

    def _tool_push_ticks(self, args: dict[str, Any]) -> str:
        sid = _validate_stream_id(_require_str(args, "stream_id"))
        rows = args.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValidationError("Parameter 'rows' must be a non-empty 2D array of numbers")
        if len(rows) > DEFAULT_MAX_PUSH_ROWS:
            raise ValidationError(
                f"Batch too large ({len(rows)} rows); maximum per call is {DEFAULT_MAX_PUSH_ROWS}"
            )
        first_row = rows[0]
        if not isinstance(first_row, list) or not first_row:
            raise ValidationError("Each entry in 'rows' must be a non-empty array of numbers")
        dim = len(first_row)

        cleaned_rows: list[list[float]] = []
        for r_idx, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != dim:
                raise ValidationError(
                    f"Row {r_idx} has length {len(row) if isinstance(row, list) else 'non-list'}, expected {dim}"
                )
            vals: list[float] = []
            for c_idx, v in enumerate(row):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValidationError(f"Non-numeric value at rows[{r_idx}][{c_idx}]")
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValidationError(f"Non-finite value at rows[{r_idx}][{c_idx}]: {v}")
                vals.append(fv)
            cleaned_rows.append(vals)

        top_k = _require_int(args, "top_k", default=5, min_val=1, max_val=64)

        if not self._blackboard.has_stream(sid):
            ch_names = args.get("channel_names")
            if ch_names is not None and (
                not isinstance(ch_names, list)
                or len(ch_names) != dim
                or not all(isinstance(n, str) and n.strip() for n in ch_names)
            ):
                raise ValidationError(
                    f"channel_names must be a list of {dim} non-empty strings"
                )
            lag = _require_int(args, "lag", default=2, min_val=1, max_val=16)
            forgetting = _require_float(
                args, "forgetting", default=0.995, exclusive_min=0.0, max_val=1.0
            )
            nonlinear_hinge = _require_bool(args, "nonlinear_hinge", default=False)
            self._blackboard.create_stream(
                sid,
                dim=dim,
                lag=lag,
                channel_names=ch_names,
                forgetting=forgetting,
                nonlinear_hinge=nonlinear_hinge,
            )

        sieve = self._blackboard.get_sieve(sid)
        if sieve.dim != dim:
            raise ValidationError(
                f"Stream {sid!r} expects dim={sieve.dim}, received rows with dim={dim}"
            )

        results = self._blackboard.push_batch(sid, cleaned_rows, return_results=True)
        assert results
        last = results[-1]
        attr = sieve.root_attribution()
        payload = {
            "stream_id": sid,
            "ticks_pushed": len(cleaned_rows),
            "watermark": self._blackboard.watermark(sid),
            "alarm": bool(any(r.alarm for r in results)),
            "last_tick": last.to_dict(),
            "top_root_causes": [item.to_dict() for item in attr.items[: min(top_k, dim)]],
        }
        return json.dumps(payload, separators=(",", ":"))

    def _tool_causal_graph(self, args: dict[str, Any]) -> str:
        sid = _validate_stream_id(_require_str(args, "stream_id"))
        min_f = _require_float(args, "min_granger_f", default=3.0, min_val=0.0)
        min_r2 = _require_float(args, "min_partial_r2", default=0.02, min_val=0.0, max_val=1.0)
        min_pcorr = _require_float(args, "min_pcorr", default=0.25, min_val=0.0, max_val=1.0)
        max_edges = _require_int(args, "max_edges", default=20, min_val=1, max_val=200)

        sieve = self._blackboard.get_sieve(sid)
        granger = sieve.granger_matrix()
        prec = sieve.precision_matrix()
        lingam = sieve.contemporaneous_orientation()
        names = sieve.channel_names
        dim = sieve.dim

        lagged = granger.top_edges(min_f=min_f, min_partial_r2=min_r2, top_k=max_edges)
        contemp: list[dict[str, Any]] = []
        for i in range(dim):
            for j in range(i + 1, dim):
                pc = prec.pcorr[i][j]
                if not math.isfinite(pc) or abs(pc) < min_pcorr:
                    continue
                r_ij = lingam[i][j]
                r_ji = lingam[j][i]
                diff = r_ij - r_ji
                if abs(diff) >= 1e-3:
                    src, dst, lr = (
                        (names[i], names[j], r_ij) if diff > 0 else (names[j], names[i], r_ji)
                    )
                    directed = True
                else:
                    src, dst, lr, directed = names[i], names[j], 0.5 * (r_ij + r_ji), False
                contemp.append(
                    {
                        "src": src,
                        "dst": dst,
                        "directed": directed,
                        "pcorr": round(pc, 4),
                        "lingam_r": round(lr, 4),
                    }
                )
        contemp.sort(key=lambda e: abs(e["pcorr"]), reverse=True)
        payload = {
            "stream_id": sid,
            "watermark": self._blackboard.watermark(sid),
            "lagged_granger_edges": lagged,
            "contemporaneous_edges": contemp[:max_edges],
        }
        return json.dumps(payload, separators=(",", ":"))

    def _tool_counterfactual(self, args: dict[str, Any]) -> str:
        sid = _validate_stream_id(_require_str(args, "stream_id"))
        channel = _require_channel(args, "channel")
        delta = _require_float(args, "delta")
        horizon = _require_int(args, "horizon", default=10, min_val=1, max_val=500)

        sieve = self._blackboard.get_sieve(sid)
        ch_idx = sieve.resolve_channel(channel)
        ch_name = sieve.channel_names[ch_idx]
        traj = sieve.counterfactual_do(ch_idx, delta, horizon=horizon)
        names = sieve.channel_names

        peak_impacts: list[dict[str, Any]] = []
        for j, name in enumerate(names):
            best_h = 0
            best_val = 0.0
            for h, row in enumerate(traj):
                if abs(row[j]) > abs(best_val):
                    best_val = row[j]
                    best_h = h
            peak_impacts.append(
                {
                    "channel": name,
                    "peak_delta": round(best_val, 6),
                    "peak_step": best_h,
                }
            )
        peak_impacts.sort(key=lambda item: abs(item["peak_delta"]), reverse=True)

        payload = {
            "stream_id": sid,
            "intervention": {"channel": ch_name, "channel_index": ch_idx, "delta": delta},
            "horizon": horizon,
            "channel_names": names,
            "peak_impacts": peak_impacts,
            "trajectory": [[round(v, 6) for v in row] for row in traj],
        }
        return json.dumps(payload, separators=(",", ":"))

    def _tool_recover_slice(self, args: dict[str, Any]) -> str:
        handle = args.get("handle")
        channel: int | str
        if handle is not None:
            if not isinstance(handle, str):
                raise ValidationError("Parameter 'handle' must be a string")
            sid, channel, start_tick, end_tick = parse_slice_handle(handle)
            sid = _validate_stream_id(sid)
        else:
            sid = _validate_stream_id(_require_str(args, "stream_id"))
            channel = _require_channel(args, "channel")
            start_tick = _require_int(args, "start_tick", min_val=0)
            end_tick = _require_int(args, "end_tick", min_val=0)

        if end_tick < start_tick:
            raise ValidationError(f"end_tick ({end_tick}) must be >= start_tick ({start_tick})")
        if end_tick - start_tick > DEFAULT_MAX_SLICE_POINTS:
            raise ValidationError(
                f"Requested slice length ({end_tick - start_tick}) exceeds cap ({DEFAULT_MAX_SLICE_POINTS})"
            )

        raw = self._blackboard.recover_slice(sid, channel, start_tick, end_tick)
        payload = {
            "stream_id": sid,
            "lossless": True,
            **raw.to_dict(),
        }
        return json.dumps(payload, separators=(",", ":"))

    def _tool_swarm_propose(self, args: dict[str, Any]) -> str:
        sid = _validate_stream_id(_require_str(args, "stream_id"))
        agent_id = _require_str(args, "agent_id").strip()
        candidate = _require_channel(args, "candidate_channel")
        delta = _require_float(args, "delta", default=1.0)
        horizon = _require_int(args, "horizon", default=10, min_val=1, max_val=500)
        confidence = _require_float(args, "confidence", default=0.5, min_val=0.0, max_val=1.0)
        rationale = args.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValidationError("Parameter 'rationale' must be a string")

        rec = self._blackboard.verify_hypothesis(
            sid,
            agent_id,
            candidate,
            delta=delta,
            horizon=horizon,
            confidence=confidence,
            rationale=rationale,
            record=True,
        )
        consensus = self._blackboard.consensus(sid)
        payload = {
            "hypothesis": rec.to_dict(),
            "consensus": consensus.to_dict(),
        }
        return json.dumps(payload, separators=(",", ":"))

    # ----- JSON-RPC 2.0 dispatch -----

    @staticmethod
    def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err_obj: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err_obj["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err_obj}

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Process a single decoded JSON-RPC 2.0 request or notification."""
        if not isinstance(message, dict):
            return self._err(None, ERR_INVALID_REQUEST, "JSON-RPC message must be an object")

        if message.get("jsonrpc") != "2.0":
            return self._err(
                message.get("id"),
                ERR_INVALID_REQUEST,
                "Invalid or missing 'jsonrpc': expected '2.0'",
            )

        method = message.get("method")
        if not isinstance(method, str) or not method:
            return self._err(
                message.get("id"),
                ERR_INVALID_REQUEST,
                "Invalid or missing 'method'",
            )

        is_notification = "id" not in message
        msg_id = message.get("id")
        params = message.get("params")
        if params is not None and not isinstance(params, dict):
            if is_notification:
                return None
            return self._err(msg_id, ERR_INVALID_PARAMS, "Parameters must be a JSON object")
        params_dict: dict[str, Any] = params if isinstance(params, dict) else {}

        # Notifications produce no response per JSON-RPC 2.0
        if method.startswith("notifications/"):
            return None

        if method == "initialize":
            req_ver = params_dict.get("protocolVersion")
            proto_ver = (
                req_ver if req_ver in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
            )
            return self._ok(
                msg_id,
                {
                    "protocolVersion": proto_ver,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            )

        if method == "ping":
            return self._ok(msg_id, {})

        if method == "tools/list":
            return self._ok(msg_id, {"tools": TOOL_DEFINITIONS})

        if method == "tools/call":
            tool_name = params_dict.get("name")
            if not isinstance(tool_name, str) or tool_name not in self._tool_handlers:
                return self._err(
                    msg_id,
                    ERR_INVALID_PARAMS,
                    f"Unknown tool: {tool_name!r}",
                )
            tool_args = params_dict.get("arguments", {})
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                return self._err(msg_id, ERR_INVALID_PARAMS, "Tool 'arguments' must be an object")

            try:
                output_text = self._tool_handlers[tool_name](tool_args)
            except (ValidationError, KeyError, IndexError, ValueError, FileNotFoundError, OSError) as exc:
                return self._ok(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                )
            except Exception as exc:
                return self._ok(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": f"Internal tool error: {exc}"}],
                        "isError": True,
                    },
                )

            return self._ok(
                msg_id,
                {
                    "content": [{"type": "text", "text": output_text}],
                    "isError": False,
                },
            )

        if is_notification:
            return None
        return self._err(msg_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method!r}")

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Run the newline-delimited JSON-RPC 2.0 stdio loop until EOF."""
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.stdout

        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = self._err(None, ERR_PARSE, f"Parse error: {exc.msg}")
                out_stream.write(json.dumps(err_resp, separators=(",", ":")) + "\n")
                out_stream.flush()
                continue

            resp = self.handle_message(decoded)
            if resp is not None:
                out_stream.write(json.dumps(resp, separators=(",", ":")) + "\n")
                out_stream.flush()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``causalsieve-mcp``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="causalsieve-mcp",
        description="Run the CausalSieve zero-dependency JSON-RPC 2.0 MCP stdio server.",
    )
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="Restrict CSV ingestion to files under this directory (repeatable).",
    )
    parser.add_argument(
        "--max-csv-bytes",
        type=int,
        default=None,
        help=f"Maximum CSV size in bytes (default {DEFAULT_MAX_CSV_BYTES}).",
    )
    ns = parser.parse_args(argv)
    server = MCPServer(allowed_roots=ns.allowed_root, max_csv_bytes=ns.max_csv_bytes)
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Caveman-style 95-99.5% token-compressed Causal Digest + Lossless Slice Recovery.

Transforms high-volume multivariate streams into ultra-compact, structured
Markdown or JSON digests tailored for LLM and multi-agent context windows,
paired with 100% lossless FP64 recovery handles (`slice://<stream_id>/<channel>/<start>..<end>`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causalsieve._native import CausalSieve, CSVAnalysisResult, PLACompression, RawSlice

_SLICE_HANDLE_RE = re.compile(
    r"^slice://(?P<stream_id>[^/]+)/(?P<channel>.+)/(?P<start>\d+)\.\.(?P<end>\d+)$"
)


# =========================================================================
# Thread-safe Lossless Stream Registry
# =========================================================================


@dataclass
class RegisteredStream:
    """Entry held by :class:`StreamStore` for exact FP64 slice recovery."""

    stream_id: str
    sieve: CausalSieve
    raw_bytes: int | None = None
    source_path: str | None = None


class StreamStore:
    """Thread-safe registry mapping ``stream_id`` to live :class:`CausalSieve` engines."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._streams: dict[str, RegisteredStream] = {}

    def register(
        self,
        sieve: CausalSieve,
        *,
        stream_id: str | None = None,
        raw_bytes: int | None = None,
        source_path: str | None = None,
    ) -> str:
        sid = stream_id or compute_stream_id(sieve)
        with self._lock:
            self._streams[sid] = RegisteredStream(
                stream_id=sid,
                sieve=sieve,
                raw_bytes=raw_bytes,
                source_path=source_path,
            )
        return sid

    def get(self, stream_id: str) -> RegisteredStream:
        with self._lock:
            entry = self._streams.get(stream_id)
            if entry is None:
                raise KeyError(
                    f"Unknown stream_id {stream_id!r}. Registered streams: {sorted(self._streams)}"
                )
            return entry

    def contains(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._streams

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._streams.keys())

    def unregister(self, stream_id: str) -> None:
        with self._lock:
            self._streams.pop(stream_id, None)

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()

    def recover_slice(
        self,
        stream_id: str,
        channel: int | str,
        start: int,
        end: int,
    ) -> RawSlice:
        entry = self.get(stream_id)
        return entry.sieve.get_raw_slice(channel, start, end)


_DEFAULT_STORE = StreamStore()


def default_stream_store() -> StreamStore:
    """Return the process-wide default :class:`StreamStore`."""
    return _DEFAULT_STORE


def register_stream(
    sieve: CausalSieve,
    *,
    stream_id: str | None = None,
    raw_bytes: int | None = None,
    source_path: str | None = None,
    store: StreamStore | None = None,
) -> str:
    """Register a :class:`CausalSieve` for subsequent lossless slice recovery."""
    target = store if store is not None else _DEFAULT_STORE
    return target.register(
        sieve,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        source_path=source_path,
    )


def parse_slice_handle(handle: str) -> tuple[str, str, int, int]:
    """Parse ``slice://<stream_id>/<channel>/<start>..<end>`` into components."""
    m = _SLICE_HANDLE_RE.match(handle.strip())
    if not m:
        raise ValueError(
            f"Invalid slice handle {handle!r}. Expected 'slice://<stream_id>/<channel>/<start>..<end>'."
        )
    start = int(m.group("start"))
    end = int(m.group("end"))
    if end < start:
        raise ValueError(f"Invalid slice handle range {start}..{end}: end < start")
    return m.group("stream_id"), m.group("channel"), start, end


def recover_slice(
    stream_id: str,
    channel: int | str,
    start: int,
    end: int,
    *,
    store: StreamStore | None = None,
) -> RawSlice:
    """Recover exact FP64 raw values for ``(stream_id, channel, [start, end))``."""
    target = store if store is not None else _DEFAULT_STORE
    return target.recover_slice(stream_id, channel, start, end)


def recover_handle(handle: str, *, store: StreamStore | None = None) -> RawSlice:
    """Recover exact FP64 raw values directly from a ``slice://...`` handle."""
    stream_id, channel, start, end = parse_slice_handle(handle)
    return recover_slice(stream_id, channel, start, end, store=store)


# =========================================================================
# Content hashing & transparent token estimation
# =========================================================================


def compute_stream_id(sieve: CausalSieve) -> str:
    """Compute a deterministic 12-hex-character content hash for a stream."""
    h = hashlib.sha256()
    ticks = sieve.ticks
    h.update(f"d={sieve.dim};p={sieve.lag};t={ticks};".encode())
    h.update("|".join(sieve.channel_names).encode("utf-8"))
    if ticks > 0:
        ring_start = max(0, ticks - sieve.raw_ring_capacity)
        tail_start = max(ring_start, ticks - 16)
        head_end = min(ticks, ring_start + 16)
        for ch in range(min(sieve.dim, 64)):
            head_slice = sieve.get_raw_slice(ch, ring_start, head_end)
            if head_slice.values:
                h.update(struct.pack(f"<{len(head_slice.values)}d", *head_slice.values))
            if tail_start > ring_start:
                tail_slice = sieve.get_raw_slice(ch, tail_start, ticks)
                if tail_slice.values:
                    h.update(struct.pack(f"<{len(tail_slice.values)}d", *tail_slice.values))
    return h.hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """Standard ~4 chars/token estimator for LLM context accounting."""
    return max(1, (len(text) + 3) // 4)


def estimate_raw_tokens(
    ticks: int,
    dim: int,
    channel_names: list[str],
    *,
    raw_bytes: int | None = None,
) -> int:
    """Estimate token count of feeding raw CSV/JSON telemetry into an LLM context."""
    if raw_bytes is not None and raw_bytes > 0:
        return max(1, (raw_bytes + 3) // 4)
    header_bytes = sum(len(n) + 1 for n in channel_names) + 10
    # Typical numeric telemetry CSV row: timestamp (~10B) + D * (~11B per FP64 + comma)
    est_bytes = header_bytes + max(0, ticks) * (10 + max(1, dim) * 12)
    return max(1, (est_bytes + 3) // 4)


# =========================================================================
# Causal Digest representation & builder
# =========================================================================


@dataclass(frozen=True, slots=True)
class CausalEdge:
    """Pruned lagged Granger or contemporaneous partial-correlation/LiNGAM edge."""

    kind: str  # "lagged" | "contemp"
    src: str
    dst: str
    directed: bool
    f_stat: float | None = None
    partial_r2: float | None = None
    pcorr: float | None = None
    lingam_r: float | None = None

    def format_arrow(self) -> str:
        if self.kind == "lagged":
            f_val = self.f_stat if self.f_stat is not None else 0.0
            r2_val = self.partial_r2 if self.partial_r2 is not None else 0.0
            return f"{self.src} --[lagged F={f_val:.2f}, pR2={r2_val:.3f}]--> {self.dst}"
        pc = self.pcorr if self.pcorr is not None else 0.0
        lr = self.lingam_r if self.lingam_r is not None else 0.0
        if self.directed:
            return f"{self.src} --[contemp pcorr={pc:+.2f}, LiNGAM={lr:+.2f}]--> {self.dst}"
        return f"{self.src} <--[contemp pcorr={pc:+.2f}, LiNGAM={lr:+.2f}]--> {self.dst}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "src": self.src,
            "dst": self.dst,
            "directed": self.directed,
        }
        if self.f_stat is not None:
            d["f_stat"] = round(self.f_stat, 4)
        if self.partial_r2 is not None:
            d["partial_r2"] = round(self.partial_r2, 4)
        if self.pcorr is not None:
            d["pcorr"] = round(self.pcorr, 4)
        if self.lingam_r is not None:
            d["lingam_r"] = round(self.lingam_r, 4)
        return d


@dataclass(frozen=True, slots=True)
class RootRow:
    """Single row of the ranked exogenous root-cause table."""

    rank: int
    channel: int
    name: str
    root_score: float
    peak_exo_z: float
    raw_z: float
    onset_tick: int
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "channel": self.channel,
            "name": self.name,
            "root_score": round(self.root_score, 4),
            "peak_exo_z": round(self.peak_exo_z, 3),
            "raw_z": round(self.raw_z, 3),
            "onset_tick": self.onset_tick,
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class PLATrajectory:
    """Compact L_inf-bounded trajectory knots for a key channel."""

    channel: str
    start_tick: int
    end_tick: int
    max_abs_err: float
    knots: list[tuple[int, float]]

    def format_line(self) -> str:
        pts = ", ".join(f"({t}, {_fmt_compact_float(v)})" for t, v in self.knots)
        return f"{self.channel}: [{pts}] max_err<={_fmt_compact_float(self.max_abs_err)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "max_abs_err": self.max_abs_err,
            "knots": [[t, round(v, 6)] for t, v in self.knots],
        }


@dataclass
class CausalDigest:
    """Token-compressed, lossless-recoverable causal summary of a stream."""

    stream_id: str
    ticks: int
    dim: int
    lag: int
    simd_isa: str
    alarm: bool
    e_value: float
    e_threshold: float
    freivalds_rel_err: float
    refresh_count: int
    retained_start: int
    retained_end: int
    root_causes: list[RootRow]
    victim_note: str | None
    edges: list[CausalEdge]
    trajectories: list[PLATrajectory]
    recovery_handles: list[str]
    raw_tokens_est: int
    digest_tokens_est: int = 0
    reduction_pct: float = 0.0
    _markdown_cache: str = field(default="", repr=False)

    def to_markdown(self) -> str:
        """Render Caveman-compressed structured Markdown for LLM context windows."""
        if self._markdown_cache:
            return self._markdown_cache
        return _render_markdown(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "ticks": self.ticks,
            "dim": self.dim,
            "lag": self.lag,
            "simd_isa": self.simd_isa,
            "alarm": self.alarm,
            "e_value": round(self.e_value, 4),
            "e_threshold": round(self.e_threshold, 2),
            "drift_guard": {
                "freivalds_rel_err": self.freivalds_rel_err,
                "refresh_count": self.refresh_count,
            },
            "retained_window": [self.retained_start, self.retained_end],
            "root_causes": [r.to_dict() for r in self.root_causes],
            "victim_note": self.victim_note,
            "causal_graph": [e.to_dict() for e in self.edges],
            "pla_trajectories": [t.to_dict() for t in self.trajectories],
            "recovery_handles": self.recovery_handles,
            "token_stats": {
                "raw_tokens_est": self.raw_tokens_est,
                "digest_tokens_est": self.digest_tokens_est,
                "reduction_pct": round(self.reduction_pct, 2),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return json.dumps(self.to_dict(), separators=(",", ":"))
        return json.dumps(self.to_dict(), indent=indent)


def _fmt_compact_float(val: float) -> str:
    if not math.isfinite(val):
        return str(val)
    av = abs(val)
    if av == 0.0:
        return "0"
    if av >= 1e6 or av < 1e-3:
        return f"{val:.3g}"
    return f"{val:.4g}"


def _render_markdown(digest: CausalDigest, *, include_token_stats: bool = True) -> str:
    alarm_str = "ACTIVE" if digest.alarm else "CLEAR"
    drift_ok = "OK" if digest.freivalds_rel_err < 1e-4 else "DRIFT"
    lines: list[str] = [
        f"# CausalDigest `{digest.stream_id}`",
        (
            f"ticks={digest.ticks} | ch={digest.dim} | lag={digest.lag} | isa={digest.simd_isa} | "
            f"alarm={alarm_str} | E={digest.e_value:.2f} (thr={digest.e_threshold:.0f}) | "
            f"drift={drift_ok} (err={digest.freivalds_rel_err:.2e}, refreshes={digest.refresh_count})"
        ),
        "",
        "## Root Causes (Exogenous vs Victim)",
        "| rank | channel | root_score | peak_exo_z | raw_z | onset_tick | verdict |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in digest.root_causes:
        onset_s = str(r.onset_tick) if r.onset_tick >= 0 else "-"
        lines.append(
            f"| {r.rank} | {r.name} | {r.root_score:.3f} | {r.peak_exo_z:+.2f} | "
            f"{r.raw_z:+.2f} | {onset_s} | {r.verdict} |"
        )
    if digest.victim_note:
        lines.append(f"> {digest.victim_note}")

    lines.append("")
    lines.append("## Causal Graph")
    if digest.edges:
        for e in digest.edges:
            lines.append(f"- {e.format_arrow()}")
    else:
        lines.append("- (no edges above threshold)")

    if digest.trajectories:
        lines.append("")
        lines.append(
            f"## PLA Knots [{digest.trajectories[0].start_tick}..{digest.trajectories[0].end_tick})"
        )
        for tr in digest.trajectories:
            lines.append(f"- {tr.format_line()}")

    lines.append("")
    lines.append("## Lossless Recovery")
    for h in digest.recovery_handles:
        lines.append(f"- `{h}`")
    if include_token_stats:
        lines.append(
            f"tokens: raw_est={digest.raw_tokens_est} | digest_est={digest.digest_tokens_est} | "
            f"reduction={digest.reduction_pct:.2f}%"
        )
    return "\n".join(lines)


def _classify_verdict(
    *,
    rank: int,
    root_score: float,
    peak_exo_z: float,
    raw_z: float,
    onset_tick: int,
    top_exo_z: float,
    top_onset: int,
    has_incoming_from_higher: bool,
    alarm: bool,
) -> str:
    abs_exo = abs(peak_exo_z)
    abs_raw = abs(raw_z)
    if rank == 1 and (abs_exo >= 2.5 or root_score >= 2.0 or alarm):
        return "EXOGENOUS_ROOT"
    if has_incoming_from_higher and abs_raw >= 2.5:
        return "DOWNSTREAM_VICTIM"
    if abs_raw >= 3.0 and (
        abs_raw > 1.5 * max(abs_exo, 1.0)
        or (top_onset >= 0 and onset_tick > top_onset and abs_exo < 0.75 * max(top_exo_z, 1e-9))
    ):
        return "DOWNSTREAM_VICTIM"
    if abs_exo >= 4.0:
        return "SECONDARY_SOURCE"
    return "NOMINAL"


def _adaptive_pla(
    sieve: CausalSieve,
    channel_idx: int,
    start_tick: int,
    end_tick: int,
    base_max_err: float,
    max_knots: int,
) -> PLACompression:
    """Run C Swing-Filter PLA, relaxing certified epsilon only if needed to fit knot cap."""
    eps = max(float(base_max_err), 1e-12)
    pla = sieve.compress_pla(channel_idx, start_tick, end_tick, eps)
    if len(pla.knots) <= max_knots:
        return pla

    # Estimate dynamic range of retained window to pick an efficient scaling step.
    raw = sieve.get_raw_slice(channel_idx, start_tick, end_tick)
    val_span = (max(raw.values) - min(raw.values)) if raw.values else 0.0
    if val_span > 0.0 and eps < 0.01 * val_span:
        eps = max(eps * 2.0, 0.01 * val_span)
        pla = sieve.compress_pla(channel_idx, start_tick, end_tick, eps)

    for _ in range(12):
        if len(pla.knots) <= max_knots:
            break
        eps *= 2.0
        pla = sieve.compress_pla(channel_idx, start_tick, end_tick, eps)
    return pla


def build_digest(
    source: CausalSieve | CSVAnalysisResult,
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
    """Build a 95-99.5% token-compressed Causal Digest and register lossless handles."""
    from causalsieve._native import CausalSieve, CSVAnalysisResult

    source_path: str | None = None
    if isinstance(source, CSVAnalysisResult):
        if not isinstance(source.sieve, CausalSieve):
            raise TypeError("build_digest requires a Tier-1 CausalSieve instance")
        sieve = source.sieve
        if raw_bytes is None:
            raw_bytes = source.raw_bytes
        source_path = source.path
    elif isinstance(source, CausalSieve):
        sieve = source
    else:
        raise TypeError(f"Expected CausalSieve or CSVAnalysisResult, got {type(source).__name__}")

    sid = register_stream(
        sieve,
        stream_id=stream_id,
        raw_bytes=raw_bytes,
        source_path=source_path,
        store=store,
    )

    ticks = sieve.ticks
    dim = sieve.dim
    names = sieve.channel_names
    attr = sieve.root_attribution(lookback=lookback)
    granger = sieve.granger_matrix()
    prec = sieve.precision_matrix()
    lingam = sieve.contemporaneous_orientation()
    state = sieve.inspect_state()

    last_res = sieve.last_result
    alarm = bool(sieve.any_alarm or (last_res.alarm if last_res is not None else False))
    e_val = max(
        sieve.max_e_value,
        last_res.e_value if last_res is not None else 1.0,
    )
    refresh_count = last_res.refresh_count if last_res is not None else 0
    e_threshold = 1.0 / sieve.e_alpha

    # Build pruned Granger + contemporaneous edges
    lagged_edges: list[CausalEdge] = []
    for e in granger.top_edges(
        min_f=min_granger_f,
        min_partial_r2=min_partial_r2,
        top_k=max(max_edges * 2, 16),
    ):
        lagged_edges.append(
            CausalEdge(
                kind="lagged",
                src=e["src"],
                dst=e["dst"],
                directed=True,
                f_stat=float(e["f_stat"]),
                partial_r2=float(e["partial_r2"]),
            )
        )

    contemp_candidates: list[tuple[float, CausalEdge]] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            pc = prec.pcorr[i][j]
            if not math.isfinite(pc) or abs(pc) < min_pcorr:
                continue
            r_ij = lingam[i][j]
            r_ji = lingam[j][i]
            diff = r_ij - r_ji
            if abs(diff) >= 1e-3:
                if diff > 0:
                    src_name, dst_name, lr = names[i], names[j], r_ij
                else:
                    src_name, dst_name, lr = names[j], names[i], r_ji
                edge = CausalEdge(
                    kind="contemp",
                    src=src_name,
                    dst=dst_name,
                    directed=True,
                    pcorr=pc,
                    lingam_r=lr,
                )
            else:
                edge = CausalEdge(
                    kind="contemp",
                    src=names[i],
                    dst=names[j],
                    directed=False,
                    pcorr=pc,
                    lingam_r=0.5 * (r_ij + r_ji),
                )
            contemp_candidates.append((abs(pc), edge))
    contemp_candidates.sort(key=lambda pair: pair[0], reverse=True)
    contemp_edges = [edge for _, edge in contemp_candidates]

    # Combine lagged + contemporaneous edges within max_edges budget
    n_lagged = min(len(lagged_edges), max(1, (max_edges * 2 + 2) // 3))
    n_contemp = min(len(contemp_edges), max(0, max_edges - n_lagged))
    if n_lagged + n_contemp < max_edges:
        extra_lagged = min(len(lagged_edges) - n_lagged, max_edges - n_lagged - n_contemp)
        n_lagged += max(0, extra_lagged)
        extra_contemp = min(len(contemp_edges) - n_contemp, max_edges - n_lagged - n_contemp)
        n_contemp += max(0, extra_contemp)
    all_edges = lagged_edges[:n_lagged] + contemp_edges[:n_contemp]

    # Build quick directed adjacency set for victim classification
    directed_pairs = {(e.src, e.dst) for e in all_edges if e.directed}

    # Ranked root causes table
    eff_top_k = max(1, min(int(top_k), dim))
    top_item = attr.items[0] if attr.items else None
    top_exo_z = abs(top_item.peak_exo_z) if top_item is not None else 0.0
    top_onset = top_item.onset_tick if top_item is not None else -1

    higher_names: set[str] = set()
    root_rows: list[RootRow] = []
    for item in attr.items[:eff_top_k]:
        has_in = any((src, item.name) in directed_pairs for src in higher_names)
        verdict = _classify_verdict(
            rank=item.rank,
            root_score=item.root_score,
            peak_exo_z=item.peak_exo_z,
            raw_z=item.raw_z,
            onset_tick=item.onset_tick,
            top_exo_z=top_exo_z,
            top_onset=top_onset,
            has_incoming_from_higher=has_in,
            alarm=alarm,
        )
        root_rows.append(
            RootRow(
                rank=item.rank,
                channel=item.channel,
                name=item.name,
                root_score=item.root_score,
                peak_exo_z=item.peak_exo_z,
                raw_z=item.raw_z,
                onset_tick=item.onset_tick,
                verdict=verdict,
            )
        )
        higher_names.add(item.name)

    # Explicitly call out highest-|raw_z| downstream victim if distinct from rank-1 root
    victim_note: str | None = None
    if attr.items:
        loudest_raw = max(attr.items, key=lambda it: abs(it.raw_z))
        rank1 = attr.items[0]
        if (
            loudest_raw.channel != rank1.channel
            and abs(loudest_raw.raw_z) >= 2.5
            and abs(loudest_raw.raw_z) > abs(rank1.raw_z)
        ):
            victim_note = (
                f"Victim trap avoided: `{loudest_raw.name}` has highest |raw_z|="
                f"{abs(loudest_raw.raw_z):.2f} (rank #{loudest_raw.rank}, "
                f"exo_z={loudest_raw.peak_exo_z:+.2f}), while exogenous root is "
                f"`{rank1.name}` (exo_z={rank1.peak_exo_z:+.2f}, onset={rank1.onset_tick})."
            )
            # Ensure the loudest victim appears in the table if it fell outside top_k
            if all(r.channel != loudest_raw.channel for r in root_rows):
                root_rows.append(
                    RootRow(
                        rank=loudest_raw.rank,
                        channel=loudest_raw.channel,
                        name=loudest_raw.name,
                        root_score=loudest_raw.root_score,
                        peak_exo_z=loudest_raw.peak_exo_z,
                        raw_z=loudest_raw.raw_z,
                        onset_tick=loudest_raw.onset_tick,
                        verdict="DOWNSTREAM_VICTIM",
                    )
                )

    # Retained window & relevant PLA window around incident/tail
    retained_start = max(0, ticks - sieve.raw_ring_capacity)
    retained_end = ticks
    pla_window_len = min(retained_end - retained_start, max(lookback, 128) if lookback > 0 else 256)
    if top_onset >= retained_start and top_onset < retained_end:
        pla_start = max(retained_start, top_onset - 16)
        pla_end = min(retained_end, max(pla_start + pla_window_len, retained_end))
        if pla_end - pla_start > 512:
            pla_start = max(retained_start, top_onset - 16)
            pla_end = min(retained_end, pla_start + 512)
    else:
        pla_end = retained_end
        pla_start = max(retained_start, pla_end - pla_window_len)

    # PLA trajectories ONLY for top root/anomalous channels (prioritizing non-nominal)
    selected_pla_channels: list[int] = []
    non_nominal = [r for r in root_rows if r.verdict != "NOMINAL"]
    nominal = [r for r in root_rows if r.verdict == "NOMINAL"]
    for r in (*non_nominal, *nominal):
        if len(selected_pla_channels) >= max(0, pla_channels):
            break
        if r.channel not in selected_pla_channels:
            selected_pla_channels.append(r.channel)

    trajectories: list[PLATrajectory] = []
    if ticks > 0 and pla_end > pla_start:
        for ch_idx in selected_pla_channels:
            pla = _adaptive_pla(
                sieve,
                ch_idx,
                pla_start,
                pla_end,
                base_max_err=max_err,
                max_knots=max(2, pla_max_knots),
            )
            trajectories.append(
                PLATrajectory(
                    channel=names[ch_idx],
                    start_tick=pla.start_tick,
                    end_tick=pla.end_tick,
                    max_abs_err=pla.max_abs_err,
                    knots=pla.knots,
                )
            )

    # Lossless recovery handles
    recovery_handles: list[str] = []
    for ch_idx in selected_pla_channels[:3]:
        recovery_handles.append(
            f"slice://{sid}/{names[ch_idx]}/{pla_start}..{pla_end}"
        )
    if not recovery_handles and dim > 0:
        recovery_handles.append(f"slice://{sid}/{names[0]}/{retained_start}..{retained_end}")

    raw_tokens_est = estimate_raw_tokens(ticks, dim, names, raw_bytes=raw_bytes)

    digest = CausalDigest(
        stream_id=sid,
        ticks=ticks,
        dim=dim,
        lag=sieve.lag,
        simd_isa=sieve.simd_isa,
        alarm=alarm,
        e_value=e_val,
        e_threshold=e_threshold,
        freivalds_rel_err=state.freivalds_rel_err,
        refresh_count=refresh_count,
        retained_start=retained_start,
        retained_end=retained_end,
        root_causes=root_rows,
        victim_note=victim_note,
        edges=all_edges,
        trajectories=trajectories,
        recovery_handles=recovery_handles,
        raw_tokens_est=raw_tokens_est,
    )

    # Enforce optional tokens_budget by progressively trimming secondary detail.
    if tokens_budget is not None and tokens_budget > 0:
        _fit_to_token_budget(digest, int(tokens_budget), sieve, max_err)

    # Finalize exact token accounting & cached Markdown
    _finalize_token_stats(digest)
    return digest


def _finalize_token_stats(digest: CausalDigest) -> None:
    # Two-pass render so the printed token count matches the final Markdown string.
    digest.digest_tokens_est = 0
    digest.reduction_pct = 0.0
    md0 = _render_markdown(digest, include_token_stats=True)
    est = estimate_tokens(md0)
    digest.digest_tokens_est = est
    if digest.raw_tokens_est > 0:
        red = 100.0 * (1.0 - (est / digest.raw_tokens_est))
        digest.reduction_pct = max(0.0, min(99.99, red))
    else:
        digest.reduction_pct = 0.0
    md_final = _render_markdown(digest, include_token_stats=True)
    digest.digest_tokens_est = estimate_tokens(md_final)
    if digest.raw_tokens_est > 0:
        red = 100.0 * (1.0 - (digest.digest_tokens_est / digest.raw_tokens_est))
        digest.reduction_pct = max(0.0, min(99.99, red))
    digest._markdown_cache = _render_markdown(digest, include_token_stats=True)


def _fit_to_token_budget(
    digest: CausalDigest,
    budget: int,
    sieve: CausalSieve,
    base_max_err: float,
) -> None:
    """Progressively trim trajectories, edges, and tail rows to meet ``budget`` tokens."""
    def _current_tokens() -> int:
        return estimate_tokens(_render_markdown(digest, include_token_stats=True))

    if _current_tokens() <= budget:
        return

    # Step 1: tighten PLA knots per trajectory via higher certified epsilon
    for cap in (6, 4):
        if _current_tokens() <= budget:
            return
        new_trajs: list[PLATrajectory] = []
        for tr in digest.trajectories:
            if len(tr.knots) > cap:
                ch_idx = sieve.resolve_channel(tr.channel)
                pla = _adaptive_pla(
                    sieve,
                    ch_idx,
                    tr.start_tick,
                    tr.end_tick,
                    base_max_err=max(base_max_err, tr.max_abs_err * 2.0),
                    max_knots=cap,
                )
                new_trajs.append(
                    PLATrajectory(
                        channel=tr.channel,
                        start_tick=pla.start_tick,
                        end_tick=pla.end_tick,
                        max_abs_err=pla.max_abs_err,
                        knots=pla.knots,
                    )
                )
            else:
                new_trajs.append(tr)
        digest.trajectories = new_trajs

    # Step 2: drop secondary trajectories down to top 1, then 0 if still over budget
    while len(digest.trajectories) > 1 and _current_tokens() > budget:
        digest.trajectories = digest.trajectories[:-1]

    # Step 3: trim lower-priority edges
    while len(digest.edges) > 2 and _current_tokens() > budget:
        digest.edges = digest.edges[:-1]

    # Step 4: keep only top 1 recovery handle if needed
    while len(digest.recovery_handles) > 1 and _current_tokens() > budget:
        digest.recovery_handles = digest.recovery_handles[:-1]

    # Step 5: drop remaining PLA trajectory if budget is very tight
    if digest.trajectories and _current_tokens() > budget:
        digest.trajectories = []

    # Step 6: trim root table to top 3 (minimum 1)
    while len(digest.root_causes) > 3 and _current_tokens() > budget:
        digest.root_causes = digest.root_causes[:-1]
    while len(digest.edges) > 0 and _current_tokens() > budget:
        digest.edges = digest.edges[:-1]
    while len(digest.root_causes) > 1 and _current_tokens() > budget:
        digest.root_causes = digest.root_causes[:-1]

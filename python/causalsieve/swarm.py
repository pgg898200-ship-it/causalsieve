"""Multi-Agent Shared Causal Blackboard (`CausalBlackboard`).

Solves three empirical multi-agent coordination failure modes:
  1. Overwrite churn (last-write-wins): hypotheses are strictly append-only
     immutable records; no agent can overwrite or erase peer findings.
  2. Stale / torn reads during live ingest: every stream maintains a monotonic
     watermark ``tick`` and serves immutable versioned ``CausalSnapshot`` objects.
  3. Uncalibrated / persuasive-text bias: every hypothesis is objectively scored
     against precision-whitened exogenous attribution and counterfactual
     ``do(channel += delta)`` propagation fit, then pooled via Brier / log
     proper-scoring weights with per-agent anti-ballot-stuffing deduplication.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from causalsieve._native import (
    CausalSieve,
    RawSlice,
    RootAttribution,
    TickResult,
)
from causalsieve.digest import StreamStore, build_digest, compute_stream_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(max(-60.0, min(60.0, s - m))) for s in scores]
    denom = sum(exps)
    if denom <= 0.0 or not math.isfinite(denom):
        n = len(scores)
        return [1.0 / n] * n
    return [e / denom for e in exps]


def _exogenous_distribution(attr: RootAttribution) -> list[float]:
    """Convert engine root attribution scores + peak_exo_z into a calibrated prior."""
    dim = len(attr.root_scores)
    if dim == 0:
        return []
    raw_ev = [
        max(0.0, attr.root_scores[i]) + 0.5 * abs(attr.peak_exo_z[i])
        for i in range(dim)
    ]
    max_ev = max(raw_ev)
    if max_ev <= 1e-9:
        return [1.0 / dim] * dim
    # Keep temperature proportional to typical shock scale so genuine >3-sigma
    # roots concentrate cleanly while nominal streams remain diffuse.
    temp = max(1.0, max_ev / 4.0)
    return _softmax([v / temp for v in raw_ev])


def _counterfactual_fit(
    trajectory: list[list[float]],
    observed_raw_z: Sequence[float],
) -> tuple[float, float, list[float]]:
    """Measure alignment between predicted ``do()`` response and observed anomaly pattern.

    Returns ``(fit_score_0_to_1, signed_cosine, peak_signed_response_per_channel)``.
    """
    dim = len(observed_raw_z)
    if not trajectory or dim == 0:
        return 0.0, 0.0, [0.0] * dim

    pred_peak = [0.0] * dim
    for h_row in trajectory:
        for j in range(dim):
            v = h_row[j]
            if abs(v) > abs(pred_peak[j]):
                pred_peak[j] = v

    obs_norm = math.sqrt(sum(v * v for v in observed_raw_z))
    pred_norm = math.sqrt(sum(v * v for v in pred_peak))
    if obs_norm <= 1e-12 or pred_norm <= 1e-12:
        return 0.0, 0.0, pred_peak

    dot_signed = sum(pred_peak[j] * observed_raw_z[j] for j in range(dim))
    dot_abs = sum(abs(pred_peak[j]) * abs(observed_raw_z[j]) for j in range(dim))
    cos_signed = max(-1.0, min(1.0, dot_signed / (pred_norm * obs_norm)))
    cos_abs = max(0.0, min(1.0, dot_abs / (pred_norm * obs_norm)))

    fit = 0.65 * max(0.0, cos_signed) + 0.35 * cos_abs
    return max(0.0, min(1.0, fit)), cos_signed, pred_peak


# =========================================================================
# Immutable Blackboard Artifacts
# =========================================================================


@dataclass(frozen=True, slots=True)
class CausalSnapshot:
    """Immutable, versioned read-consistent snapshot of a live stream."""

    stream_id: str
    as_of_tick: int
    snapshot_version: int
    created_at_utc: str
    dim: int
    channel_names: tuple[str, ...]
    alarm: bool
    e_value: float
    attribution: RootAttribution
    top_lagged_edges: tuple[dict[str, Any], ...]
    top_contemp_edges: tuple[dict[str, Any], ...]
    digest_markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "as_of_tick": self.as_of_tick,
            "snapshot_version": self.snapshot_version,
            "created_at_utc": self.created_at_utc,
            "dim": self.dim,
            "channel_names": list(self.channel_names),
            "alarm": self.alarm,
            "e_value": round(self.e_value, 4),
            "attribution": self.attribution.to_dict(),
            "top_lagged_edges": list(self.top_lagged_edges),
            "top_contemp_edges": list(self.top_contemp_edges),
        }


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """Append-only, objectively scored root-cause hypothesis from an agent."""

    hypothesis_id: str
    stream_id: str
    agent_id: str
    as_of_tick: int
    created_at_utc: str
    candidate_channel: int
    candidate_name: str
    delta: float
    horizon: int
    stated_confidence: float
    rationale: str
    exogenous_rank: int
    exogenous_prob: float
    exogenous_log_likelihood: float
    counterfactual_fit: float
    counterfactual_cosine: float
    objective_score: float
    brier_loss: float
    log_loss: float
    calibration_weight: float
    predicted_peak_impacts: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "stream_id": self.stream_id,
            "agent_id": self.agent_id,
            "as_of_tick": self.as_of_tick,
            "created_at_utc": self.created_at_utc,
            "candidate_channel": self.candidate_channel,
            "candidate_name": self.candidate_name,
            "delta": self.delta,
            "horizon": self.horizon,
            "stated_confidence": round(self.stated_confidence, 4),
            "rationale": self.rationale,
            "exogenous_rank": self.exogenous_rank,
            "exogenous_prob": round(self.exogenous_prob, 4),
            "exogenous_log_likelihood": round(self.exogenous_log_likelihood, 4),
            "counterfactual_fit": round(self.counterfactual_fit, 4),
            "counterfactual_cosine": round(self.counterfactual_cosine, 4),
            "objective_score": round(self.objective_score, 4),
            "brier_loss": round(self.brier_loss, 4),
            "log_loss": round(self.log_loss, 4),
            "calibration_weight": round(self.calibration_weight, 4),
            "predicted_peak_impacts": [
                {"channel": name, "delta": round(val, 4)}
                for name, val in self.predicted_peak_impacts
            ],
        }


@dataclass(frozen=True, slots=True)
class ConsensusReport:
    """Proper-scoring-pooled multi-agent root-cause consensus."""

    stream_id: str
    as_of_tick: int
    total_hypotheses: int
    participating_agents: tuple[str, ...]
    winner_channel: int
    winner_name: str
    winner_probability: float
    ranked_consensus: tuple[dict[str, Any], ...]
    agent_weights: dict[str, float]
    normalized_entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "as_of_tick": self.as_of_tick,
            "total_hypotheses": self.total_hypotheses,
            "participating_agents": list(self.participating_agents),
            "winner_channel": self.winner_channel,
            "winner_name": self.winner_name,
            "winner_probability": round(self.winner_probability, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "agent_weights": {k: round(v, 4) for k, v in self.agent_weights.items()},
            "ranked_consensus": list(self.ranked_consensus),
        }


# =========================================================================
# Internal per-stream state container
# =========================================================================


class _StreamEntry:
    def __init__(
        self,
        stream_id: str,
        sieve: CausalSieve,
        *,
        raw_bytes: int | None = None,
        max_snapshots: int = 64,
    ) -> None:
        self.stream_id = stream_id
        self.sieve = sieve
        self.raw_bytes = raw_bytes
        self.lock = threading.RLock()
        self.version = 0
        self.max_snapshots = max(1, int(max_snapshots))
        self.snapshots: OrderedDict[int, CausalSnapshot] = OrderedDict()
        self.hypotheses: list[HypothesisRecord] = []


# =========================================================================
# Public CausalBlackboard
# =========================================================================


class CausalBlackboard:
    """Thread-safe shared causal blackboard for concurrent ingest + multi-agent swarms."""

    def __init__(
        self,
        *,
        max_snapshots_per_stream: int = 64,
        stream_store: StreamStore | None = None,
    ) -> None:
        self._global_lock = threading.RLock()
        self._entries: dict[str, _StreamEntry] = {}
        self._max_snapshots = max(1, int(max_snapshots_per_stream))
        self._store = stream_store if stream_store is not None else StreamStore()

    @property
    def store(self) -> StreamStore:
        """Underlying :class:`StreamStore` backing lossless slice recovery."""
        return self._store

    def _get_entry(self, stream_id: str) -> _StreamEntry:
        with self._global_lock:
            entry = self._entries.get(stream_id)
            if entry is None:
                raise KeyError(
                    f"Unknown stream_id {stream_id!r}. Active streams: {sorted(self._entries)}"
                )
            return entry

    def create_stream(
        self,
        stream_id: str,
        dim: int,
        *,
        lag: int = 2,
        channel_names: Sequence[str] | None = None,
        **sieve_kwargs: Any,
    ) -> str:
        """Create and register a new named streaming :class:`CausalSieve`."""
        if not stream_id or not stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        sieve = CausalSieve(
            dim=dim,
            lag=lag,
            channel_names=channel_names,
            **sieve_kwargs,
        )
        return self.register_sieve(stream_id, sieve)

    def register_sieve(
        self,
        stream_id: str | None,
        sieve: CausalSieve,
        *,
        raw_bytes: int | None = None,
        overwrite: bool = True,
    ) -> str:
        """Attach an existing :class:`CausalSieve` instance to the blackboard."""
        sid = stream_id.strip() if stream_id else compute_stream_id(sieve)
        with self._global_lock:
            if sid in self._entries and not overwrite:
                raise ValueError(f"Stream {sid!r} already exists on blackboard")
            entry = _StreamEntry(
                sid,
                sieve,
                raw_bytes=raw_bytes,
                max_snapshots=self._max_snapshots,
            )
            self._entries[sid] = entry
            self._store.register(sieve, stream_id=sid, raw_bytes=raw_bytes)
        return sid

    def has_stream(self, stream_id: str) -> bool:
        with self._global_lock:
            return stream_id in self._entries

    def list_streams(self) -> list[dict[str, Any]]:
        with self._global_lock:
            entries = list(self._entries.values())
        out: list[dict[str, Any]] = []
        for entry in entries:
            with entry.lock:
                out.append(
                    {
                        "stream_id": entry.stream_id,
                        "dim": entry.sieve.dim,
                        "lag": entry.sieve.lag,
                        "watermark": entry.sieve.ticks,
                        "hypotheses": len(entry.hypotheses),
                        "snapshots": len(entry.snapshots),
                        "alarm": entry.sieve.any_alarm,
                    }
                )
        return out

    def watermark(self, stream_id: str) -> int:
        """Return the monotonic ingested tick count for ``stream_id``."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            return entry.sieve.ticks

    def get_sieve(self, stream_id: str) -> CausalSieve:
        """Return the underlying :class:`CausalSieve` (caller should respect thread safety)."""
        return self._get_entry(stream_id).sieve

    def push(self, stream_id: str, x: Any, *, return_vectors: bool = True) -> TickResult:
        """Thread-safely push a single tick into ``stream_id``."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            return entry.sieve.push(x, return_vectors=return_vectors)

    def push_batch(
        self,
        stream_id: str,
        X: Any,
        *,
        return_results: bool = True,
        return_exo_z: bool = False,
        return_innov: bool = False,
    ) -> list[TickResult] | None:
        """Thread-safely push an atomic batch of ticks into ``stream_id``."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            return entry.sieve.push_batch(
                X,
                return_results=return_results,
                return_exo_z=return_exo_z,
                return_innov=return_innov,
            )

    def snapshot(
        self,
        stream_id: str,
        as_of_tick: int | None = None,
        *,
        lookback: int = 0,
        top_k: int = 5,
        max_edges: int = 8,
    ) -> CausalSnapshot:
        """Return an immutable versioned causal snapshot at ``as_of_tick`` (or current watermark)."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            if as_of_tick is not None:
                cached = entry.snapshots.get(int(as_of_tick))
                if cached is not None:
                    return cached
                if int(as_of_tick) != entry.sieve.ticks:
                    available = sorted(entry.snapshots.keys())
                    raise KeyError(
                        f"No retained snapshot at tick {as_of_tick} for stream {stream_id!r}. "
                        f"Current watermark={entry.sieve.ticks}, retained snapshots={available}."
                    )

            tick = entry.sieve.ticks
            existing = entry.snapshots.get(tick)
            if existing is not None and lookback == 0:
                return existing

            entry.version += 1
            attr = entry.sieve.root_attribution(lookback=lookback)
            digest = build_digest(
                entry.sieve,
                top_k=top_k,
                lookback=lookback,
                max_edges=max_edges,
                stream_id=entry.stream_id,
                raw_bytes=entry.raw_bytes,
                store=self._store,
            )
            lagged = tuple(e.to_dict() for e in digest.edges if e.kind == "lagged")
            contemp = tuple(e.to_dict() for e in digest.edges if e.kind == "contemp")
            snap = CausalSnapshot(
                stream_id=entry.stream_id,
                as_of_tick=tick,
                snapshot_version=entry.version,
                created_at_utc=_utc_now_iso(),
                dim=entry.sieve.dim,
                channel_names=tuple(entry.sieve.channel_names),
                alarm=digest.alarm,
                e_value=digest.e_value,
                attribution=attr,
                top_lagged_edges=lagged,
                top_contemp_edges=contemp,
                digest_markdown=digest.to_markdown(),
            )
            entry.snapshots[tick] = snap
            entry.snapshots.move_to_end(tick)
            while len(entry.snapshots) > entry.max_snapshots:
                entry.snapshots.popitem(last=False)
            return snap

    def verify_hypothesis(
        self,
        stream_id: str,
        agent_id: str,
        candidate_channel: int | str,
        delta: float = 1.0,
        horizon: int = 10,
        *,
        confidence: float = 0.5,
        rationale: str = "",
        lookback: int = 0,
        record: bool = True,
    ) -> HypothesisRecord:
        """Score an agent's root-cause hypothesis objectively and append it to the log."""
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not math.isfinite(delta):
            raise ValueError(f"delta must be finite, got {delta}")
        if horizon < 1 or horizon > 1000:
            raise ValueError(f"horizon must be in [1, 1000], got {horizon}")
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        entry = self._get_entry(stream_id)
        with entry.lock:
            sieve = entry.sieve
            ch_idx = sieve.resolve_channel(candidate_channel)
            ch_name = sieve.channel_names[ch_idx]
            tick = sieve.ticks

            attr = sieve.root_attribution(lookback=lookback)
            exo_probs = _exogenous_distribution(attr)
            exo_prob = exo_probs[ch_idx]
            exo_ll = math.log(max(exo_prob, 1e-12))
            exo_rank = attr.ranked_channels.index(ch_idx) + 1

            traj = sieve.counterfactual_do(ch_idx, float(delta), horizon=int(horizon))
            cf_fit, cf_cos, peak_vec = _counterfactual_fit(traj, attr.raw_z)

            # Objective empirical target score in [0, 1] combining exogenous + counterfactual math
            objective_score = max(0.0, min(1.0, 0.65 * exo_prob + 0.35 * cf_fit))

            # Proper scoring rules (Brier + binary cross-entropy) against objective evidence
            q = max(1e-3, min(1.0 - 1e-3, float(confidence)))
            y = objective_score
            brier_loss = (q - y) ** 2
            log_loss = -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))
            calibration_weight = math.exp(-2.0 * brier_loss) * (0.20 + 0.80 * objective_score)

            # Summarize top impacted downstream channels from counterfactual rollout
            impact_pairs = [
                (sieve.channel_names[j], peak_vec[j])
                for j in range(sieve.dim)
                if abs(peak_vec[j]) > 1e-9
            ]
            impact_pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)

            hyp_num = len(entry.hypotheses) + 1
            rec = HypothesisRecord(
                hypothesis_id=f"hyp-{hyp_num:04d}",
                stream_id=entry.stream_id,
                agent_id=agent_id.strip(),
                as_of_tick=tick,
                created_at_utc=_utc_now_iso(),
                candidate_channel=ch_idx,
                candidate_name=ch_name,
                delta=float(delta),
                horizon=int(horizon),
                stated_confidence=float(confidence),
                rationale=str(rationale)[:512],
                exogenous_rank=exo_rank,
                exogenous_prob=exo_prob,
                exogenous_log_likelihood=exo_ll,
                counterfactual_fit=cf_fit,
                counterfactual_cosine=cf_cos,
                objective_score=objective_score,
                brier_loss=brier_loss,
                log_loss=log_loss,
                calibration_weight=calibration_weight,
                predicted_peak_impacts=tuple(impact_pairs[:8]),
            )
            if record:
                entry.hypotheses.append(rec)
            return rec

    def hypotheses(
        self,
        stream_id: str,
        *,
        agent_id: str | None = None,
        channel: int | str | None = None,
    ) -> tuple[HypothesisRecord, ...]:
        """Return an immutable tuple of recorded append-only hypotheses."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            ch_filter = (
                entry.sieve.resolve_channel(channel) if channel is not None else None
            )
            out = [
                h
                for h in entry.hypotheses
                if (agent_id is None or h.agent_id == agent_id)
                and (ch_filter is None or h.candidate_channel == ch_filter)
            ]
            return tuple(out)

    def consensus(self, stream_id: str, *, lookback: int = 0) -> ConsensusReport:
        """Compute proper-scoring-calibrated consensus attribution across all agents."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            sieve = entry.sieve
            dim = sieve.dim
            names = sieve.channel_names
            tick = sieve.ticks
            attr = sieve.root_attribution(lookback=lookback)
            prior = _exogenous_distribution(attr)

            # Group by agent to compute per-agent Brier calibration weight & prevent ballot stuffing
            by_agent: dict[str, list[HypothesisRecord]] = {}
            for h in entry.hypotheses:
                by_agent.setdefault(h.agent_id, []).append(h)

            agent_weights: dict[str, float] = {}
            # Start with engine's objective exogenous prior (weight = 1.0)
            pooled = [p for p in prior]

            for agent_id, recs in sorted(by_agent.items()):
                mean_brier = sum(r.brier_loss for r in recs) / len(recs)
                mean_obj = sum(r.objective_score for r in recs) / len(recs)
                w_agent = math.exp(-2.5 * mean_brier) * (0.25 + 0.75 * mean_obj)
                agent_weights[agent_id] = w_agent

                # Deduplicate multiple proposals from the same agent for the same channel
                per_ch: dict[int, list[HypothesisRecord]] = {}
                for r in recs:
                    per_ch.setdefault(r.candidate_channel, []).append(r)

                agent_votes = [0.0] * dim
                for ch_idx, ch_recs in per_ch.items():
                    best_support = sum(
                        r.calibration_weight * (0.3 + 0.7 * r.stated_confidence)
                        for r in ch_recs
                    ) / len(ch_recs)
                    agent_votes[ch_idx] = best_support

                vote_sum = sum(agent_votes)
                if vote_sum > 1e-12:
                    for ch_idx in range(dim):
                        pooled[ch_idx] += w_agent * (agent_votes[ch_idx] / vote_sum)

            total_mass = sum(pooled)
            probs = (
                [v / total_mass for v in pooled]
                if total_mass > 1e-12
                else [1.0 / dim] * dim
            )

            ranked_idx = sorted(range(dim), key=lambda i: probs[i], reverse=True)
            ranked_rows = [
                {
                    "rank": rank_pos + 1,
                    "channel": ch_idx,
                    "name": names[ch_idx],
                    "probability": round(probs[ch_idx], 4),
                    "engine_prior": round(prior[ch_idx], 4),
                    "peak_exo_z": round(attr.peak_exo_z[ch_idx], 3),
                    "raw_z": round(attr.raw_z[ch_idx], 3),
                }
                for rank_pos, ch_idx in enumerate(ranked_idx)
            ]

            entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
            norm_entropy = entropy / math.log(dim) if dim > 1 else 0.0
            winner = ranked_idx[0]

            return ConsensusReport(
                stream_id=entry.stream_id,
                as_of_tick=tick,
                total_hypotheses=len(entry.hypotheses),
                participating_agents=tuple(sorted(by_agent.keys())),
                winner_channel=winner,
                winner_name=names[winner],
                winner_probability=probs[winner],
                ranked_consensus=tuple(ranked_rows),
                agent_weights=agent_weights,
                normalized_entropy=max(0.0, min(1.0, norm_entropy)),
            )

    def recover_slice(
        self,
        stream_id: str,
        channel: int | str,
        start_tick: int,
        end_tick: int,
    ) -> RawSlice:
        """Thread-safely recover lossless exact FP64 values from ``stream_id``."""
        entry = self._get_entry(stream_id)
        with entry.lock:
            return entry.sieve.get_raw_slice(channel, start_tick, end_tick)

    def close(self, stream_id: str | None = None) -> None:
        """Close one named stream or all streams managed by the blackboard."""
        with self._global_lock:
            if stream_id is not None:
                entry = self._entries.pop(stream_id, None)
                self._store.unregister(stream_id)
                if entry is not None:
                    with entry.lock:
                        entry.sieve.close()
                return
            entries = list(self._entries.values())
            self._entries.clear()
            self._store.clear()
        for entry in entries:
            with entry.lock:
                entry.sieve.close()

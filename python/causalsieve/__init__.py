"""CausalSieve — Sub-millisecond SIMD online multivariate causal discovery & stream digests.

Zero runtime dependencies. Uses PEP 562 lazy attribute loading with zero
top-level stdlib imports so CLI startup adds <1 ms of package overhead.
"""

__version__ = "0.2.0"
CAUSALSIEVE_ABI_VERSION = 1

__all__ = [
    "CAUSALSIEVE_ABI_VERSION",
    "__version__",
    "CSConfig",
    "CSTickResult",
    "CSVAnalysisResult",
    "CausalBlackboard",
    "CausalDigest",
    "CausalEdge",
    "CausalSieve",
    "CausalSieveError",
    "CausalSieveLibraryError",
    "CausalSnapshot",
    "ConsensusReport",
    "EngineState",
    "GrangerResult",
    "HypothesisRecord",
    "MCPServer",
    "PLACompression",
    "PLATrajectory",
    "PrecisionResult",
    "RawSlice",
    "RootAttribution",
    "RootAttributionItem",
    "RootRow",
    "ScreenedActiveGraph",
    "ScreenedBatchResult",
    "ScreenedSieve",
    "StreamStore",
    "TickResult",
    "analyze_csv",
    "analyze_dataframe",
    "analyze_jsonl",
    "analyze_otlp_metrics",
    "analyze_prometheus",
    "build_digest",
    "compute_stream_id",
    "default_stream_store",
    "estimate_raw_tokens",
    "estimate_tokens",
    "library_path",
    "load_library",
    "parse_slice_handle",
    "rcaeval_rca",
    "recover_handle",
    "recover_slice",
    "register_stream",
    "simd_isa",
]

_LAZY_MAP = {
    "CSConfig": ("causalsieve._native", "CSConfig"),
    "CSTickResult": ("causalsieve._native", "CSTickResult"),
    "CSVAnalysisResult": ("causalsieve._native", "CSVAnalysisResult"),
    "CausalSieve": ("causalsieve._native", "CausalSieve"),
    "CausalSieveError": ("causalsieve._native", "CausalSieveError"),
    "CausalSieveLibraryError": ("causalsieve._native", "CausalSieveLibraryError"),
    "EngineState": ("causalsieve._native", "EngineState"),
    "GrangerResult": ("causalsieve._native", "GrangerResult"),
    "PLACompression": ("causalsieve._native", "PLACompression"),
    "PrecisionResult": ("causalsieve._native", "PrecisionResult"),
    "RawSlice": ("causalsieve._native", "RawSlice"),
    "RootAttribution": ("causalsieve._native", "RootAttribution"),
    "RootAttributionItem": ("causalsieve._native", "RootAttributionItem"),
    "ScreenedActiveGraph": ("causalsieve._native", "ScreenedActiveGraph"),
    "ScreenedBatchResult": ("causalsieve._native", "ScreenedBatchResult"),
    "ScreenedSieve": ("causalsieve._native", "ScreenedSieve"),
    "TickResult": ("causalsieve._native", "TickResult"),
    "analyze_csv": ("causalsieve._native", "analyze_csv"),
    "library_path": ("causalsieve._native", "library_path"),
    "load_library": ("causalsieve._native", "load_library"),
    "simd_isa": ("causalsieve._native", "simd_isa"),
    "analyze_dataframe": ("causalsieve.adapters", "analyze_dataframe"),
    "analyze_jsonl": ("causalsieve.adapters", "analyze_jsonl"),
    "analyze_otlp_metrics": ("causalsieve.adapters", "analyze_otlp_metrics"),
    "analyze_prometheus": ("causalsieve.adapters", "analyze_prometheus"),
    "rcaeval_rca": ("causalsieve.adapters", "rcaeval_rca"),
    "CausalDigest": ("causalsieve.digest", "CausalDigest"),
    "CausalEdge": ("causalsieve.digest", "CausalEdge"),
    "PLATrajectory": ("causalsieve.digest", "PLATrajectory"),
    "RootRow": ("causalsieve.digest", "RootRow"),
    "StreamStore": ("causalsieve.digest", "StreamStore"),
    "build_digest": ("causalsieve.digest", "build_digest"),
    "compute_stream_id": ("causalsieve.digest", "compute_stream_id"),
    "default_stream_store": ("causalsieve.digest", "default_stream_store"),
    "estimate_raw_tokens": ("causalsieve.digest", "estimate_raw_tokens"),
    "estimate_tokens": ("causalsieve.digest", "estimate_tokens"),
    "parse_slice_handle": ("causalsieve.digest", "parse_slice_handle"),
    "recover_handle": ("causalsieve.digest", "recover_handle"),
    "recover_slice": ("causalsieve.digest", "recover_slice"),
    "register_stream": ("causalsieve.digest", "register_stream"),
    "CausalBlackboard": ("causalsieve.swarm", "CausalBlackboard"),
    "CausalSnapshot": ("causalsieve.swarm", "CausalSnapshot"),
    "ConsensusReport": ("causalsieve.swarm", "ConsensusReport"),
    "HypothesisRecord": ("causalsieve.swarm", "HypothesisRecord"),
    "MCPServer": ("causalsieve.mcp_server", "MCPServer"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'causalsieve' has no attribute {name!r}")
    module_name, attr_name = target
    mod = __import__(module_name, fromlist=[attr_name])
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val

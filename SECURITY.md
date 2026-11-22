# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `0.1.x` | Yes |

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately via GitHub Security
Advisories rather than opening a public issue:

- https://github.com/pgg898200-ship-it/causalsieve/security/advisories/new

Include:
1. Affected component (`csrc/` native core, Python `ctypes` bindings, CLI, or `causalsieve-mcp` stdio server).
2. Minimal reproducible input or proof-of-concept script.
3. Operating system, CPU architecture, and Python/compiler version.

We aim to acknowledge reports within 72 hours and publish coordinated fixes with
regression tests.

## Security Model & Hardening Controls

CausalSieve is designed for local and agent-integrated telemetry analysis with
zero runtime dependencies and zero outbound network calls:

1. **Memory Safety (`csrc/`):**
   - Zero hot-path heap allocations inside tick processing; 64-byte aligned RAII
     buffers (`AlignedBuffer<T>`).
   - Every C99 ABI entry point validates pointer nullness, channel bounds, lag
     bounds, and IEEE-754 finiteness (`std::isfinite`) before mutating state
     (all-or-nothing atomic rejection of `NaN`/`Inf`).
   - CI runs the full C++20 test suite under AddressSanitizer +
     UndefinedBehaviorSanitizer (`ASan+UBSan`) with zero findings.

2. **MCP Server (`causalsieve-mcp`):**
   - Strictly local JSON-RPC 2.0 over stdin/stdout (no listening network socket).
   - `--allowed-root <dir>` confines `sieve_ingest_csv` file reads to resolved
     regular files (`S_ISREG`) within the allowed directory tree, blocking `..`
     path traversal, symlink escapes, and special devices (`/dev/*`, FIFOs).
   - Enforces strict caps on CSV file size (`--max-csv-bytes`, default 100 MiB),
     batch push rows (`50,000`), slice recovery points (`100,000`), and
     counterfactual horizon length.
   - Rejects malformed JSON, non-object requests, boolean values masquerading as
     numbers, `NaN`/`Inf` payloads, and path-like `stream_id` strings without
     corrupting active stream state.

3. **Build & Supply Chain:**
   - Zero third-party Python runtime dependencies (`dependencies = []`).
   - Release wheels and source distributions are built in automated CI with
     published `SHA256SUMS` and PyPI OIDC Trusted Publishing.

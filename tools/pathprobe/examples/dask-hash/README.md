# When does speeding up a Dask task shorten the run?

A two-file verification job computes both SHA-256 digests and joins the results. We changed the 32 MiB file's read size from 64 bytes to 64 KiB and left the 64 MiB sibling's reader at 64 KiB. Every execution returned the same two digests. With two workers the sibling absorbed much of the candidate's speedup. With a single worker nearly the entire speedup carried over to completion.

| Local workers | Observed saving | PathProbe predicted saving | Full task saving | PathProbe absolute error | Full task saving absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| One | 36.89 ms | 14.21 ms | 38.15 ms | 22.77 ms | **1.38 ms** |
| Two | 15.09 ms | 16.15 ms | 37.62 ms | **1.32 ms** | 21.82 ms |

Each cell is a median over ten before/after pairs. The error columns summarize errors calculated separately for each pair. Both estimates receive the candidate's measured duration from the edited run. PathProbe replaces that one duration on the original graph. The full-saving estimate subtracts the edited candidate duration from its original duration. This tests how a known task speedup translates into completion time.

The two reads overlapped in all 20 two-worker executions and none of the 20 one-worker executions. The more accurate estimate switched with the worker count in all ten pairs. On this simple graph the same distinction can be computed by hand. For workers running in parallel, compare the larger of the two read durations before and after the edit. For one serial worker, compare their sums. No general-purpose index is needed for that calculation.

## Measurement and replay

PGG Engineering collected this controlled trace on November 27, 2026, using generated files on one AMD EPYC 9V74 host. Each Dask worker occupied a separate process with one thread. Each read was assigned to a worker. With two workers the reads were assigned to different workers. Input files were warmed before timing, and the baseline ran first in half the pairs. Unique task keys forced each execution to recompute the digests.

The observed span starts when the first reader enters its task and ends when the join has both digests. It includes scheduling gaps and transfer to the join. Client submission, cluster startup and delivery to the client fall outside that interval. The local processes share a monotonic clock. PathProbe uses worker compute durations from Dask's task stream. This measures a local warm-cache workload. Remote storage and network transfers can give different results.

[Raw graphs, worker intervals, spans, digests and versions](observations.json) · [Capture and offline replay](experiment.py)

From the PathProbe directory, replay the recorded pairs with the Python standard library. To collect fresh measurements with the recorded Dask Distributed version, run the second command with uv installed.

```sh
python examples/dask-hash/experiment.py
uv run --no-project --with 'distributed==2026.11.0' python examples/dask-hash/experiment.py --capture /tmp/pathprobe-hash.json
```

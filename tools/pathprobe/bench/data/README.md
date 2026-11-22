# Two measured scientific workflows

WfCommons’ WfInstances project collects task-level execution traces. Both of these Pegasus instances credit the author `rafsilva` and were generated with the WRENCH Pegasus parser. The upstream files include specification dependencies and measured runtimes for each task. These are scientific workflow executions. PathProbe's forecasts use their observed task times in an unlimited-parallelism dependency model. Queueing, worker contention, file transfer, and scheduler overhead can make the observed wall-clock duration longer. Individual duration perturbations in the benchmark are simulated.

| Fixture | Application | Tasks | Edges | Upstream execution |
| --- | --- | ---: | ---: | --- |
| cycles.json | Agroecosystem simulation | 134 | 194 | [cycles-chameleon-1l-2c-9p-001.json](https://github.com/wfcommons/WfInstances/blob/e1600aa63731fb3fcdfeee2f1eee7b30a5036378/pegasus/cycles/cycles-chameleon-1l-2c-9p-001.json) |
| montage.json | Astronomical image mosaicking | 472 | 1,284 | [montage-chameleon-dss-10d-001.json](https://github.com/wfcommons/WfInstances/blob/e1600aa63731fb3fcdfeee2f1eee7b30a5036378/pegasus/montage/montage-chameleon-dss-10d-001.json) |

The numeric projections in this directory retain the upstream LGPL version 3 terms. See [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING). The project code is licensed separately. The upstream repository, pinned revision and license are available from [WfInstances](https://github.com/wfcommons/WfInstances/tree/e1600aa63731fb3fcdfeee2f1eee7b30a5036378).

On 2026-11-22, PathProbe projected each original into a duration mapping and a dependency edge list. Task IDs are zero-based indices in specification order, serialized as strings. Durations are measured seconds converted to integral microseconds with decimal nearest-even rounding. Other input fields are discarded. Each fixture records its original URL and SHA-256.

Regenerate with the Python standard library from the repository root.

```sh
python bench/wfcommons.py
```

To regenerate offline, supply `--source-dir DIRECTORY` with the two original files under their upstream basenames.

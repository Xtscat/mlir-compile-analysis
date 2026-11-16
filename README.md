# MLIR Compile Analysis

Pass-level timing for MLIR-based compilers, reduced to one table in execution order.

The output answers four questions about a compilation:

1. which passes ran, in what order;
2. how long each one took, and how its time splits between its own work and the
   work of passes nested inside it;
3. how many times each one ran (a pass fires once per IR unit — a model with
   1,043 dispatch functions runs every codegen pass 1,043 times);
4. how many operations each one saw going in and out.

The collector is plain upstream MLIR: it hooks `MLIRContext`'s action handler and
writes a Chrome Trace. Nothing in it refers to IREE or any other project. A
25-line driver is the only project-specific part (see
[Porting](#porting-to-another-mlir-compiler)).

## Why not `--mlir-timing`

`--mlir-timing` prints per-pass wall time aggregated over executions. This tool
keeps three things that aggregation throws away:

- **Order and nesting.** One row per distinct step, parent before child, so the
  pipeline reads top to bottom. `--mlir-timing` sorts by cost.
- **Per-execution granularity on demand.** Rows aggregate executions only when
  a pass runs many times under the same parent; the count stays visible.
  Passes that run once (the expensive ones: serialization, linking) are never
  averaged together with anything.
- **Operation counts.** Each row records the operation count of its scope before
  and after, so a pass that adds 220,000 operations is visible as such.

## Output

One CSV. `examples/output/iree-color-fp16-pass-order.csv` is the reference run
(465 rows) — an excerpt:

```
seq,depth,parent_seq,stage,pass,executions,start_s,total_s,self_s,other_s,ops_before,ops_after
307,0,0,iree-hal,OpToOpPassAdaptor,1,17.651,41.312,0.117,0.000,36555,259070
340,1,307,iree-hal,TranslateAllExecutablesPass,1043,17.653,39.219,0.043,0.000,31,99
341,2,340,iree-hal,OpToOpPassAdaptor,1043,17.653,39.176,0.117,0.000,31,99
342,3,341,iree-hal,TranslateTargetExecutableVariantsPass,1043,17.654,39.059,1.089,0.000,29,97
343,4,342,iree-hal,OpToOpPassAdaptor,1043,17.654,37.943,1.054,0.000,29,100
345,5,343,iree-hal,OpPipelineAdaptorPass,4172,17.654,32.979,3.083,0.000,24,85
346,6,345,iree-hal,LLVMCPULowerExecutableTargetPass,1043,17.654,20.438,4.353,0.000,20,33
```

| Column | Meaning |
| --- | --- |
| `seq` | reading order, 1 = first to execute |
| `depth` | 0 = top-level pipeline step, 1+ = nested inside the row above |
| `parent_seq` | `seq` of the enclosing row, 0 at top level |
| `stage` | pipeline stage, from the driver's stage markers |
| `pass` | pass name |
| `executions` | how many executions this row aggregates |
| `start_s` | first execution's start, seconds from the trace origin |
| `total_s` | wall time of this step, nested work included |
| `self_s` | `total_s` minus every nested span |
| `other_s` | time in nested non-pass actions (pattern walks, conversion drivers) |
| `ops_before` / `ops_after` | operation count of the scope at the first and last execution |

Every row satisfies

```
total_s = self_s + Σ(child total_s) + other_s
```

which is the property to check after changing anything: in the reference run it
holds for all 465 rows, and the 150 top-level rows sum to the trace's root span.

`other_s` exists because some passes spend their time in a driver loop that is
not a pass: `ConvertToLLVMPass` delegating to a conversion driver, a
`CanonicalizerPass` walking patterns. That time is inside `total_s` but belongs
to no child row, so without this column the identity above would not close.

## How it works

```
model.mlir ──► driver (install collector, run pipelines) ──► trace.json ──► pass-order.py ──► pass-order.csv
```

1. **Collector** — `include/CompileAnalysis.h`, header-only. It keeps the
   handler already installed on the context (wrapping it, not replacing it) and
   records every MLIR action as a begin/end span. Enabled by environment:

   | Variable | Effect |
   | --- | --- |
   | `COMPILE_ANALYSIS` | trace path; unset or empty disables the collector |
   | `COMPILE_ANALYSIS_OPS=0` | disable the per-execution operation census |
   | `COMPILE_ANALYSIS_MAX_KINDS` | distinct operation kinds per event (default 64) |
   | `COMPILE_ANALYSIS_PATTERNS=0` | disable pattern-attempt recording |

2. **Driver** — an in-process `PassManager` runner that installs the collector
   and calls `compile_analysis::markStage("flow")` at stage boundaries. Stage
   markers are not cosmetic: several passes appear in two pipelines
   (`iree-util-verify-initialization-order` is both the first stream pass and the
   second flow pass), so slicing by "the n-th occurrence of a pass" misattributes
   whole stages.

3. **Reducer** — `tools/pass-order.py`, streaming, no full `json.load`; it folds
   the raw event stream into one row per pass per parent. It reads 3.7 GB in
   about 30 s.

## Quick start (IREE)

```bash
# 1. build the driver against an existing IREE build tree (multi-GB binary,
#    static link of IREE + MLIR + LLVM, several minutes)
cmake -G Ninja -S examples/iree -B build/iree-harness \
      -DIREE_ROOT=$HOME/Projects/iree -DIREE_BUILD=$HOME/Projects/iree/build
cmake --build build/iree-harness

# 2. collect and reduce
./examples/iree/run.sh model.mlir            # trace + table
./examples/iree/run.sh model.mlir collect    # trace only
./examples/iree/run.sh model.mlir order      # table from an existing trace
```

`model.mlir` is any MLIR file IREE can consume; the reference run used a
39.6 MB fp16 ONNX model imported to 78.9 MB of MLIR.

## Porting to another MLIR compiler

The collector and the reducer are project-independent. What a port needs:

```cpp
#include "CompileAnalysis.h"

mlir::MLIRContext context;
// ... register the project's dialects and passes ...
bool on = compile_analysis::installIfRequested(context);  // reads the environment

mlir::PassManager passManager(&context, mlir::ModuleOp::getOperationName());
if (failed(mlir::parsePassPipeline(stage, passManager)))
  return 1;
compile_analysis::markStage(stage);                    // optional, one line
if (failed(passManager.run(*module)))
  return 1;
```

Requirements:

- MLIR with action tracing (`mlir/IR/Action.h`, upstream since LLVM 19).
- The code has to run `PassManager` itself. A finished CLI binary with hidden
  symbols is not hookable from a plugin — with IREE that route fails outright:
  `nm -D libIREECompiler.so` has no `AbstractPluginRegistration`, so a dynamic
  plugin cannot be loaded.
- Multi-threaded pass managers need a thread-safe handler; this one serializes
  only stream writes, and timestamps plus the census run outside the lock.
- The collector costs real time (an action per pass execution plus two O(n)
  walks per execution with the census on), so keep it out of production builds.

The IREE harness (`examples/iree/iree-harness.cpp`) is the worked example: it
registers IREE's dialects and passes, parses the pipeline from `--pipeline`, and
wraps five pipelines in `markStage` calls.

## Reference run

Measured on IREE `9b906db8e3` (2026-11-06), `RelWithDebInfo` (`-O2 -g -DNDEBUG`,
`-fno-omit-frame-pointer`), `--mlir-disable-threading`, `llvm-cpu` target with
`--iree-llvmcpu-target-cpu=generic`.

| | |
|---|---|
| model | 39.6 MB fp16 ONNX → 78.9 MB MLIR, 1,826 compute nodes and 1,695 initializers |
| trace | 3.7 GB, 172,230 pass executions, 223 distinct passes |
| table | 465 rows: 150 top-level, 315 nested, depth up to 7 |

Time by stage: `iree-global-optimization` 13 steps 2.291 s,
`iree-dispatch-creation` 9 steps 2.820 s, `iree-flow` 14 steps 1.533 s,
`iree-stream` 78 steps 8.504 s, `iree-hal` 36 steps **63.213 s** — total
78.361 s, of which codegen is 41.312 s and serialization 16.891 s.

Two observations this table makes easy and `--mlir-timing` does not:

- 155 of the 223 passes never appear at top level; they exist only inside the
  1,043 per-executable codegen pipelines.
- The single largest leaf is `SerializeTargetExecutablesPass`: 16.529 s of its
  own time in 16.891 s of serialization, one execution out of 172,230. The pass
  with the most executions is `CSEPass`, 15,694 times, for 0.503 s in total —
  execution count and cost are unrelated.

## Limitations

- **Trace size.** 3.7 GB for this model with the census on. Do not point
  `COMPILE_ANALYSIS` at a tmpfs; a 9.8 GB tmpfs filled up mid-run and truncated
  the trace in the middle of an event.
- **Census cost.** Two scope walks per pass execution. `COMPILE_ANALYSIS_OPS=0`
  makes the trace far smaller at the cost of the `ops_*` columns.
- **Operation kinds are capped** at `COMPILE_ANALYSIS_MAX_KINDS` (64) per event;
  beyond that the event is marked truncated.
- **No pattern-level success rates outside the driver's own listeners.** A pass
  that builds its `GreedyRewriteConfig` internally offers no place to attach a
  listener, so rewrite-driver statistics cannot be collected for it.
- **Single-threaded by convention.** The collector is thread-safe, but timings
  from a 16-thread codegen are not comparable between runs; the reference
  numbers are all single-threaded.

## Profiling, not timing

Sampling and instrumentation answer different questions. For "where inside a
pass is the time", use `perf` and
[Brendan Gregg's FlameGraph](https://github.com/brendangregg/FlameGraph):

```bash
perf record -F 500 -g -- ./driver model.mlir --pipeline=...
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

`-g` needs frame pointers — IREE builds `RelWithDebInfo` with
`-fno-omit-frame-pointer` for exactly this reason. On WSL2 there is no hardware
PMU, so `-e cpu-clock` is required, and `kernel.perf_event_paranoid` must be 1 or
lower.

## Layout

```
include/CompileAnalysis.h                  collector, header only
tools/pass-order.py                        trace -> one ordered CSV
examples/iree/iree-harness.cpp             worked example: in-process IREE driver
examples/iree/CMakeLists.txt               builds it against an IREE build tree
examples/iree/run.sh                       collect + reduce in one command
examples/output/iree-color-fp16-pass-order.csv   reference output
```

## License

MIT, see [LICENSE](LICENSE).

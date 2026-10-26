# minillm

A small LLM inference engine in C++20. Target model:
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
(Llama architecture, safetensors weights). Compute is fp32 throughout.

Correctness over speed. Every kernel exists twice: a naive scalar reference
that is checked against PyTorch, and an optimized version that is checked
against the reference. Nothing gets faster before it is right.

```
include/minillm/   the frozen interfaces — nine headers, no implementation
src/ops_ref/       naive scalar reference ops (the in-repo oracle)
src/ops/           the optimized kernels
oracle/            generate.py: the PyTorch golden data
bench/             minillm_bench: ops:: vs ref::, prefill/decode tok/s, peak RSS
ci/                run.sh: the whole CI pipeline as one local command
docs/              ARCHITECTURE.md and one file per workflow
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before writing any code. It
has the data flow of a forward pass, every layout convention, and the tolerance
policy.

> `docs/` is **local-only and not committed** (see `.gitignore`). If you cloned
> this repo and `docs/` is missing, that is why — ask whoever set it up for a
> copy.

## Requirements

- CMake >= 3.24, Ninja, a C++20 compiler (AppleClang 21 / Clang 16 / GCC 13+)
- Python 3.10+ for the oracle generator
- `ccache` is used automatically if it is on `PATH`

doctest and nlohmann/json are fetched by CMake. There are no other
dependencies and there will not be any.

## Build and test

```sh
cmake --preset asan            # configure
cmake --build --preset asan    # build
ctest --preset asan            # test
```

Four presets, each with a matching build and test preset:

| Preset | What it is | When |
| --- | --- | --- |
| `debug` | `-O0 -g`, no sanitizer | in a debugger |
| `asan` | `-O2 -g`, ASan + UBSan, UB aborts | **the default** — develop here |
| `tsan` | `-O2 -g`, ThreadSanitizer | anything threaded |
| `release` | `-O3 -DNDEBUG` | the only preset to quote timings from |

Build trees go in `build/<preset>/`. The build must be **warning-free**:
`-Wall -Wextra -Wpedantic` everywhere, and a warning is treated as a bug.

Run one module's tests directly for a faster loop:

```sh
./build/asan/tests/test_matmul              # all cases
./build/asan/tests/test_matmul -ts="*rope*" # a subset (doctest flags)
ctest --preset asan -R matmul               # via ctest
```

## Running

`minillm` is the command-line front end. Once `oracle/generate.py` has
downloaded the checkpoint (see below):

```sh
./build/release/src/minillm --model models/SmolLM2-135M --prompt "The capital of France is"
```

It streams the continuation to stdout, and the load banner, timings and
anything else to stderr. `--help` lists every flag.

`--weights` chooses how the projections and the embedding are held in memory.
The norms stay f32, and the arithmetic is f32 whichever format you choose.

| `--weights` | Format | SmolLM2-135M | Weights computed with |
| --- | --- | --- | --- |
| `auto` (default) | bf16 where that is lossless, f32 otherwise | 269 MB (bf16) | the checkpoint's, unchanged |
| `f32` | f32 | 538 MB | the checkpoint's, unchanged |
| `bf16` | bf16, or an error where bf16 cannot hold a weight | 269 MB | the checkpoint's, unchanged |
| `q8_0` | 8-bit values with an f16 scale per 32 | 143 MB | rounded to 8 bits, held to their own oracle and a KL budget |

`auto` converts to bf16 only when every value it would convert is exact in
bf16, as it is for any checkpoint stored in bf16, SmolLM2-135M's included.
Otherwise it keeps f32, for example for a checkpoint stored in f32. Either
way no weight changes, so the results match `f32` up to the order of the
additions, and the greedy tokens are the fp32 oracle's. The load banner names
the format and why `auto` chose it:

```
minillm: loaded models/SmolLM2-135M (llama, 30 layers, hidden 576, vocab 49152) in 0.15 s; bf16 weights (auto: exact in bf16), 269.1 MB; up to 10 threads, 4 for small calls
```

## Generating the oracle

The golden data is **not** committed — it is reproducible, and
`oracle/data/` and `models/` are gitignored. Tests that need it skip with a
message naming the missing file, so a fresh clone builds and tests green
before you run this.

```sh
python3 -m venv oracle/.venv
oracle/.venv/bin/pip install -r oracle/requirements.txt
oracle/.venv/bin/python oracle/generate.py
```

One command, idempotent, fixed seeds. It downloads the checkpoint (~270 MB)
into `models/SmolLM2-135M/`, loads it with transformers in float32 with
`attn_implementation="eager"`, and writes ~215 files into `oracle/data/`:

| Directory | What |
| --- | --- |
| `ops/` | inputs and torch outputs for every op in `ops.h`, at toy shapes and at the model's real shapes |
| `tokenizer/` | 26-case edge-case corpus with token ids, with and without special tokens |
| `weights/` | per-layer-type samples: shape, stored dtype, first 16 values, fp32 sum |
| `attention/` | layers 0 and 15 stage by stage: q/k/v projections, post-RoPE q/k, attention output, o_proj |
| `model/` | 3 prompts: input ids, all 31 hidden states, logits, 64-token greedy generation |
| `manifest.json` | every file with its shape, dtype and description, plus the versions used |

Use `--skip-download` to regenerate from an already-downloaded checkpoint
without touching the network. `--model-id`, `--models-dir` and `--out-dir`
override the defaults.

After generating, re-run `ctest --preset asan`: the skip summary printed at the
end of each test binary should be empty.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request, on Linux (GCC) and macOS (AppleClang). Each of its steps is a
stage of [`ci/run.sh`](ci/run.sh), so the same pipeline runs locally from a
clean clone with one command:

```sh
ci/run.sh              # oracle -> asan -> tsan -> bench
ci/run.sh asan         # or any subset of stages, in the order given
```

A run fails on:

- **Any warning** while building `asan`, `tsan` or `release`. Compiler warnings
  become errors through `CMAKE_COMPILE_WARNING_AS_ERROR`, which stays set in
  that build tree afterwards, so an incremental build cannot hide one. The
  build log is also grepped for linker and CMake warnings.
- A failing test under ASan+UBSan or under TSan.
- **Any skipped oracle check.** With `oracle/data/` and `models/` in place there
  are none, so a skip means one of them is missing or broken. That should be a
  red run, not a suspiciously fast green one. Skips that are pending another
  workflow's module are listed but not counted; they are an explicit list,
  `PENDING_SKIPS` in `ci/run.sh`, and every other skip fails the run however
  it is worded. `MINILLM_ALLOW_ORACLE_SKIPS=1` turns the failure into a report.
- The benchmark exiting non-zero.

A second job, `fresh clone, no oracle`, runs the three commands from
[Build and test](#build-and-test) verbatim, with no oracle and no caches. It
checks that the oracle checks skip there instead of failing.

CI caches `oracle/data/` + `models/` (key: `oracle/generate.py` +
`oracle/requirements.txt`), the FetchContent sources in `.cache/fetchcontent/`
(key: the pinned URLs and hashes in `CMakeLists.txt`), and ccache.

## Benchmarks

```sh
cmake --preset release && cmake --build --preset release
./build/release/bench/minillm_bench                   # from the repo root
./build/release/bench/minillm_bench --filter matmul   # a subset; see --help
./build/release/bench/minillm_bench --threads 1,4,10  # kernel thread scaling
```

Quote numbers from the `release` preset only. The binary refuses to run from a
sanitizer or unoptimized build. It reads the checkpoint from
`models/SmolLM2-135M/` (the oracle generator downloads it) and exits with
status 2 if that is missing.

### Reading the benchmark output

**Whole model** comes first: prefill and decode throughput in tokens/s, then
peak RSS.

- *Prefill* is one forward pass over a 128-token synthetic prompt, starting
  from an empty cache.
- *Decode* is the 64 greedy one-token steps that follow it.

The two have different bottlenecks. Prefill is compute-bound on matmul. Decode
is memory-bound: every token streams all the weights once.

**ops:: vs ref::** has one row per op, at the shape the forward pass actually
runs it with:

- `prefill.*`: 128 rows.
- `decode.*`: 1 row, attending over 192 positions.
- `last_token.*`: the LM head and argmax, which run once per pass.

`kv_proj` and `gate_up` each stand for two matmuls of that shape per layer.

| Column | Meaning |
| --- | --- |
| `ops::`, `ref::` | median time per call over the samples (5 by default) |
| `+/-` | half the spread of the samples: (max − min) / 2 / median |
| `calls` | samples × calls per sample; fast ops are batched so each sample lasts ≥ 50 ms |
| `thr` | the kernel thread count `ops::` ran with (`--threads`; `0`, the default, resolves to every hardware thread; on Apple silicon only calls big enough to gain from the efficiency cores use more than the performance cores' threads) |
| `speedup` | `ref::` / `ops::`, measured on interleaved pairs of samples, median over the pairs; > 1 means `ops::` is faster |
| `scaling` | with several `--threads` counts: `ops::` at the first count / `ops::` at `thr`, over the same interleaved samples |

Read every speedup together with its `+/-`. An op that has not been optimized
yet forwards `ops::` to `ref::`, so its true speedup is exactly 1.00x. Whatever
its row shows is therefore the noise floor: about ±1% on a quiet machine, and a
lot more while other jobs are running. An optimization that does not clear the
noise floor has not been measured.

**Runtime** times the thread pool itself: one `kernel_parallel_for` over 1024
items with a trivial body, at each `--threads` count. `runtime.parallel_for.hot`
calls back to back, as the forward pass does (about 240 calls per decode step);
`runtime.parallel_for.cold` calls after the workers have gone to sleep, so it
includes waking them.

**Machine-readable lines** close the run, one per measurement:

```
BENCH {"name":"op.decode.matmul.q_proj.ops","threads":4,"value":187.9,"unit":"us","iterations":1360,"samples":5,"min":184.0,"max":191.4}
```

| Field | Meaning |
| --- | --- |
| `threads` | the kernel thread count it was measured at: 1 for `.ref`, which never uses the pool; the largest `--threads` count for `memory.*` |
| `value` | the median |
| `iterations` | timed calls in total (timed runs, for `model.*`) |
| `samples`, `min`, `max` | the spread; absent on peak RSS, which is a single reading |

Names follow these patterns:

| Name | Unit |
| --- | --- |
| `op.<regime>.<op>.ops` / `.ref` | µs per call |
| `op.<regime>.<op>.speedup` | ratio |
| `op.<regime>.<op>.scaling` | ratio, only with several `--threads` counts |
| `model.prefill`, `model.decode` | tok/s |
| `memory.peak_rss.model` | MiB, after load + prefill + decode |
| `memory.peak_rss.process` | MiB, whole run |

`ci/run.sh bench` writes them to `build/release/bench-results.jsonl`, and CI
uploads that file as an artifact on every run, so drift can be tracked over
time.

## Working on a module

Each module is owned by one workflow with its own file in
[`docs/workflows/`](docs/workflows/) listing the files it owns, the oracle data
it tests against, and its done criteria. These are local-only, as noted above.

| | Workflow | Owns |
| --- | --- | --- |
| 01 | [loader](docs/workflows/01-loader.md) | `src/loader/` |
| 02 | [tokenizer](docs/workflows/02-tokenizer.md) | `src/tokenizer/` |
| 03 | [norm + elementwise](docs/workflows/03-norm-elementwise.md) | `src/ops/norm.cpp`, `src/ops/elementwise.cpp` |
| 04 | [matmul](docs/workflows/04-matmul.md) | `src/ops/matmul.cpp` |
| 05 | [rope](docs/workflows/05-rope.md) | `src/ops/rope.cpp` |
| 06 | [attention + KV cache](docs/workflows/06-attention-kv.md) | `src/attention/` |
| 07 | [sampling](docs/workflows/07-sampling.md) | `src/sampling/` |
| 08 | [runtime](docs/workflows/08-runtime.md) | `src/runtime/` |
| 09 | [cli](docs/workflows/09-cli.md) | `src/cli/` |
| 10 | [bench + CI](docs/workflows/10-bench-ci.md) | `bench/`, `ci/`, `.github/workflows/` |

Three rules, in priority order:

1. **`include/` is frozen.** Those nine headers are the contract between ten
   modules developed in parallel. If one looks wrong, stop and report it
   instead of editing it.
2. **`src/ops_ref/` never changes.** It is the oracle every optimization is
   judged against. If a reference op is wrong, that is a bug report, not a
   patch.
3. **No workflow edits a build file.** `CMakeLists.txt` already lists every
   source file any workflow will own, stubs included.

You may freely add private headers and sources inside your own
`src/<module>/` directory.

## Status

The forward pass is assembled. `minillm` runs SmolLM2-135M end to end, and
its 64-token greedy generations match the PyTorch oracle's id for id; every
hidden state and the final logits are checked against torch layer by layer
(`tests/model/`).

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
| 10 | [bench + CI](docs/workflows/10-bench-ci.md) | `bench/`, CI config |

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

Phase 0. `Tensor`, every reference op and the KV cache are complete and tested
against PyTorch. Every other module is a stub that compiles, links, and throws
`std::logic_error("not implemented: ...")`, with a placeholder test that
asserts the stub is wired up and says which workflow will replace it.

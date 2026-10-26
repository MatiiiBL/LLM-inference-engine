#!/usr/bin/env bash
# minillm CI pipeline — owned by workflow 10-bench-ci.
#
# One command, from a clean clone:
#
#   ci/run.sh               every stage: oracle asan tsan bench
#   ci/run.sh asan bench    just these stages, in this order
#
# .github/workflows/ci.yml runs exactly these stages, so a green local run is
# a green CI run on the same compiler.
#
# Stages:
#   oracle  generate oracle/data/ and models/ (python oracle/generate.py)
#           unless they are already there
#   asan    build the asan preset with zero warnings, run ctest, and fail if
#           any check was skipped other than those pending another workflow
#   tsan    the same for the tsan preset
#   bench   build the release preset with zero warnings and run minillm_bench;
#           results land in build/release/bench-{output.txt,results.jsonl}
#
# Environment:
#   MINILLM_ALLOW_ORACLE_SKIPS=1  report those skips instead of failing
#   MINILLM_BENCH_ARGS="..."      extra minillm_bench arguments, e.g. "--samples 7"
#   PYTHON=python3.12             interpreter for oracle/.venv (default python3)
#
# Written for bash 3.2, the one macOS ships.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD

log() { printf '\n==> %s\n' "$*"; }
die() {
  printf '\nci/run.sh: error: %s\n' "$*" >&2
  exit 1
}

require_tools() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' is not on PATH (see README.md, Requirements)"
  done
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

# ---------------------------------------------------------------------------
# FetchContent sources: downloaded once into .cache/fetchcontent/ and shared by
# every preset via FETCHCONTENT_SOURCE_DIR_<NAME>, so CI can cache them. The
# URL and SHA256 are read out of CMakeLists.txt, so they cannot drift from
# what a plain `cmake --preset` fetches.
# ---------------------------------------------------------------------------

FETCH_ARGS=()

prepare_deps() {
  [[ ${#FETCH_ARGS[@]} -eq 0 ]] || return 0
  local name url hash dir upper
  while read -r name url hash; do
    [[ -n $url && $hash =~ ^SHA256=[0-9a-f]{64}$ ]] ||
      die "CMakeLists.txt: FetchContent_Declare($name) needs a URL and a URL_HASH SHA256=..."
    hash=${hash#SHA256=}
    dir="$ROOT/.cache/fetchcontent/$name"
    if [[ ! -f "$dir/.minillm-sha256-$hash" ]]; then
      log "fetch $name"
      rm -rf "$dir" "$dir.download"
      mkdir -p "$dir"
      curl -fsSL --retry 3 -o "$dir.download" "$url"
      [[ $(sha256 "$dir.download") == "$hash" ]] || die "$name: SHA256 mismatch for $url"
      tar -xf "$dir.download" -C "$dir" --strip-components=1
      rm -f "$dir.download"
      touch "$dir/.minillm-sha256-$hash"
    fi
    upper=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')
    FETCH_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_${upper}=$dir")
  done < <(awk '
    /FetchContent_Declare\(/ {
      name = $0; sub(/.*FetchContent_Declare\(/, "", name); sub(/[ \t)].*/, "", name); url = ""
    }
    name != "" && $1 == "URL"      { url = $2 }
    name != "" && $1 == "URL_HASH" { print name, url, $2; name = "" }
  ' CMakeLists.txt)
  [[ ${#FETCH_ARGS[@]} -gt 0 ]] || die "found no FetchContent_Declare in CMakeLists.txt"
}

# ---------------------------------------------------------------------------
# Build and test
# ---------------------------------------------------------------------------

# Configures and builds one preset with every compiler warning as an error,
# then fails if the log mentions a warning at all: -Werror covers only the
# compiler, not the linker or CMake itself.
#
# CMAKE_COMPILE_WARNING_AS_ERROR is a cache variable, so it stays set in
# build/<preset>/ afterwards. That is deliberate: a translation unit that
# warns then never produces an object, so an incremental build cannot hide it.
build_preset() {
  local preset=$1
  local build_log="build/$preset/ci-build.log"
  prepare_deps
  log "build $preset"
  mkdir -p "build/$preset"
  {
    cmake --preset "$preset" -DCMAKE_COMPILE_WARNING_AS_ERROR=ON "${FETCH_ARGS[@]}"
    cmake --build --preset "$preset"
  } 2>&1 | tee "$build_log"
  if grep -nE 'warning:|CMake (Deprecation )?Warning' "$build_log"; then
    die "$preset: the build printed the warnings above; it must be warning-free"
  fi
}

# Runs ctest, then audits what was skipped. Every test binary ends with
#
#   [minillm] N check(s) skipped:
#     - <reason>
#   [minillm] oracle dir: <dir>
#
# and ctest keeps that output in LastTest.log. A skip pending another workflow's
# module is expected until that workflow lands, so it is listed, not counted.
# Any other skip fails the run: it means oracle/data/ or the checkpoint in
# models/ is absent or broken (or, for the CLI's "binary not found", the build
# is), and that must be a red run rather than a suspiciously fast green one.
#
# So the audit matches an allowlist of pending reasons, PENDING_SKIPS, not a
# denylist of missing-data ones. The suites word a missing file in at least
# seven ways ("missing oracle data:", "missing model file:", "missing
# checkpoint:", "missing checkpoint file:", "cli: missing ...", "cli end to
# end: missing oracle data: ...", "cli end to end: --model '...' is not a model
# directory: ..."), and the next suite may invent an eighth. A denylist lets
# that one through as green; an allowlist fails it closed. The price is that a
# reworded pending skip fails too, naming itself, until its entry is updated.
#
# Each entry is an extended regex for one whole reason, anchored at both ends,
# with no wildcard that could match a missing file's path. Delete an entry when
# its module lands. The tokenizer has, so the CLI's "the tokenizer is still a
# stub" is deliberately absent: if it shows up again, that is a regression.
PENDING_SKIPS=()

test_preset() {
  local preset=$1
  local test_log="build/$preset/Testing/Temporary/LastTest.log"
  log "test $preset"
  ctest --preset "$preset"

  local skips reason re pending="" counted="" n=0
  skips=$(awk '
    /^\[minillm\] [0-9]+ check\(s\) skipped:/ { on = 1; next }
    /^\[minillm\] oracle dir:/                { on = 0 }
    on && /^  - / { sub(/^  - /, ""); print }
  ' "$test_log")
  while IFS= read -r reason; do
    [[ -n $reason ]] || continue
    # ${a[@]+"${a[@]}"}: bash 3.2 calls an empty array unbound under set -u.
    for re in ${PENDING_SKIPS[@]+"${PENDING_SKIPS[@]}"}; do
      if [[ $reason =~ $re ]]; then
        pending+="  - $reason"$'\n'
        continue 2
      fi
    done
    counted+="  - $reason"$'\n'
    n=$((n + 1))
  done <<<"$skips"

  if [[ -n $pending ]]; then
    printf '\n%s: skipped, pending other workflows (not counted):\n%s' "$preset" "$pending"
  fi
  if [[ $n -eq 0 ]]; then
    printf '\n%s: zero oracle checks skipped\n' "$preset"
    return
  fi
  printf '\n%s: %s check(s) skipped, not pending another workflow:\n%s' "$preset" "$n" "$counted"
  if [[ ${MINILLM_ALLOW_ORACLE_SKIPS:-0} == 1 ]]; then
    echo "(allowed by MINILLM_ALLOW_ORACLE_SKIPS=1)"
  else
    die "$preset: checks were skipped; run 'ci/run.sh oracle' (or set MINILLM_ALLOW_ORACLE_SKIPS=1). A skip that is genuinely pending another workflow belongs in PENDING_SKIPS."
  fi
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

stage_oracle() {
  if [[ -f oracle/data/manifest.json && -f models/SmolLM2-135M/model.safetensors ]]; then
    log "oracle: oracle/data/ and models/ are present, not regenerating (delete oracle/data/ to force)"
    return
  fi
  log "oracle: generating oracle/data/ and models/"
  local python=${PYTHON:-python3}
  require_tools "$python"
  [[ -x oracle/.venv/bin/python ]] || "$python" -m venv oracle/.venv
  local pip_args=(--quiet --disable-pip-version-check -r oracle/requirements.txt)
  # PyPI's Linux torch wheel bundles CUDA, gigabytes of it; the oracle only
  # needs the CPU build.
  [[ $(uname -s) != Linux ]] || pip_args+=(--extra-index-url https://download.pytorch.org/whl/cpu)
  oracle/.venv/bin/python -m pip install "${pip_args[@]}"
  oracle/.venv/bin/python oracle/generate.py
}

stage_asan() {
  build_preset asan
  test_preset asan
}

stage_tsan() {
  build_preset tsan
  test_preset tsan
}

# The Q8_0 quality gate (tests/model, proposal 14 §5.3 layer 4) runs only
# from a release build: it pushes about 4000 teacher-forced positions through
# an fp32 and a Q8_0 engine, which no sanitizer stage could afford. The bench
# stage builds release anyway, so the gate runs here. Every stage that gets
# this far has the oracle and the checkpoint, so a skip fails like a failure,
# and so does a filter that no longer matches the test.
run_q8_0_kl_gate() {
  log "Q8_0 KL gate (release)"
  local out=build/release/q8_0-kl-gate-output.txt
  local start=$SECONDS
  build/release/tests/test_model --test-case='SmolLM2 Q8_0: KL from the fp32 engine*' 2>&1 |
    tee "$out" || die "the Q8_0 KL gate failed; see the output above"
  if grep -qE '^\[minillm\] [0-9]+ check\(s\) skipped' "$out"; then
    die "the Q8_0 KL gate skipped its checks; it needs oracle/data/ and models/"
  fi
  grep -qE 'test cases: +1 \| +1 passed \| +0 failed' "$out" ||
    die "the Q8_0 KL gate did not run exactly one test case; has it been renamed?"
  printf '\nQ8_0 KL gate: passed in %ss\n' "$((SECONDS - start))"
}

stage_bench() {
  build_preset release
  run_q8_0_kl_gate
  log "bench"
  local out=build/release/bench-output.txt
  # A failed run must not leave the previous run's results behind to be
  # uploaded as this one's.
  rm -f "$out" build/release/bench-results.jsonl
  # MINILLM_BENCH_ARGS is split into words on purpose.
  # shellcheck disable=SC2086
  build/release/bench/minillm_bench ${MINILLM_BENCH_ARGS:-} | tee "$out"
  grep '^BENCH ' "$out" | cut -c7- >build/release/bench-results.jsonl
  printf '\nbench: %s measurement(s) in build/release/bench-results.jsonl\n' \
    "$(grep -c . build/release/bench-results.jsonl)"
}

main() {
  local stages=("$@")
  [[ ${#stages[@]} -gt 0 ]] || stages=(oracle asan tsan bench)
  local stage
  for stage in "${stages[@]}"; do
    case $stage in
      oracle | asan | tsan | bench) ;;
      *) die "unknown stage '$stage' (stages: oracle asan tsan bench)" ;;
    esac
  done
  require_tools cmake ninja ctest curl tar awk

  local start=$SECONDS
  for stage in "${stages[@]}"; do
    "stage_$stage"
  done
  log "ci/run.sh ${stages[*]}: passed in $((SECONDS - start))s"
}

main "$@"

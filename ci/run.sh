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
#           any oracle check was skipped
#   tsan    the same for the tsan preset
#   bench   build the release preset with zero warnings and run minillm_bench;
#           results land in build/release/bench-{output.txt,results.jsonl}
#
# Environment:
#   MINILLM_ALLOW_ORACLE_SKIPS=1  report skipped oracle checks instead of failing
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
# and ctest keeps that output in LastTest.log. A skip that names missing oracle
# data means the oracle is broken or absent, which must fail the run rather
# than pass as a suspiciously fast green one. A skip pending another workflow's
# module is expected until that workflow lands, so it is listed, not counted.
test_preset() {
  local preset=$1
  local test_log="build/$preset/Testing/Temporary/LastTest.log"
  log "test $preset"
  ctest --preset "$preset"

  local skips oracle pending
  skips=$(awk '
    /^\[minillm\] [0-9]+ check\(s\) skipped:/ { on = 1; next }
    /^\[minillm\] oracle dir:/                { on = 0 }
    on && /^  - / { sub(/^  - /, ""); print }
  ' "$test_log")
  oracle=$(printf '%s\n' "$skips" | grep '^missing oracle data:' || true)
  pending=$(printf '%s\n' "$skips" | grep -v '^missing oracle data:' | grep . || true)

  if [[ -n $pending ]]; then
    printf '\n%s: skipped, pending other workflows (not counted):\n%s\n' "$preset" "$(sed 's/^/  - /' <<<"$pending")"
  fi
  if [[ -z $oracle ]]; then
    printf '\n%s: zero oracle checks skipped\n' "$preset"
    return
  fi
  printf '\n%s: %s oracle check(s) skipped:\n%s\n' "$preset" "$(grep -c . <<<"$oracle")" "$(sed 's/^/  - /' <<<"$oracle")"
  if [[ ${MINILLM_ALLOW_ORACLE_SKIPS:-0} == 1 ]]; then
    echo "(allowed by MINILLM_ALLOW_ORACLE_SKIPS=1)"
  else
    die "$preset: oracle checks were skipped; run 'ci/run.sh oracle' (or set MINILLM_ALLOW_ORACLE_SKIPS=1)"
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

stage_bench() {
  build_preset release
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

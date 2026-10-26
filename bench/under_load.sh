#!/usr/bin/env bash
# Runs a command while every core is busy — owned by workflow 18.
#
#   bench/under_load.sh [-n N] [--] command [args...]
#
# Starts N busy-looping processes (default: one per hardware thread), runs the
# command, then kills them, whether the command passed, failed or was
# interrupted. The exit status is the command's.
#
# The loops are plain `while :; do :; done` shells: pure user-space work, no
# syscalls, no memory traffic, the same scheduling class as the command. They
# stand in for "other sessions are building and benchmarking", in a form that
# is the same on every run.
#
# Example — decode at the default thread count against one thread, samples
# interleaved, under load:
#
#   bench/under_load.sh build/release/bench/minillm_bench \
#     --model-dir models/SmolLM2-135M --filter model.decode --threads 0,1
#
# Written for bash 3.2, the one macOS ships.

set -euo pipefail

usage() {
  sed -n '3,4p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

n=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -n)
      [[ $# -ge 2 && $2 =~ ^[0-9]+$ ]] || usage
      n=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h | --help) usage ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || usage

if [[ -z $n ]]; then
  n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)
fi

pids=()
stop_load() {
  local pid
  for pid in ${pids[@]+"${pids[@]}"}; do kill "$pid" 2>/dev/null || true; done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" 2>/dev/null || true; done
  pids=()
}
trap stop_load EXIT
trap 'exit 130' INT TERM

for ((i = 0; i < n; i++)); do
  bash -c 'while :; do :; done' &
  pids+=("$!")
done
printf 'under_load: %d busy loop(s) running\n' "$n" >&2

status=0
"$@" || status=$?
exit "$status"

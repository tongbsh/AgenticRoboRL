#!/usr/bin/env bash
# Week-2 leak self-check (research/PLAN_C_then_A.md 2.2): the SFT planner on 4 tasks x 1 init under all four
# info x dynamics cells, on the first GPU of $GPUS only, then the leak checks. All checks must pass before C0.
#   research/c/smoke.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/c.env"
read -r -a G <<< "$GPUS"
export CELL_GPUS=${G[0]}
for cell in "visible coupled" "hidden free" "hidden coupled" "visible free"; do
  # shellcheck disable=SC2086
  "$HERE/run_cell.sh" smoke sft $cell
done
"$CTL_PYTHON" "$HERE/leak_check.py" "$RUNS_DIR/smoke" --json "$RUNS_DIR/smoke/leak_check.json"
"$CTL_PYTHON" "$HERE/analyze.py" --runs "$RUNS_DIR" --stages smoke --out "$RUNS_DIR/analysis/smoke" > /dev/null
echo "smoke OK: see $RUNS_DIR/analysis/smoke/report.md (16 episodes; numbers are not meaningful, the checks are)"

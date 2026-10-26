#!/usr/bin/env bash
# All cells of one plan-C stage, then the VLM judge, the leak checks and the analysis (research/c/README.md).
#   research/c/run_stage.sh c0       C0 pilot:      {sft, rl} x {V, M-all} on val inits 45-46        (640 episodes)
#   research/c/run_stage.sh c0_ext   C0 extension:  same cells on val inits 47-49                     (960 episodes)
#   research/c/run_stage.sh c1       C1 audit:      {sft, rl} x {V, M-all} on eval inits 1-10        (3200 episodes)
#                                                   (+ base model with INCLUDE_BASE=1)
#   research/c/run_stage.sh c2       C2 2x2:        {sft, rl} x {M-info, M-dyn} on eval inits 1-5    (1600 episodes)
#                                                   (+ M-field / M-prompt with C2_SPLIT=1)
# Each cell is resumable (run the stage again); a finished cell is skipped quickly.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/c.env"
STAGE=${1:?stage}
case $STAGE in
  c0|c0_ext) CELLS=("sft visible coupled" "sft hidden free" "rl visible coupled" "rl hidden free") ;;
  c1) CELLS=("sft visible coupled" "sft hidden free" "rl visible coupled" "rl hidden free")
      if [ "${INCLUDE_BASE:-0}" = 1 ]; then CELLS+=("base visible coupled" "base hidden free"); fi ;;
  c2) CELLS=("sft hidden coupled" "sft visible free" "rl hidden coupled" "rl visible free")
      if [ "${C2_SPLIT:-0}" = 1 ]; then CELLS+=("sft fields coupled" "sft prompt coupled" "rl fields coupled" "rl prompt coupled"); fi ;;
  *) echo "bad stage $STAGE"; exit 2 ;;
esac
for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  "$HERE/run_cell.sh" "$STAGE" $cell || echo "WARNING: cell '$cell' had controller failures; rerun the stage to resume"
done
"$HERE/run_judge.sh" "$STAGE"
"$CTL_PYTHON" "$HERE/leak_check.py" "$RUNS_DIR/$STAGE" --json "$RUNS_DIR/$STAGE/leak_check.json" \
  || echo "WARNING: leak checks failed, see $RUNS_DIR/$STAGE/leak_check.json -- do not use this stage before fixing"
case $STAGE in
  c0) STAGES="c0" ;; c0_ext) STAGES="c0 c0_ext" ;; c1) STAGES="c1" ;; c2) STAGES="c1 c2" ;;
esac
# shellcheck disable=SC2086
"$CTL_PYTHON" "$HERE/analyze.py" --runs "$RUNS_DIR" --stages $STAGES --out "$RUNS_DIR/analysis/${STAGES// /+}"

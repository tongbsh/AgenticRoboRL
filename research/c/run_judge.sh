#!/usr/bin/env bash
# The external VLM judge over every episode of a stage (after its cells; the GPU must be free).
#   research/c/run_judge.sh <stage>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/c.env"
STAGE=${1:?stage}
LOG="$RUNS_DIR/$STAGE/judge_server.log"
mkdir -p "$RUNS_DIR/$STAGE"
CUDA_VISIBLE_DEVICES=$JUDGE_GPU setsid "$PLANNER_ENV/bin/vllm" serve "$JUDGE_MODEL" --served-model-name "$JUDGE_MODEL" \
  --host 127.0.0.1 --port "$JUDGE_PORT" --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt.image 2 --limit-mm-per-prompt.video 0 > "$LOG" 2>&1 &
SERVER=$!
trap 'kill -TERM -- -$SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$JUDGE_PORT/v1/models" > /dev/null; then break; fi
  kill -0 "$SERVER" 2>/dev/null || { echo "judge server died, see $LOG"; exit 1; }
  sleep 5
done
"$CTL_PYTHON" "$HERE/vlm_judge.py" --model-url "http://127.0.0.1:$JUDGE_PORT/v1" --model "$JUDGE_MODEL" "$RUNS_DIR/$STAGE"

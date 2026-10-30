#!/usr/bin/env bash
# Offline jobs over finished episodes, on one free GPU ($OFFLINE_GPU); nothing here touches an episode's outcome.
#   research/c/run_offline.sh probe <stage> [planner ...]   probe variants (probe_variants.py), planners sft rl by
#                                                           default; each planner's own weights are served in turn
#   research/c/run_offline.sh judge <stage>                 the stronger VLM judge ($STRONG_JUDGE_MODEL) ->
#                                                           vlm_judge_strong.json
# LIMIT=10 runs on the first 10 episodes only (a test). Resumable: episodes with an output file are skipped.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/c.env"
JOB=${1:?probe or judge}; STAGE=${2:?stage}; shift 2
GPU=${OFFLINE_GPU:-$JUDGE_GPU}; PORT=${OFFLINE_PORT:-18600}
LIMIT_ARGS=(--limit "${LIMIT:-0}")
SERVER=""
trap '[ -n "$SERVER" ] && kill -TERM -- -$SERVER 2>/dev/null || true' EXIT

serve() {  # <model path> <log> <extra vllm args...>
  local model=$1 log=$2; shift 2
  CUDA_VISIBLE_DEVICES=$GPU setsid "$PLANNER_ENV/bin/vllm" serve "$model" --served-model-name "$model" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt.video 0 "$@" > "$log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then return 0; fi
    kill -0 "$SERVER" 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 5
  done
  echo "server not ready after 30 min, see $log"; exit 1
}
stop() { kill -TERM -- -"$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; SERVER=""; }

case $JOB in
  probe)
    PLANNERS=("$@"); [ ${#PLANNERS[@]} -eq 0 ] && PLANNERS=(sft rl)
    for p in "${PLANNERS[@]}"; do
      case $p in sft) W=$SFT_WEIGHTS ;; rl) W=$RL_WEIGHTS ;; base) W=$BASE_WEIGHTS ;; *) echo "bad planner $p"; exit 2 ;; esac
      echo "[$(date +%FT%T)] probe variants: $STAGE / $p on GPU $GPU"
      # the planner's serving flags of harnesses/rpent_libero_c/harness.yaml (same template, tools and parsers)
      serve "$W" "$RUNS_DIR/$STAGE/offline_probe_$p.log" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 \
        --limit-mm-per-prompt.image 128 --enable-prefix-caching --enable-auto-tool-choice --generation-config auto \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3
      "$CTL_PYTHON" "$HERE/probe_variants.py" --model-url "http://127.0.0.1:$PORT/v1" --model "$W" --planner "$p" \
        "${LIMIT_ARGS[@]}" "$RUNS_DIR/$STAGE"
      stop
    done ;;
  judge)
    echo "[$(date +%FT%T)] stronger judge: $STAGE with $STRONG_JUDGE_MODEL on GPU $GPU"
    serve "$STRONG_JUDGE_MODEL" "$RUNS_DIR/$STAGE/offline_judge.log" --max-model-len 16384 --max-num-seqs 16 \
      --limit-mm-per-prompt.image 2
    "$CTL_PYTHON" "$HERE/vlm_judge.py" --model-url "http://127.0.0.1:$PORT/v1" --model "$STRONG_JUDGE_MODEL" \
      --out-name vlm_judge_strong.json "${LIMIT_ARGS[@]}" "$RUNS_DIR/$STAGE"
    stop ;;
  *) echo "bad job $JOB"; exit 2 ;;
esac
echo "[$(date +%FT%T)] done; rerun analyze.py on the stage to see the new signals"

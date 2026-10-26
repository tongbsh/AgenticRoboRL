#!/usr/bin/env bash
# One cell of plan C: one planner under one oracle condition on one stage's initial states, spread over $GPUS
# (one rl/local_ctl.py evaluation controller per GPU, episodes dealt round-robin).
#   research/c/run_cell.sh <stage> <planner> <oracle_info> <oracle_dynamics>
#     stage            c0 | c0_ext | c1 | c2 | smoke         (research/c/tasks/<stage>.jsonl)
#     planner          sft | rl | base                       ($SFT_WEIGHTS / $RL_WEIGHTS / $BASE_WEIGHTS)
#     oracle_info      visible | hidden | fields | prompt
#     oracle_dynamics  coupled | free
# Resumable: run the same command again and finished episodes are kept. Stop: touch $RUNS_DIR/<stage>/<cell>/g*/STOP.
# CELL_GPUS="0" overrides $GPUS for this cell (smoke.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/c.env"
GPUS=${CELL_GPUS:-$GPUS}
STAGE=${1:?stage}; PLANNER=${2:?planner}; INFO=${3:?oracle_info}; DYN=${4:?oracle_dynamics}
case $STAGE in c0|c0_ext|smoke) KEY=val_init_ids ;; c1|c2) KEY=eval_init_ids ;; *) echo "bad stage $STAGE"; exit 2 ;; esac
case $PLANNER in sft) W=$SFT_WEIGHTS ;; rl) W=$RL_WEIGHTS ;; base) W=$BASE_WEIGHTS ;; *) echo "bad planner $PLANNER"; exit 2 ;; esac
case $INFO in visible|hidden|fields|prompt) ;; *) echo "bad oracle_info $INFO"; exit 2 ;; esac
case $DYN in coupled|free) ;; *) echo "bad oracle_dynamics $DYN"; exit 2 ;; esac

CELL="${PLANNER}__${INFO}-${DYN}"
DIR="$RUNS_DIR/$STAGE/$CELL"
mkdir -p "$DIR"
cat > "$DIR/cell.json" <<JSON
{"stage": "$STAGE", "planner": "$PLANNER", "weights": "$W", "info": "$INFO", "dynamics": "$DYN",
 "init_key": "$KEY", "tasks": "$HERE/tasks/$STAGE.jsonl", "max_turns": $MAX_TURNS, "max_model_len": $MAX_MODEL_LEN}
JSON
read -r -a G <<< "$GPUS"
N=${#G[@]}
echo "[$(date +%FT%T)] cell $STAGE/$CELL on GPUs ${G[*]} -> $DIR"
pids=()
for i in "${!G[@]}"; do
  g=${G[$i]}
  "$CTL_PYTHON" "$AGENTICROBORL_DIR/rl/local_ctl.py" eval \
    --config "$AGENTICROBORL_DIR/harnesses/rpent_libero_c/harness.yaml" \
    --run-dir "$DIR/g$g" --weights "$W" --tag "$STAGE" \
    --tasks "$HERE/tasks/$STAGE.jsonl" --init-key "$KEY" --shard "$i/$N" --shard-by episode \
    --var gpu="$g" --var model_port=$((PORT_BASE_MODEL + g)) --var proxy_port=$((PORT_BASE_PROXY + g)) \
    --var vla_port=$((PORT_BASE_VLA + g)) --var sam3_port=$((PORT_BASE_SAM3 + g)) \
    --var oracle_info="$INFO" --var oracle_dynamics="$DYN" \
    --var vllm_mem="$VLLM_MEM" --var max_model_len="$MAX_MODEL_LEN" --var max_turns="$MAX_TURNS" \
    > "$DIR/g$g.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
reports=("$DIR"/g*/eval/"$STAGE"/report.json)
if [ -e "${reports[0]}" ]; then
  "$CTL_PYTHON" "$AGENTICROBORL_DIR/rl/local_ctl.py" merge-eval --sum-tasks --out "$DIR/report.json" --tag "$STAGE" "${reports[@]}"
fi
echo "[$(date +%FT%T)] cell $STAGE/$CELL done (controller failures: $fail); logs $DIR/g*.log"
exit $fail

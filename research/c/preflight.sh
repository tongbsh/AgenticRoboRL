#!/usr/bin/env bash
# Static checks before the first episode (research/c/README.md, step 4). No episode is run; takes ~1 minute.
#   research/c/preflight.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/c.env" ] || { echo "FAIL: copy research/c/c.env.example to research/c/c.env and fill it in"; exit 1; }
source "$HERE/c.env"
EXPECTED_TREE=10a16d6ff9c04d3cbf14433ee7fe6728b9ec41e6
fails=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  FAIL  $*"; fails=$((fails + 1)); }
warn() { echo "  warn  $*"; }

echo "paths"
for v in AGENTICROBORL_DIR RPENT_DIR RPENT_ENV PLANNER_ENV PI05_CHECKPOINT; do
  [ -d "${!v:-}" ] && ok "$v=${!v}" || bad "$v is not a directory: ${!v:-unset}"
done
for v in SFT_WEIGHTS RL_WEIGHTS JUDGE_MODEL; do
  [ -f "${!v:-}/config.json" ] && ok "$v=${!v}" || bad "$v has no config.json: ${!v:-unset}"
done
[ -f "${SAM3_CHECKPOINT:-}" ] && ok "SAM3_CHECKPOINT=$SAM3_CHECKPOINT" || bad "SAM3_CHECKPOINT is not a file: ${SAM3_CHECKPOINT:-unset}"
[ "${INCLUDE_BASE:-0}" = 1 ] && { [ -f "${BASE_WEIGHTS:-}/config.json" ] && ok "BASE_WEIGHTS" || bad "BASE_WEIGHTS has no config.json"; }
[ -x "$PLANNER_ENV/bin/vllm" ] && ok "vllm in PLANNER_ENV" || bad "no $PLANNER_ENV/bin/vllm"
[ -x "$RPENT_ENV/bin/python" ] && ok "python in RPENT_ENV" || bad "no $RPENT_ENV/bin/python"

echo "RPent"
tree=$(cd "$RPENT_DIR" && git add -A && git write-tree 2>/dev/null)
[ "$tree" = "$EXPECTED_TREE" ] && ok "tree $tree (f70d720 + both patches)" \
  || bad "RPENT_DIR tree $tree != $EXPECTED_TREE: rebuild it with research/c/setup_rpent.sh"
where=$(cd "$RPENT_DIR" && "$RPENT_ENV/bin/python" -c "import rpent, robots.libero.prompts.oracle_free as o; print(rpent.__file__)" 2>&1)
case $where in "$RPENT_DIR"/*) ok "RPENT_ENV imports rpent from RPENT_DIR" ;; *) bad "RPENT_ENV imports rpent from: $where (pip install -e \"$RPENT_DIR[full]\")" ;; esac
if (cd "$RPENT_DIR" && "$RPENT_ENV/bin/python" -c "import pytest" 2>/dev/null); then
  if (cd "$RPENT_DIR" && RPENT_DIR="$RPENT_DIR" "$RPENT_ENV/bin/python" -m pytest -q -p no:cacheprovider "$HERE/tests/test_rpent_oracle.py" > /tmp/c_preflight_rpent.log 2>&1); then
    ok "oracle patch unit tests (fake LIBERO, no GPU)"
  else
    bad "oracle patch unit tests failed, see /tmp/c_preflight_rpent.log"
  fi
else
  warn "pytest missing in RPENT_ENV: $RPENT_ENV/bin/pip install pytest, then rerun to test the oracle patch"
fi

echo "controller / vLLM"
"$CTL_PYTHON" -c "import yaml, fastapi, uvicorn, httpx, numpy" 2>/dev/null && ok "CTL_PYTHON has yaml fastapi uvicorn httpx numpy" \
  || bad "CTL_PYTHON ($CTL_PYTHON) lacks one of yaml fastapi uvicorn httpx numpy"
ver=$("$PLANNER_ENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null)
case $ver in 0.28.*) ok "vLLM $ver" ;; "") bad "vllm does not import in PLANNER_ENV" ;; *) warn "vLLM $ver (the paper used 0.28.0)" ;; esac
"$CTL_PYTHON" - "$AGENTICROBORL_DIR" "$HERE" <<'PY' && ok "harness config renders" || bad "harness config does not render"
import sys
from pathlib import Path
repo, here = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "rl"))
import harness_cfg as HC
cfg = HC.apply_vars(HC.load(repo / "harnesses/rpent_libero_c/harness.yaml"), {"proxy_port": "1", "model_port": "2"})
for stage in ("c0", "c0_ext", "c1", "c2", "smoke"):
    t = HC.load_tasks(here / "tasks" / f"{stage}.jsonl")
v = HC.episode_vars(cfg, t[0], 45, 0, Path("/tmp/x"), "/w", {"gpu": 0, "vla_port": 3, "sam3_port": 4, "oracle_info": "hidden",
                    "oracle_dynamics": "free", "vllm_mem": 0.5, "max_model_len": 1, "max_turns": 1, "round": 0})
HC.fmt(cfg["episode"]["cmd"], "cmd", **v); HC.fmt(cfg["episode"]["result_cmd"], "result_cmd", **v)
PY

echo "GPUs"
if command -v nvidia-smi > /dev/null; then
  for g in $GPUS; do
    line=$(nvidia-smi -i "$g" --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null)
    [ -n "$line" ] && ok "GPU $g: $line" || bad "GPU $g not visible"
  done
else
  bad "nvidia-smi not found"
fi
for g in $GPUS; do
  for base in "$PORT_BASE_MODEL" "$PORT_BASE_PROXY" "$PORT_BASE_VLA" "$PORT_BASE_SAM3"; do
    port=$((base + g))
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then warn "port $port is already in use"; fi
  done
done

[ $fails -eq 0 ] && echo "preflight OK" || echo "preflight: $fails check(s) failed"
exit $((fails > 0))

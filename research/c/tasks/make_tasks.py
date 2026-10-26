#!/usr/bin/env python3
"""Write the tasks.jsonl subsets of plan C from harnesses/rpent_libero/tasks.jsonl (same 80 LIBERO-PRO tasks; only
the initial states run in each stage differ). Training inits (init_ids 11-44) are untouched, so every file passes
rl/harness_cfg.py's disjointness check.

  c0.jsonl      C0 pilot          val_init_ids  = [45, 46]
  c0_ext.jsonl  C0 extension      val_init_ids  = [47, 48, 49]
  c1.jsonl      C1 main audit     eval_init_ids = [1..10]  (the paper's evaluation inits)
  c2.jsonl      C2 2x2 cells      eval_init_ids = [1..5]
  smoke.jsonl   leak self-check   4 tasks (one per suite family), val_init_ids = [45]

  python research/c/tasks/make_tasks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "harnesses" / "rpent_libero" / "tasks.jsonl"
STAGES = {
    "c0": {"val_init_ids": [45, 46]},
    "c0_ext": {"val_init_ids": [47, 48, 49]},
    "c1": {"eval_init_ids": list(range(1, 11))},
    "c2": {"eval_init_ids": list(range(1, 6))},
    "smoke": {"val_init_ids": [45]},
}
SMOKE_TASKS = ("libero_object_task/0", "libero_goal_task/0", "libero_spatial_task/0", "libero_10_task/0")


def main():
    tasks = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    assert len(tasks) == 80, len(tasks)
    for name, override in STAGES.items():
        chosen = [t for t in tasks if name != "smoke" or t["task_id"] in SMOKE_TASKS]
        with open(HERE / f"{name}.jsonl", "w") as f:
            for t in chosen:
                f.write(json.dumps({**t, **override}) + "\n")
        print(f"{name}.jsonl: {len(chosen)} tasks, {override}")


if __name__ == "__main__":
    main()

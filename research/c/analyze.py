#!/usr/bin/env python3
"""Analysis of plan C (research/PLAN_C_then_A.md sections 1.4, 1.8, 2.4-2.5).

Reads every episode under <runs>/<stage>/<cell>/ (cell.json written by run_cell.sh: planner, oracle info, oracle
dynamics), and writes to --out:

  episodes.csv   one row per episode (outcomes, behaviour, self-assessments)
  summary.json   everything below, machine-readable
  report.md      the same as tables

Outcomes per episode (from the transcript's "oracle" block, patches/rpent/rpent_oracle_c.patch):
  Y_sticky   the task succeeded at some env step (the paper's success)
  Y_final    the success predicate held when the planner called finish (or at the end, without finish)
  claimed    finish was called with a status starting with "succ"
  FC         false claim: claimed and not Y_final
  MS         missed success: Y_sticky and not (claimed and Y_final)
  NF         no finish call (turn / time budget ran out, or an error)
  timed_out  the planner hit RPent's --planner-timeout-s (a subset of NF; wall-clock, so it depends on the GPU and on
             whatever else runs on it -- compare it across the cells of a contrast)
Infrastructure failures (rl/adapters/rpent_libero.py) are left out; pairs need both episodes.

Conditions: V = visible/coupled, M-all = hidden/free, M-info = hidden/coupled, M-dyn = visible/free,
M-field = fields/coupled, M-prompt = prompt/coupled.

C0 decision (section 2.5): endpoints per planner, paired V vs M-all by (task, init, attempt), task-cluster bootstrap
CI, two-sided cluster sign-flip p, Holm over all planners x endpoints:
  final_drop   Y_final(V) - Y_final(M-all)    threshold 0.05, lower bound 0.03
  fc_increase  FC(M-all) - FC(V)              threshold 0.10, lower bound 0.05
  nf_increase  NF(M-all) - NF(V)              threshold 0.10, lower bound 0.05
  C_HOLDS  some endpoint >= threshold with Holm p < 0.05
  C_WEAK   every endpoint of every planner below its lower bound
  EXPAND   anything else (middle zone, or a threshold met without significance): run c0_ext and decide again on the
           pooled stages
A0 decision (section 2.5), on the M-all episodes, per self-assessment (p_probe, f_vlm, f_status) against Y_final:
  A_GATE         some signal with AUROC in [0.65, 0.97] and ECE > 0.05: run the online gate (section 2.6)
  A_GATE_LOW_ROOM  informative (AUROC in [0.65, 0.97]) but already calibrated (ECE <= 0.05)
  A_TOO_EASY     some signal with AUROC > 0.97 and ECE < 0.03 and none in the gate range: harder setting
  A_WEAK_SIGNAL  every AUROC < 0.65: add the verify tool / wrist view first (section 2.5)

  python research/c/analyze.py --runs runs/c --stages c0 --out runs/c/analysis/c0
  python research/c/analyze.py --runs runs/c --stages c0 c0_ext --out runs/c/analysis/c0_pooled
  python research/c/analyze.py --runs runs/c --stages c1 c2 --out runs/c/analysis/c1_c2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "rl"))
import stats as S  # noqa: E402
from adapters import rpent_libero as RA  # noqa: E402

CONDITIONS = {
    ("visible", "coupled"): "V",
    ("hidden", "free"): "M-all",
    ("hidden", "coupled"): "M-info",
    ("visible", "free"): "M-dyn",
    ("fields", "coupled"): "M-field",
    ("prompt", "coupled"): "M-prompt",
}
ENDPOINTS = {  # name: (outcome, sign: +1 = M-all minus V, -1 = V minus M-all), threshold, lower bound
    "final_drop": ("Y_final", -1, 0.05, 0.03),
    "fc_increase": ("FC", +1, 0.10, 0.05),
    "nf_increase": ("NF", +1, 0.10, 0.05),
}
SECONDARY = {"sticky_drop": ("Y_sticky", -1), "turns_change": ("turns", +1), "ms_increase": ("MS", +1)}
SIGNALS = ("p_probe", "f_vlm", "f_status")
_CELL = re.compile(r"^(?P<planner>[^_]+(?:_[^_]+)*)__(?P<info>[a-z]+)-(?P<dyn>[a-z]+)$")


def load_json_last(path: Path):
    text = path.read_text()
    try:
        return json.loads(text)
    except ValueError:
        dec, i, last = json.JSONDecoder(), 0, None
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            last, i = dec.raw_decode(text, i)
        return last


def cell_meta(cell_dir: Path) -> dict | None:
    meta = cell_dir / "cell.json"
    if meta.is_file():
        return json.loads(meta.read_text())
    m = _CELL.match(cell_dir.name)
    return {"planner": m["planner"], "info": m["info"], "dynamics": m["dyn"]} if m else None


def opt_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def episode_row(tr_path: Path, stage: str, meta: dict) -> dict | None:
    v = RA.read_verdict(tr_path)
    tr = load_json_last(tr_path) or {}
    o = tr.get("oracle") or {}
    ep = tr_path.parent
    stats = tr.get("stats") or {}
    status = str(o.get("finish_status") or "").strip().lower()
    finish = bool(o.get("finish_called"))
    claimed = finish and status.startswith("succ")
    y_sticky = bool(o["sticky_success"]) if "sticky_success" in o else bool(tr.get("solved"))
    y_final = o.get("final_success")
    log = o.get("log") or []
    first = next((i for i, e in enumerate(log) if e.get("success_ever")), None)
    probe, judge = opt_json(ep / "self_probe.json"), opt_json(ep / "vlm_judge.json")
    return {
        "stage": stage, "planner": meta["planner"], "info": meta["info"], "dynamics": meta["dynamics"],
        "condition": CONDITIONS.get((meta["info"], meta["dynamics"]), f"{meta['info']}/{meta['dynamics']}"),
        "task_id": f"{v['suite']}/{v['task']}", "init": v["seed"], "attempt": v["attempt"],
        "valid": v["failure_kind"] != "infrastructure" and y_final is not None,
        "failure_kind": v["failure_kind"], "patched": bool(o),
        "Y_sticky": int(y_sticky), "Y_final": int(bool(y_final)) if y_final is not None else None,
        "finish_called": int(finish), "finish_status": status or None, "claimed": int(claimed),
        "FC": int(claimed and not bool(y_final)), "MS": int(y_sticky and not (claimed and bool(y_final))),
        "NF": int(not finish), "timed_out": int("timed out" in str(v.get("error") or "").lower()),
        "turns": stats.get("turns_used"), "tool_calls": stats.get("tool_calls"),
        "requests": stats.get("requests"), "elapsed_s": tr.get("elapsed_s"),
        "calls_after_success": (len(log) - first - 1) if first is not None else None,
        "first_success_env_step": o.get("first_success_env_step"), "env_steps": o.get("env_steps"),
        "p_probe": probe.get("p_probe"), "f_vlm": judge.get("p_vlm"),
        "f_status": float(claimed) if finish else 0.0, "episode_dir": str(ep),
    }


def collect(runs: Path, stages: list[str]) -> list[dict]:
    rows = []
    for stage in stages:
        sdir = runs / stage
        if not sdir.is_dir():
            raise SystemExit(f"no stage directory {sdir}")
        for cell in sorted(p for p in sdir.iterdir() if p.is_dir()):
            meta = cell_meta(cell)
            if meta is None:
                continue
            for tr in sorted(cell.rglob("transcript_*.json")):
                if ".stale_" in tr.as_posix() or "/attempt_" not in tr.as_posix():
                    continue
                row = episode_row(tr, stage, meta)
                if row:
                    rows.append(row)
    return rows


def key(r: dict) -> tuple:
    return (r["task_id"], r["init"], r["attempt"])


def cells(rows: list[dict]) -> dict:
    """(planner, condition) -> {episode key: row} (valid episodes; a repeated key keeps the first)."""
    out: dict = defaultdict(dict)
    for r in rows:
        if r["valid"]:
            out[(r["planner"], r["condition"])].setdefault(key(r), r)
    return out


def summarise_cells(rows: list[dict]) -> list[dict]:
    by: dict = defaultdict(list)
    for r in rows:
        by[(r["planner"], r["condition"])].append(r)
    out = []
    for (planner, cond), rs in sorted(by.items()):
        val = [r for r in rs if r["valid"]]
        n = len(val)
        row = {"planner": planner, "condition": cond, "episodes": len(rs), "valid": n,
               "infrastructure": sum(r["failure_kind"] == "infrastructure" for r in rs),
               "unpatched": sum(not r["patched"] for r in rs)}
        for k in ("Y_sticky", "Y_final", "claimed", "FC", "MS", "NF", "timed_out"):
            s = sum(r[k] for r in val)
            row[k] = s / n if n else None
            if k in ("Y_sticky", "Y_final"):
                row[k + "_ci"] = list(S.wilson(s, n))
        turns = [r["turns"] for r in val if isinstance(r["turns"], (int, float))]
        row["turns_mean"] = sum(turns) / len(turns) if turns else None
        row["oracle_gap"] = (row["Y_sticky"] - row["Y_final"]) if n else None
        for sig in ("p_probe", "f_vlm"):
            row[sig + "_coverage"] = (sum(r[sig] is not None for r in val) / n) if n else None
        out.append(row)
    return out


def paired(c: dict, planner: str, a: str, b: str, outcome: str, sign: int) -> tuple[list, list, list]:
    """(clusters, diffs, keys) of (b - a) * ... : sign +1 -> outcome(b) - outcome(a); sign -1 -> outcome(a) - outcome(b)."""
    A, B = c.get((planner, a), {}), c.get((planner, b), {})
    ks = sorted(set(A) & set(B), key=str)
    cl, d = [], []
    for k in ks:
        x, y = A[k][outcome], B[k][outcome]
        if x is None or y is None:
            continue
        cl.append(k[0])
        d.append((y - x) if sign > 0 else (x - y))
    return cl, d, ks


def c0_decision(c: dict, planners: list[str], seed: int) -> dict:
    tests, pvals = {}, {}
    for p in planners:
        for name, (outcome, sign, thr, low) in {**ENDPOINTS, **{k: (*v, None, None) for k, v in SECONDARY.items()}}.items():
            cl, d, _ = paired(c, p, "V", "M-all", outcome, sign)
            res = S.paired_effect(cl, d, seed=seed)
            res.update(threshold=thr, lower_bound=low)
            tests[f"{p}:{name}"] = res
            if name in ENDPOINTS:
                pvals[f"{p}:{name}"] = res["p"]
    adj = S.holm(pvals)
    for k, v in adj.items():
        tests[k]["p_holm"] = v
    primary = [tests[k] for k in pvals if tests[k]["mean"] is not None]
    if not primary:
        return {"decision": "NO_DATA", "tests": tests}
    holds = any(t["mean"] >= t["threshold"] and (t["p_holm"] or 1) < 0.05 for t in primary)
    weak = all(t["mean"] < t["lower_bound"] for t in primary)
    decision = "C_HOLDS" if holds else "C_WEAK" if weak else "EXPAND"
    did = {}
    if len(planners) >= 2 and {"sft", "rl"} <= set(planners):
        for name, (outcome, sign, *_rest) in ENDPOINTS.items():
            cl_r, d_r, k_r = paired(c, "rl", "V", "M-all", outcome, sign)
            cl_s, d_s, k_s = paired(c, "sft", "V", "M-all", outcome, sign)
            m_r, m_s = dict(zip(k_r, d_r)), dict(zip(k_s, d_s))
            common = sorted(set(m_r) & set(m_s), key=str)
            did[name] = S.bootstrap_diff_of_diffs([k[0] for k in common], [m_r[k] for k in common],
                                                  [m_s[k] for k in common], seed=seed)
    return {"decision": decision, "tests": tests, "rl_minus_sft": did}


def decomposition(c: dict, planners: list[str], seed: int) -> dict:
    """C2: information and dynamics channel effects (only where the cells exist)."""
    out = {}
    plan = {
        "info_effect_coupled (V - M-info)": ("V", "M-info"),
        "dyn_effect_hidden (M-info - M-all)": ("M-info", "M-all"),
        "dyn_effect_visible (V - M-dyn)": ("V", "M-dyn"),
        "info_effect_free (M-dyn - M-all)": ("M-dyn", "M-all"),
        "field_effect (V - M-field)": ("V", "M-field"),
        "prompt_effect (V - M-prompt)": ("V", "M-prompt"),
    }
    for p in planners:
        for label, (a, b) in plan.items():
            if (p, a) not in c or (p, b) not in c:
                continue
            for outcome, sign in (("Y_final", -1), ("FC", +1), ("NF", +1)):
                cl, d, _ = paired(c, p, a, b, outcome, sign)
                out[f"{p}:{label}:{outcome}"] = S.paired_effect(cl, d, seed=seed)
    return out


def a0(c: dict, planners: list[str], seed: int, qs=(0.05, 0.10, 0.20)) -> dict:
    out = {"per_planner": {}, "pp": {}}
    decisions = []
    for p in planners:
        eps = list(c.get((p, "M-all"), {}).values())
        if not eps:
            continue
        cal = {}
        for sig in SIGNALS:
            pairs = [(r[sig], r["Y_final"]) for r in eps if r[sig] is not None and r["Y_final"] is not None]
            cal[sig] = S.calibration([x for x, _ in pairs], [y for _, y in pairs]) if pairs else {"n": 0}
            cal[sig]["coverage"] = len(pairs) / len(eps)
        out["per_planner"][p] = cal
        aurocs = {s: cal[s].get("auroc") for s in SIGNALS}
        eces = {s: cal[s].get("ece") for s in SIGNALS}
        if any(a is not None and 0.65 <= a <= 0.97 and eces[s] > 0.05 for s, a in aurocs.items()):
            decisions.append("A_GATE")
        elif any(a is not None and 0.65 <= a <= 0.97 for a in aurocs.values()):
            decisions.append("A_GATE_LOW_ROOM")
        elif any(a is not None and a > 0.97 and eces[s] < 0.03 for s, a in aurocs.items()):
            decisions.append("A_TOO_EASY")
        else:
            decisions.append("A_WEAK_SIGNAL")
        # semi-synthetic prediction-powered variance (fixed episodes, resampled label masks)
        for sig in ("p_probe", "f_vlm", "mean(p_probe,f_vlm)"):
            if sig == "mean(p_probe,f_vlm)":
                vals = [((r["p_probe"] + r["f_vlm"]) / 2) if r["p_probe"] is not None and r["f_vlm"] is not None
                        else None for r in eps]
            else:
                vals = [r[sig] for r in eps]
            keep = [(v, r) for v, r in zip(vals, eps) if v is not None and r["Y_final"] is not None]
            if len(keep) < 0.9 * len(eps) or len(keep) < 20:
                continue
            for q in qs:
                out["pp"][f"{p}:{sig}:q={q}"] = S.pp_resample([r["Y_final"] for _, r in keep], [v for v, _ in keep],
                                                              [r["task_id"] for _, r in keep], q, seed=seed)
    order = ["A_GATE", "A_GATE_LOW_ROOM", "A_TOO_EASY", "A_WEAK_SIGNAL"]
    out["decision"] = min(decisions, key=order.index) if decisions else "NO_DATA"
    out["decision_per_planner"] = dict(zip([p for p in planners if c.get((p, "M-all"))], decisions))
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def report(summary: dict) -> str:
    L = [f"# Plan C analysis: {', '.join(summary['stages'])}", ""]
    L += ["## Cells", "", "| planner | condition | valid / episodes | Y_sticky | Y_final [95% CI] | claimed | FC | MS | NF | "
          "timeouts | turns | oracle gap | probe cov. | judge cov. |", "|" + "---|" * 14]
    for r in summary["cells"]:
        ci = r.get("Y_final_ci") or [None, None]
        L.append(f"| {r['planner']} | {r['condition']} | {r['valid']} / {r['episodes']} | {fmt(r['Y_sticky'])} | "
                 f"{fmt(r['Y_final'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(r['claimed'])} | {fmt(r['FC'])} | "
                 f"{fmt(r['MS'])} | {fmt(r['NF'])} | {fmt(r['timed_out'])} | {fmt(r['turns_mean'], 1)} | "
                 f"{fmt(r['oracle_gap'])} | "
                 f"{fmt(r['p_probe_coverage'], 2)} | {fmt(r['f_vlm_coverage'], 2)} |")
    c0 = summary.get("c0")
    if c0 and c0.get("tests"):
        L += ["", f"## C decision: **{c0['decision']}**", "",
              "| planner:endpoint | pairs (tasks) | mean | 95% CI | p | p (Holm) | threshold |", "|" + "---|" * 7]
        for k, t in c0["tests"].items():
            L.append(f"| {k} | {t['n']} ({t.get('n_clusters')}) | {fmt(t['mean'])} | [{fmt(t['ci'][0])}, {fmt(t['ci'][1])}] "
                     f"| {fmt(t['p'], 4)} | {fmt(t.get('p_holm'), 4)} | {fmt(t.get('threshold'))} |")
        if c0.get("rl_minus_sft"):
            L += ["", "### RL amplification (effect under RL minus effect under SFT)", "",
                  "| endpoint | pairs | mean | 95% CI | p |", "|---|---|---|---|---|"]
            for k, t in c0["rl_minus_sft"].items():
                L.append(f"| {k} | {t['n']} | {fmt(t['mean'])} | [{fmt(t['ci'][0])}, {fmt(t['ci'][1])}] | {fmt(t['p'], 4)} |")
    if summary.get("decomposition"):
        L += ["", "## Information / dynamics decomposition", "", "| effect | pairs | mean | 95% CI | p |",
              "|---|---|---|---|---|"]
        for k, t in summary["decomposition"].items():
            L.append(f"| {k} | {t['n']} | {fmt(t['mean'])} | [{fmt(t['ci'][0])}, {fmt(t['ci'][1])}] | {fmt(t['p'], 4)} |")
    a = summary.get("a0")
    if a and a.get("per_planner"):
        L += ["", f"## A0 decision: **{a['decision']}** {a.get('decision_per_planner')}", "",
              "| planner | signal | coverage | AUROC | ECE | Brier | reliability | resolution | base rate |",
              "|" + "---|" * 9]
        for p, cal in a["per_planner"].items():
            for sig, m in cal.items():
                L.append(f"| {p} | {sig} | {fmt(m.get('coverage'), 2)} | {fmt(m.get('auroc'))} | {fmt(m.get('ece'))} | "
                         f"{fmt(m.get('brier'))} | {fmt(m.get('reliability'))} | {fmt(m.get('resolution'))} | "
                         f"{fmt(m.get('base_rate'))} |")
        if a.get("pp"):
            L += ["", "### Semi-synthetic prediction-powered variance (fixed episodes, resampled label masks)", "",
                  "var ratio: PP / label-only variance of the success-rate estimate; noise ratio: share of the "
                  "per-episode label noise left after the predictor (= 1 - corr^2).", "",
                  "| planner:signal:q | n | corr | var(label-only) | var(PP) | var ratio | noise ratio | Brier(signal) |",
                  "|" + "---|" * 8]
            for k, m in a["pp"].items():
                L.append(f"| {k} | {m['n']} | {fmt(m['corr'])} | {fmt(m['var_label_only'], 5)} | {fmt(m['var_pp'], 5)} | "
                         f"{fmt(m['var_ratio'])} | {fmt(m['noise_ratio'])} | {fmt(m['brier_of_f'])} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="the runs directory of plan C (RUNS_DIR)")
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = collect(a.runs, a.stages)
    if not rows:
        raise SystemExit("no episodes found")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = cells(rows)
    planners = sorted({p for p, _ in c})
    summary = {"stages": a.stages, "n_episodes": len(rows), "cells": summarise_cells(rows),
               "c0": c0_decision(c, planners, a.seed), "decomposition": decomposition(c, planners, a.seed),
               "a0": a0(c, planners, a.seed)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (a.out / "report.md").write_text(report(summary))
    print(report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

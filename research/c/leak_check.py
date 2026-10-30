#!/usr/bin/env python3
"""Leak and consistency checks of plan C (research/PLAN_C_then_A.md section 2.2): run them on the first episodes of
every condition before the real runs, and on every finished stage before it is analysed.

For every episode (a directory with RPent's transcript_*.json, its proxy records rl_turns/ and states.json):

  patched     the transcript has the "oracle" block of patches/rpent/rpent_oracle_c.patch
  records     one proxy record per planner request (stats.requests)
  (a) text    oracle_info hidden|prompt: no oracle term in what the planner was given -- system / user messages, the
              tool descriptions, and tool results (a read of a file the planner wrote itself is reported apart:
              the planner may use the word in its own audit)
  (b) fields  oracle_info hidden|fields: no oracle field in any tool result or in states.json; the run's own scrub
              counter (oracle.scrub_residual) is 0
  (c) free    oracle_dynamics free: no env step refused after the first success
  (d) coupled oracle_dynamics coupled: no env step after the end of the action chunk in which success first held
              (env_steps - first_success_env_step < VLA_CHUNK). A pi0.5 chunk runs as one RPC, so the original
              protocol lets the rest of that chunk execute and can undo the success; such an episode is a warning
              (finish then sees final_success false), not a failure

  python research/c/leak_check.py runs/c/c0 [--json out.json]
Exit code 1 when any check fails.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(
    r"(?<![a-z])terminat|task_success|libero_terminated|success_once|success_at_end|signaled term", re.IGNORECASE
)
ORACLE_KEYS = {"terminated", "truncated", "libero_terminated", "task_success", "success_once", "success_at_end"}
REFUSED = re.compile(r"signaled term|action rejected by the environment", re.IGNORECASE)
VLA_CHUNK = 5  # env steps per pi0.5 action chunk (RPent pi05_vla_server num_action_chunks)


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


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def keys_in(obj) -> set:
    out = set()
    if isinstance(obj, dict):
        out |= set(obj)
        for v in obj.values():
            out |= keys_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= keys_in(v)
    return out


def check_episode(ep: Path, tr_path: Path) -> dict:
    res = {"episode": str(ep), "info": None, "dynamics": None, "fail": [], "warn": []}
    tr = load_json_last(tr_path) or {}
    o = tr.get("oracle")
    if not isinstance(o, dict):
        res["fail"].append("patched: transcript has no oracle block (RPent without rpent_oracle_c.patch?)")
        return res
    info, dyn = o.get("info"), o.get("dynamics")
    res["info"], res["dynamics"] = info, dyn
    hide_text, hide_fields = info in ("hidden", "prompt"), info in ("hidden", "fields")
    turns = sorted((ep / "rl_turns").glob("turn_*.json.gz"))
    requests = (tr.get("stats") or {}).get("requests")
    if isinstance(requests, int) and requests != len(turns):
        res["fail"].append(f"records: {len(turns)} proxy records for {requests} requests")
    recs = [json.loads(gzip.decompress(t.read_bytes())) for t in turns]

    # (a) / (b) everything the planner was given
    hits_text, hits_field, hits_self = [], [], []
    seen_ids = set()
    for k, rec in enumerate(recs):
        names = {}
        for m in rec.get("messages") or []:
            for c in m.get("tool_calls") or []:
                names[c.get("id")] = (c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")
        if hide_text and k == 0:
            tools = (rec.get("request") or {}).get("tools") or []
            if FORBIDDEN.search(json.dumps(tools)):
                hits_text.append("tool descriptions")
        for i, m in enumerate(rec.get("messages") or []):
            role, txt = m.get("role"), text_of(m.get("content"))
            mid = (role, m.get("tool_call_id"), i if role != "tool" else None, txt[:200])
            if mid in seen_ids:
                continue  # the same history message, seen in an earlier record
            seen_ids.add(mid)
            if role in ("system", "user") and hide_text and FORBIDDEN.search(txt):
                hits_text.append(f"{role} message (record {k})")
            if role != "tool":
                continue
            name, args = names.get(m.get("tool_call_id"), (None, None))
            own_file = name == "read_text_file" and "guides" not in str(args or "")
            if hide_fields:
                try:
                    keys = keys_in(json.loads(txt))
                except ValueError:
                    keys = {k_ for k_ in ORACLE_KEYS if f'"{k_}"' in txt}
                if keys & ORACLE_KEYS and not own_file:
                    hits_field.append(f"{name} result (record {k}): {sorted(keys & ORACLE_KEYS)}")
            if (hide_text or hide_fields) and FORBIDDEN.search(txt):
                (hits_self if own_file else hits_text if hide_text else hits_field).append(f"{name} result (record {k})")
            if dyn == "free" and REFUSED.search(txt):
                res["fail"].append(f"free: a refused env step in a {name} result (record {k})")
    if hits_text:
        res["fail"].append(f"(a) text: {len(hits_text)} oracle term(s): {hits_text[:3]}")
    if hits_field:
        res["fail"].append(f"(b) fields: {len(hits_field)} oracle field(s): {hits_field[:3]}")
    if hits_self:
        res["warn"].append(f"the planner read back its own file containing an oracle word ({len(hits_self)}x)")
    if hide_fields:
        if o.get("scrub_residual"):
            res["fail"].append(f"(b) fields: scrub_residual = {o['scrub_residual']}")
        try:
            steps = json.loads((ep / "states.json").read_text()).get("steps") or []
            if keys_in(steps) & ORACLE_KEYS:
                res["fail"].append("(b) fields: states.json carries oracle fields")
        except (OSError, ValueError):
            res["warn"].append("no readable states.json")

    # (c) free dynamics: nothing refused after the first success
    log = o.get("log") or []
    first = next((i for i, e in enumerate(log) if e.get("success_ever")), None)
    if dyn == "free" and first is not None:
        n_err = sum(1 for e in log[first + 1:] if e.get("error"))
        if n_err:
            res["warn"].append(f"(c) free: {n_err} tool error(s) after the first success (check they are not refusals)")
    # (d) coupled dynamics: the env stops within the action chunk in which success first held
    if dyn == "coupled" and o.get("sticky_success"):
        first_step, steps = o.get("first_success_env_step"), o.get("env_steps")
        if not isinstance(first_step, int) or not isinstance(steps, int):
            res["warn"].append("(d) coupled: no first_success_env_step / env_steps in the oracle block")
        elif steps - first_step >= VLA_CHUNK:
            res["fail"].append(f"(d) coupled: {steps - first_step} env steps after the first success (step {first_step}),"
                               f" the env must stop within its action chunk ({VLA_CHUNK} steps)")
        elif o.get("finish_called") and o.get("final_success") is False:
            res["warn"].append(f"(d) coupled: success at env step {first_step} undone by the rest of its action chunk "
                               f"(stopped at {steps}); finish saw final_success false")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write every episode's result here")
    a = ap.parse_args()
    results = []
    for root in a.roots:
        for tr in sorted(root.rglob("transcript_*.json")):
            if ".stale_" not in tr.as_posix():
                results.append(check_episode(tr.parent, tr))
    by_cond = collections.defaultdict(lambda: {"episodes": 0, "failed": 0, "warned": 0})
    for r in results:
        c = by_cond[f"{r['info']}/{r['dynamics']}"]
        c["episodes"] += 1
        c["failed"] += bool(r["fail"])
        c["warned"] += bool(r["warn"])
    print(json.dumps(dict(by_cond), indent=1))
    for r in results:
        if r["fail"]:
            print("FAIL", r["episode"], *r["fail"], sep="\n  ")
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
    n_fail = sum(bool(r["fail"]) for r in results)
    print(f"{len(results)} episodes, {n_fail} with a failed check")
    return 1 if n_fail or not results else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Offline variants of the A0 self-assessment probe (research/PLAN_C_then_A.md section 2.6, precondition of A-G).

A0's probe (self_probe.py) asks after the planner's finish call, and its answer follows the finish status (AUROC of
p_probe and f_status agree to 0.003 in C0). These variants ask whether the planner knows more than it states:

  p_pre_yesno    the conversation up to the planner's last decision (its last request, before it answered -- for an
                 episode that called finish: before the finish call); "does the goal hold now? Yes or No";
                 P(yes) from the first answer token's logprobs
  p_pre_verbal   the same conversation; a probability in [0, 1] as text (A0's wording, before the decision)
  p_post_yesno   A0's conversation (after the finish call and its result); Yes or No; P(yes) from the logprobs

Same requests as A0 otherwise: the episode's own planner weights, its tools with tool_choice none, greedy, thinking
off. Run after the episodes against a vLLM server of the planner (research/c/run_offline.sh starts one), one planner
at a time: --planner picks the cells (cell.json) whose episodes that planner ran.

Writes <episode dir>/probe_variants.json; episodes that already have one are skipped (--force to redo).

  python research/c/probe_variants.py --model-url http://127.0.0.1:18600/v1 --model <served name> --planner sft runs/c/c0
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import self_probe as SP  # noqa: E402
import vlm_judge as VJ  # noqa: E402

PRE = "Pause before your next action and do not call any tool. Based only on what you have observed so far, "
POST = "The episode is over; do not call any tool. Based only on what you observed in this episode, "
YESNO = "does the task's goal condition hold in the current state right now? Answer with Yes or No only."
VERBAL = ("what is the probability (a number between 0 and 1) that the task's goal condition holds in the current "
          "state right now? Answer with the number only.")
VARIANTS = {"p_pre_yesno": (PRE + YESNO, "pre", "yesno"),
            "p_pre_verbal": (PRE + VERBAL, "pre", "verbal"),
            "p_post_yesno": (POST + YESNO, "post", "yesno")}


def conversation(record: dict, ep: Path, when: str) -> list[dict]:
    """The messages a variant's question is appended to (without the question)."""
    if when == "pre":
        return SP.restore_images(record.get("messages") or [], ep)
    messages, _ = SP.probe_messages(record, ep)
    return messages[:-1]  # drop A0's question


def ask(url: str, model: str, messages: list[dict], tools, kind: str, timeout_s: float) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    if kind == "yesno":
        body.update(max_tokens=1, logprobs=True, top_logprobs=20)
    else:
        body["max_tokens"] = 512
    if tools:
        body["tools"], body["tool_choice"] = tools, "none"
    resp = SP.post(url.rstrip("/") + "/chat/completions", body, timeout_s)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if kind == "yesno":
        p = VJ.p_yes_from_logprobs(choice)
        source = "logprobs"
        if p is None and text.strip():
            a = text.strip().lower()
            p, source = (1.0 if a.startswith("yes") else 0.0 if a.startswith("no") else None), "text"
        return {"p": p, "raw": text, "source": source}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    p = SP.parse_probability(text) if text else SP.parse_probability(reasoning, last=True)
    return {"p": p, "raw": text or reasoning, "source": "content" if text else "reasoning"}


def run(ep: Path, url: str, model: str, timeout_s: float = 600.0, retries: int = 3) -> dict:
    out: dict = {k: None for k in VARIANTS}
    out.update(model=model, n_records=0, error=None, details={})
    turns = sorted((ep / "rl_turns").glob("turn_*.json.gz"))
    out["n_records"] = len(turns)
    if not turns:
        out["error"] = "no proxy records"
        return out
    record = json.loads(gzip.decompress(turns[-1].read_bytes()))
    tools = (record.get("request") or {}).get("tools")
    errors = []
    for name, (question, when, kind) in VARIANTS.items():  # in this order, so the server's prefix cache is reused
        messages = conversation(record, ep, when) + [{"role": "user", "content": question}]
        for attempt in range(retries):
            try:
                res = ask(url, model, messages, tools, kind, timeout_s)
                out[name], out["details"][name] = res["p"], res
                break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                if attempt == retries - 1:
                    errors.append(f"{name}: {exc!r}")
                time.sleep(2.0 * (attempt + 1))
        if out[name] is None and name not in " ".join(errors):
            errors.append(f"{name}: no answer")
    out["error"] = "; ".join(errors) or None
    return out


def episodes(roots: list[Path], planner: str):
    for root in roots:
        for cell in sorted({p.parent for p in Path(root).rglob("cell.json")}):
            try:
                meta = json.loads((cell / "cell.json").read_text())
            except (OSError, ValueError):
                continue
            if meta.get("planner") != planner:
                continue
            for ep, _tr in VJ.episode_dirs([cell]):
                if "/attempt_" in ep.as_posix():
                    yield ep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="stage or cell directories")
    ap.add_argument("--model-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planner", required=True, help="only the cells this planner ran (cell.json)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0, help="probe at most this many episodes (a test run)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    todo = [ep for ep in episodes(a.roots, a.planner) if a.force or not (ep / "probe_variants.json").is_file()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} {a.planner} episodes to probe", flush=True)
    n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run, ep, a.model_url, a.model, a.timeout_s): ep for ep in todo}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            ep, res = futs[f], f.result()
            (ep / "probe_variants.json").write_text(json.dumps(res, indent=1))
            n_err += res["error"] is not None
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} probed, {n_err} with errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

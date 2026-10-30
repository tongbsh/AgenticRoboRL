#!/usr/bin/env python3
"""External success judge for plan C / A0 (research/PLAN_C_then_A.md section 2.3): an open VLM looks at the episode's
final third-person and wrist images plus the task language and gives P(task completed).

Which images: the step at which the planner called `finish` (transcript "oracle.finish_step"), else the last recorded
step. RPent stores step NN of artifact X as <episode dir>/X/NN.png; the high-resolution renders are preferred.
The task language comes from the run's states.json.

P(yes) is read from the first answer token's top logprobs (yes-like vs no-like tokens); when the server returns no
logprobs the answer text decides (1.0 / 0.0).

Run it after the episodes, against a vLLM server of the judge model (research/c/run_judge.sh starts one):
  python research/c/vlm_judge.py --model-url http://127.0.0.1:18500/v1 --model Qwen/Qwen3-VL-8B-Instruct runs/c/c0
Writes <episode dir>/vlm_judge.json; episodes that already have one are skipped (--force to redo). A second, stronger
judge writes beside it with --out-name vlm_judge_strong.json (research/c/run_offline.sh judge); analyze.py reads that
one as f_vlm_strong.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM = "You are a careful, strict judge of robot manipulation outcomes."
QUESTION = (
    "Task given to the robot: \"{task}\"\n"
    "The first image is a third-person view and the second image is the robot's wrist camera, both taken at the end "
    "of the episode. Is the task fully completed in this final state? Answer with Yes or No only."
)
YES = ("yes",)
NO = ("no",)
THIRD_PERSON = ("agentview_high.png", "agentview.png")
WRIST = ("wrist_high.png", "wrist.png")


def episode_dirs(roots: list[Path]):
    for root in roots:
        for tr in sorted(Path(root).rglob("transcript_*.json")):
            if ".stale_" not in tr.as_posix():
                yield tr.parent, tr


def load_transcript(path: Path) -> dict:
    """RPent appends its record; a re-run appends a second one -- the last one counts."""
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
        if last is None:
            raise
        return last


def final_step(tr: dict, ep: Path) -> int | None:
    o = tr.get("oracle") or {}
    for key in ("finish_step", "latest_step"):
        if isinstance(o.get(key), int):
            return o[key]
    try:
        steps = json.loads((ep / "states.json").read_text()).get("steps") or []
        return steps[-1]["step_idx"] if steps else None
    except (OSError, ValueError, KeyError):
        return None


def task_language(ep: Path) -> str | None:
    try:
        steps = json.loads((ep / "states.json").read_text()).get("steps") or []
    except (OSError, ValueError):
        return None
    for s in reversed(steps):
        lang = (s.get("extras") or {}).get("task_language")
        if lang:
            return lang
    return None


def artifact(ep: Path, names: tuple[str, ...], step: int) -> Path | None:
    for name in names:
        p = ep / name / f"{step:02d}.png"
        if p.is_file():
            return p
    return None


def data_url(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def p_yes_from_logprobs(choice: dict) -> float | None:
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if not content:
        return None
    yes = no = 0.0
    for cand in content[0].get("top_logprobs") or []:
        tok = (cand.get("token") or "").strip().lower()
        if tok in YES:
            yes += math.exp(cand["logprob"])
        elif tok in NO:
            no += math.exp(cand["logprob"])
    return yes / (yes + no) if yes + no > 0 else None


def judge(ep: Path, tr_path: Path, url: str, model: str, timeout_s: float) -> dict:
    out = {"p_vlm": None, "answer": None, "step": None, "images": [], "task": None, "model": model, "error": None}
    try:
        tr = load_transcript(tr_path)
    except (OSError, ValueError) as exc:
        out["error"] = f"transcript: {exc}"
        return out
    step = final_step(tr, ep)
    out["step"], out["task"] = step, task_language(ep)
    if step is None or not out["task"]:
        out["error"] = "no final step or no task language"
        return out
    imgs = [artifact(ep, THIRD_PERSON, step), artifact(ep, WRIST, step)]
    if imgs[0] is None:
        out["error"] = f"no third-person image at step {step}"
        return out
    imgs = [p for p in imgs if p is not None]
    out["images"] = [str(p.relative_to(ep)) for p in imgs]
    content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in imgs]
    content.append({"type": "text", "text": QUESTION.format(task=out["task"])})
    body = {"model": model, "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read())
            choice = resp["choices"][0]
            out["answer"] = (choice.get("message") or {}).get("content")
            p = p_yes_from_logprobs(choice)
            if p is None and out["answer"]:
                a = out["answer"].strip().lower()
                p = 1.0 if a.startswith("yes") else 0.0 if a.startswith("no") else None
            out["p_vlm"] = p
            out["error"] = None if p is not None else "no yes/no answer"
            return out
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            out["error"] = f"request failed: {exc!r}"
            time.sleep(2.0 * (attempt + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="stage or cell directories (searched for transcripts)")
    ap.add_argument("--model-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--out-name", default="vlm_judge.json", help="per-episode output file")
    ap.add_argument("--limit", type=int, default=0, help="judge at most this many episodes (a test run)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    todo = [(ep, tr) for ep, tr in episode_dirs(a.roots) if a.force or not (ep / a.out_name).is_file()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(todo)} episodes to judge", flush=True)
    n_err = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(judge, ep, tr, a.model_url, a.model, a.timeout_s): ep for ep, tr in todo}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            ep, res = futs[f], f.result()
            (ep / a.out_name).write_text(json.dumps(res, indent=1))
            n_err += res["error"] is not None
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} judged, {n_err} errors", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

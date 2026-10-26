#!/usr/bin/env python3
"""A0 self-assessment probe (plan C, research/PLAN_C_then_A.md section 2.3): after an episode, ask the planner -- with the
full context of its last request, as the recording proxy stored it -- how likely it is that the task's goal holds now.

The episode is over when this runs, so the probe changes nothing in it. The request goes to the model server directly,
not through the proxy, so the episode's proxy records keep exactly one record per planner call.

Reconstruction of the conversation (from <episode dir>/rl_turns/turn_NNN.json.gz, the last record):
  record["messages"]   the prompt of the last planner call (images restored from <episode dir>/rl_images/)
  record["message"]    the planner's answer to it (content + tool calls; its reasoning is not sent back, as in the run)
  tool results         for a `finish` call: the finish result; for any other call (the episode ended on its turn or time
                       budget): a note that the episode ended
  probe question       PROBE below
The model's tools are passed (so the prompt prefix matches and the server's prefix cache is hit) with tool_choice none,
greedy decoding and thinking off. The answer's first number in [0, 1] (or a percentage) is p_probe; when the model
answers only in its reasoning, the LAST such number there is used (source "reasoning"), so that coordinates the
model mentions on the way do not count.

Writes <episode dir>/self_probe.json: {"p_probe", "raw", "finish_called", "n_records", "error", ...}.

  python research/c/self_probe.py --episode-dir <dir> --model-url http://127.0.0.1:18100/v1 --model <served name>
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROBE = (
    "The episode is over; do not call any tool. Based only on what you observed in this episode, what is the "
    "probability (a number between 0 and 1) that the task's goal condition holds in the final state right now? "
    "Answer with the number only."
)
_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(%?)")


def parse_probability(text: str | None, last: bool = False) -> float | None:
    """The first (or ``last``) number in ``text`` that reads as a probability: 0..1, or 0..100 followed by '%'."""
    if not text:
        return None
    matches = list(_NUM.finditer(text))
    for m in (reversed(matches) if last else matches):
        x = float(m.group(1))
        if m.group(2) == "%":
            if 0.0 <= x <= 100.0:
                return x / 100.0
            continue
        if 0.0 <= x <= 1.0:
            return x
    return None


def restore_images(messages: list[dict], ep_dir: Path) -> list[dict]:
    """Proxy records store images as {"type": "image_ref", "path": "rl_images/<sha1>.<ext>", "mime": ...}."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_ref":
                raw = (ep_dir / part["path"]).read_bytes()
                url = f"data:{part.get('mime', 'image/png')};base64,{base64.b64encode(raw).decode()}"
                image_url = {"url": url}
                if part.get("detail") is not None:
                    image_url["detail"] = part["detail"]
                parts.append({"type": "image_url", "image_url": image_url})
            else:
                parts.append(part)
        out.append({**msg, "content": parts})
    return out


def probe_messages(record: dict, ep_dir: Path) -> tuple[list[dict], bool]:
    """(messages for the probe request, whether the last answer called finish)."""
    messages = restore_images(record.get("messages") or [], ep_dir)
    answer = record.get("message") or {}
    calls = answer.get("tool_calls") or []
    assistant = {"role": "assistant", "content": answer.get("content") or ""}
    if calls:
        assistant["tool_calls"] = calls
    messages.append(assistant)
    finish_called = False
    for call in calls:
        name = (call.get("function") or {}).get("name")
        if name == "finish":
            finish_called = True
            try:
                args = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except ValueError:
                args = {}
            content = json.dumps({"_finish": True, **(args if isinstance(args, dict) else {})})
        else:
            content = json.dumps({"note": "the episode ended before this tool ran"})
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
    messages.append({"role": "user", "content": PROBE})
    return messages, finish_called


def post(url: str, body: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def run(ep_dir: Path, model_url: str, model: str, timeout_s: float = 600.0, retries: int = 3) -> dict:
    out: dict = {"p_probe": None, "source": None, "raw": None, "finish_called": None, "n_records": 0, "error": None,
                 "model": model, "probe": PROBE}
    turns = sorted((ep_dir / "rl_turns").glob("turn_*.json.gz"))
    out["n_records"] = len(turns)
    if not turns:
        out["error"] = "no proxy records"
        return out
    record = json.loads(gzip.decompress(turns[-1].read_bytes()))
    messages, out["finish_called"] = probe_messages(record, ep_dir)
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}}
    tools = (record.get("request") or {}).get("tools")
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "none"
    last_exc = None
    for attempt in range(retries):
        try:
            resp = post(model_url.rstrip("/") + "/chat/completions", body, timeout_s)
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            out["raw"] = text if text else reasoning
            out["source"] = "content" if text else "reasoning"
            out["p_probe"] = parse_probability(text) if text else parse_probability(reasoning, last=True)
            out["usage"] = resp.get("usage")
            return out
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    out["error"] = f"request failed: {last_exc!r}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--model-url", required=True, help="the model server's OpenAI base URL (not the proxy's)")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args()
    out = run(a.episode_dir, a.model_url, a.model, a.timeout_s)
    (a.episode_dir / "self_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("p_probe", "finish_called", "n_records", "error")}))
    return 0 if out["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())

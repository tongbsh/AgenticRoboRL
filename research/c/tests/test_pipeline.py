"""analyze.py, leak_check.py, self_probe.py and vlm_judge.py on synthetic run directories (no RPent, no GPU)."""
from __future__ import annotations

import base64
import gzip
import json
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

import analyze as A
import leak_check as L
import self_probe as SP
import vlm_judge as VJ

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def write_episode(cell_dir: Path, stage: str, suite: str, task: int, seed: int, *, y_sticky: bool, y_final: bool,
                  finish: bool, status: str, info: str, dyn: str, p_probe=None, f_vlm=None, records=None):
    ep = cell_dir / "g0" / "eval" / stage / "eps" / "local" / suite / f"task_{task:02d}" / f"seed_{seed:03d}" / "attempt_00"
    ep.mkdir(parents=True)
    log = [{"tool": "move_to", "step": 1, "env_steps": 10, "success_now": y_sticky, "success_ever": y_sticky, "error": False}]
    oracle = {"info": info, "dynamics": dyn, "finish_called": finish, "finish_status": status if finish else None,
              "finish_step": 1 if finish else None, "final_success": y_final, "sticky_success": y_sticky,
              "scrub_residual": 0, "log": log, "latest_step": 1}
    records = records or []
    tr = {"suite": suite, "task": task, "seed": seed, "solved": y_sticky, "agent_error": None, "oracle": oracle,
          "stats": {"turns_used": 12, "tool_calls": 11, "requests": len(records) if records else None}, "messages": []}
    (ep / "transcript_x.json").write_text(json.dumps(tr))
    (ep / "rl_turns").mkdir()
    for i, rec in enumerate(records):
        (ep / "rl_turns" / f"turn_{i:03d}.json.gz").write_bytes(gzip.compress(json.dumps(rec).encode()))
    if p_probe is not None:
        (ep / "self_probe.json").write_text(json.dumps({"p_probe": p_probe}))
    if f_vlm is not None:
        (ep / "vlm_judge.json").write_text(json.dumps({"p_vlm": f_vlm}))
    return ep


def make_stage(root: Path, stage: str, drop: float, seed: int = 0, n_tasks: int = 20, inits=(45, 46)):
    rng = np.random.default_rng(seed)
    for planner, extra in (("sft", 0.0), ("rl", 0.1)):
        for info, dyn in (("visible", "coupled"), ("hidden", "free")):
            cell = root / stage / f"{planner}__{info}-{dyn}"
            cell.mkdir(parents=True)
            (cell / "cell.json").write_text(json.dumps({"planner": planner, "info": info, "dynamics": dyn}))
            for t in range(n_tasks):
                for s in inits:
                    u = np.random.default_rng(zlib.crc32(f"{planner}/{t}/{s}".encode())).random()  # same in both conditions
                    base = u < 0.7
                    hidden = info == "hidden"
                    y_final = bool(base and not (hidden and u < drop + (extra if planner == "rl" else 0)))
                    claimed = y_final or (hidden and rng.random() < 0.15)
                    p = float(np.clip((0.8 if y_final else 0.3) + rng.normal(0, 0.1), 0, 1))
                    write_episode(cell, stage, "libero_object_task", t, s, y_sticky=bool(base), y_final=y_final,
                                  finish=True, status="success" if claimed else "failure", info=info, dyn=dyn,
                                  p_probe=p if hidden else None, f_vlm=p if hidden else None)


def test_analysis_c_holds_with_a_large_drop(tmp_path):
    make_stage(tmp_path, "c0", drop=0.3)
    rows = A.collect(tmp_path, ["c0"])
    assert len(rows) == 2 * 2 * 20 * 2
    c = A.cells(rows)
    res = A.c0_decision(c, ["rl", "sft"], seed=0)
    assert res["decision"] == "C_HOLDS"
    assert res["tests"]["sft:final_drop"]["mean"] > 0.2
    assert res["rl_minus_sft"]["final_drop"]["mean"] > 0           # RL drops more (by construction)
    a0 = A.a0(c, ["rl", "sft"], seed=0)
    assert a0["decision"] in ("A_GATE", "A_GATE_LOW_ROOM", "A_TOO_EASY")
    assert a0["per_planner"]["sft"]["p_probe"]["auroc"] > 0.9
    assert any(k.endswith("q=0.1") for k in a0["pp"])


def test_analysis_c_weak_without_effect(tmp_path):
    make_stage(tmp_path, "c0", drop=0.0)
    rows = A.collect(tmp_path, ["c0"])
    for r in rows:   # no false claims, no effect at all
        r["FC"] = 0
        if r["planner"] == "rl":
            r["Y_final"] = next(x["Y_final"] for x in rows if x["planner"] == "rl" and x["condition"] == "V"
                                and A.key(x) == A.key(r))
    res = A.c0_decision(A.cells(rows), ["rl", "sft"], seed=0)
    assert res["decision"] == "C_WEAK"


def test_main_writes_report(tmp_path):
    make_stage(tmp_path, "c0", drop=0.3)
    import sys
    argv = sys.argv
    sys.argv = ["analyze.py", "--runs", str(tmp_path), "--stages", "c0", "--out", str(tmp_path / "out")]
    try:
        assert A.main() == 0
    finally:
        sys.argv = argv
    assert "C decision" in (tmp_path / "out" / "report.md").read_text()
    assert (tmp_path / "out" / "episodes.csv").read_text().count("\n") == 161


# ---------------------------------------------------------------------------
# leak_check
# ---------------------------------------------------------------------------

def rec(tool_content: str, system="You are a robot planner."):
    return {"request": {"tools": [{"type": "function", "function": {"name": "move_to", "description": "Move."}}]},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": "go"},
                         {"role": "assistant", "content": "", "tool_calls": [
                             {"id": "c1", "type": "function", "function": {"name": "move_to", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": tool_content}],
            "message": {"content": "ok"}}


def test_leak_check_passes_clean_and_flags_leaks(tmp_path):
    cell = tmp_path / "c0" / "sft__hidden-free"
    clean = write_episode(cell, "c0", "libero_goal_task", 1, 45, y_sticky=True, y_final=True, finish=True,
                          status="success", info="hidden", dyn="free", records=[rec(json.dumps({"step": 3}))])
    (clean / "states.json").write_text(json.dumps({"steps": [{"step_idx": 0, "state": {}}]}))
    assert L.check_episode(clean, clean / "transcript_x.json")["fail"] == []
    leaky = write_episode(cell, "c0", "libero_goal_task", 2, 45, y_sticky=True, y_final=True, finish=True,
                          status="success", info="hidden", dyn="free",
                          records=[rec(json.dumps({"step": 3, "terminated": True}),
                                       system="produce top-level `terminated == true`")])
    (leaky / "states.json").write_text(json.dumps({"steps": [{"step_idx": 0, "terminated": False}]}))
    fails = " ".join(L.check_episode(leaky, leaky / "transcript_x.json")["fail"])
    assert "(a) text" in fails and "(b) fields" in fails and "states.json" in fails
    unpatched = cell / "x"
    unpatched.mkdir()
    (unpatched / "transcript_y.json").write_text(json.dumps({"solved": True}))
    assert "patched" in L.check_episode(unpatched, unpatched / "transcript_y.json")["fail"][0]


def test_leak_check_coupled_final_success(tmp_path):
    cell = tmp_path / "c0" / "sft__visible-coupled"
    ep = write_episode(cell, "c0", "libero_goal_task", 1, 45, y_sticky=True, y_final=False, finish=True,
                       status="success", info="visible", dyn="coupled")
    assert any("(d) coupled" in f for f in L.check_episode(ep, ep / "transcript_x.json")["fail"])


# ---------------------------------------------------------------------------
# self_probe / vlm_judge against a stand-in OpenAI server
# ---------------------------------------------------------------------------

class _Server:
    def __init__(self, reply):
        self.bodies = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                data = json.dumps(reply).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()


@pytest.mark.parametrize("text,p", [("0.73", 0.73), ("About 85%", 0.85), ("1", 1.0), ("7 or 0.2", 0.2), ("none", None)])
def test_parse_probability(text, p):
    assert SP.parse_probability(text) == p


def test_parse_probability_from_reasoning_takes_the_last_number():
    assert SP.parse_probability("bowl at x=0.15, y=0.3 ... so I'd say 0.7", last=True) == 0.7


def test_self_probe_rebuilds_the_last_request(tmp_path):
    ep = tmp_path / "ep"
    (ep / "rl_images").mkdir(parents=True)
    (ep / "rl_images" / "abc.png").write_bytes(PNG)
    record = {"request": {"tools": [{"type": "function", "function": {"name": "finish"}}]},
              "messages": [{"role": "system", "content": "sys"},
                           {"role": "user", "content": [{"type": "text", "text": "look"},
                                                        {"type": "image_ref", "path": "rl_images/abc.png", "mime": "image/png"}]}],
              "message": {"role": "assistant", "content": "done", "reasoning_content": "secret",
                          "tool_calls": [{"id": "f1", "type": "function",
                                          "function": {"name": "finish", "arguments": "{\"status\": \"success\"}"}}]}}
    (ep / "rl_turns").mkdir()
    (ep / "rl_turns" / "turn_000.json.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
    srv = _Server({"choices": [{"message": {"content": "0.62"}}]})
    try:
        out = SP.run(ep, srv.url, "m")
    finally:
        srv.close()
    assert out["p_probe"] == 0.62 and out["finish_called"] is True and out["error"] is None
    body = srv.bodies[0]
    msgs = body["messages"]
    assert body["tool_choice"] == "none" and body["temperature"] == 0.0
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[2]["role"] == "assistant" and "reasoning_content" not in msgs[2]
    assert msgs[3]["role"] == "tool" and json.loads(msgs[3]["content"])["_finish"] is True
    assert msgs[4] == {"role": "user", "content": SP.PROBE}


def test_vlm_judge_reads_final_images_and_logprobs(tmp_path):
    cell = tmp_path / "c0" / "sft__hidden-free"
    ep = write_episode(cell, "c0", "libero_goal_task", 1, 45, y_sticky=True, y_final=True, finish=True,
                       status="success", info="hidden", dyn="free")
    (ep / "states.json").write_text(json.dumps({"steps": [{"step_idx": 0, "extras": {"task_language": "open the drawer"}},
                                                          {"step_idx": 1, "extras": {"task_language": "open the drawer"}}]}))
    for name in ("agentview_high.png", "wrist.png"):
        (ep / name).mkdir()
        (ep / name / "01.png").write_bytes(PNG)
    reply = {"choices": [{"message": {"content": "Yes"}, "logprobs": {"content": [{"token": "Yes", "logprob": -0.1,
             "top_logprobs": [{"token": "Yes", "logprob": float(np.log(0.6))}, {"token": "No", "logprob": float(np.log(0.2))}]}]}}]}
    srv = _Server(reply)
    try:
        out = VJ.judge(ep, ep / "transcript_x.json", srv.url, "judge", 30)
    finally:
        srv.close()
    assert out["p_vlm"] == pytest.approx(0.75) and out["step"] == 1 and out["task"] == "open the drawer"
    assert out["images"] == ["agentview_high.png/01.png", "wrist.png/01.png"]
    assert "open the drawer" in srv.bodies[0]["messages"][1]["content"][-1]["text"]

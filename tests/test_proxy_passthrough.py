"""rl/proxy.py: RL mode forces the RL sampling; --passthrough keeps the client's sampling (evaluations) and still
records the call. Uses the minimal harness's stand-in model server; no GPU."""
from __future__ import annotations

import gzip
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "rl"))
from fastapi.testclient import TestClient  # noqa: E402

import proxy as P  # noqa: E402
from rl_sampling import RL_SAMPLING  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def upstream():
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "harnesses" / "minimal" / "fake_model_server.py"), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


BODY = {"model": "m", "messages": [{"role": "user", "content": "go"}], "temperature": 0.7, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 1.5, "seed": 3407, "top_logprobs": 3, "logprobs": True}


def one_call(upstream: str, out: Path, passthrough: bool) -> dict:
    app = P.build_app(upstream, P.Recorder(out, "v0"), 30.0, passthrough=passthrough)
    with TestClient(app) as client:
        r = client.post("/ep/local/t0/seed_045/attempt_00/v1/chat/completions", json=BODY)
        assert r.status_code == 200, r.text
        assert "prompt_token_ids" not in r.json()                      # the client gets the usual shape back
    rec = sorted((out / "local/t0/seed_045/attempt_00/rl_turns").glob("turn_*.json.gz"))
    assert len(rec) == 1
    return json.loads(gzip.decompress(rec[0].read_bytes()))


def test_rl_mode_forces_rl_sampling(upstream, tmp_path):
    rec = one_call(upstream, tmp_path, passthrough=False)
    assert all(rec["sent_sampling"][k] == v for k, v in RL_SAMPLING.items())
    assert rec["sent_sampling"]["seed"] == 3407
    assert rec["output_token_ids"] and len(rec["output_logprobs"]) == len(rec["output_token_ids"])


def test_passthrough_keeps_client_sampling(upstream, tmp_path):
    rec = one_call(upstream, tmp_path, passthrough=True)
    s = rec["sent_sampling"]
    assert (s["temperature"], s["top_p"], s["top_k"], s["presence_penalty"], s["seed"]) == (0.7, 0.95, 20, 1.5, 3407)
    assert rec["prompt_token_ids"] and rec["output_token_ids"]          # still recorded


def test_harness_cfg_passthrough_needs_proxy(tmp_path):
    import harness_cfg as HC
    (tmp_path / "tasks.jsonl").write_text(json.dumps({"task_id": "t", "init_ids": [1]}) + "\n")
    base = "name: x\nadapter: result_json\ntasks: tasks.jsonl\nproxy_port: 1\nmodel: {url: 'http://h'}\nepisode: {cmd: c}\n"
    (tmp_path / "ok.yaml").write_text(base + "eval: {passthrough: true}\n")
    assert HC.load(tmp_path / "ok.yaml")["eval"]["passthrough"] is True
    (tmp_path / "bad.yaml").write_text(base + "eval: {passthrough: true, direct: true}\n")
    with pytest.raises(HC.ConfigError):
        HC.load(tmp_path / "bad.yaml")

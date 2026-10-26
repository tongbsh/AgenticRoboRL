"""patches/rpent/rpent_oracle_c.patch on a real LiberoToolkit driven by the fake LIBERO backend (tests/fake_libero.py).
Needs RPENT_DIR = an RPent checkout with both patches (research/c/setup_rpent.sh) and RPent's Python dependencies; no
simulator, no GPU. Skipped when RPENT_DIR is not set.

  RPENT_DIR=/path/to/RPent-c $RPENT_ENV/bin/python -m pytest -q research/c/tests/test_rpent_oracle.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if not os.environ.get("RPENT_DIR"):
    pytest.skip("RPENT_DIR not set", allow_module_level=True)

from fake_libero import META, SUCCESS_Z, FakeLiberoRpc, FakeVla  # noqa: E402
from robots.libero.env_client import LiberoEnvClient  # noqa: E402
from robots.libero.prompts import oracle_free as OF  # noqa: E402
from robots.libero.robot_spec import _parse_config, get_robot_spec  # noqa: E402
from robots.libero.toolkit import LiberoToolkit  # noqa: E402
from rpent.dashboard.events import NullDashboardEventSink  # noqa: E402
from rpent.utils.logging import init_output_dir  # noqa: E402

UP = {"xyz": [0.0, 0.0, 1.35], "max_steps": 40}       # crosses SUCCESS_Z on the way up
DOWN = {"xyz": [0.0, 0.0, 1.00], "max_steps": 40}     # back below it: success undone


def toolkit(tmp_path: Path, info: str, dyn: str) -> LiberoToolkit:
    init_output_dir(tmp_path)                                     # rpent.cli.main does this before any tool runs
    env = LiberoEnvClient(FakeLiberoRpc(), expected_meta=META)
    return LiberoToolkit(primitives_kwargs={"env": env, "model": FakeVla(), "sam3_client": None},
                         dashboard_events=NullDashboardEventSink(), state_output_dir=tmp_path,
                         knowledge_profile="proven", oracle_info=info, oracle_dynamics=dyn)


def keys(obj) -> set:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(keys(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(keys(v) for v in obj)) if obj else set()
    return set()


def no_images(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_image_bytes"}


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def test_visible_coupled_is_the_original_behaviour(tmp_path):
    tk = toolkit(tmp_path, "visible", "coupled")
    up = tk.execute_tool("move_to", UP).result
    assert up["terminated"] is True and up["log"]["result"]["terminated"] is True
    assert up["log"]["result"]["final_eef_pos"][2] < 1.35      # stopped as soon as the predicate held
    down = tk.execute_tool("move_to", DOWN).result
    assert "signaled term" in down["error"]                      # later env steps refused, with the original text
    states = json.loads((tmp_path / "states.json").read_text())["steps"]
    assert states[-1]["terminated"] is True
    tk.execute_tool("finish", {"status": "success", "summary": "s"})
    s = tk.oracle_summary()
    assert s["sticky_success"] and s["final_success"] and s["finish_status"] == "success"
    assert s["first_success_env_step"] == s["env_steps"]       # nothing ran after success


def test_free_dynamics_keeps_running_and_can_undo_success(tmp_path):
    tk = toolkit(tmp_path, "hidden", "free")
    up = tk.execute_tool("move_to", UP).result
    assert up["log"]["result"]["final_eef_pos"][2] == pytest.approx(1.35, abs=0.02)   # ran to its own target
    down = tk.execute_tool("move_to", DOWN).result
    assert "error" not in down
    tk.execute_tool("finish", {"status": "success", "summary": "s"})
    s = tk.oracle_summary()
    assert s["sticky_success"] is True and s["final_success"] is False and s["success_now_at_end"] is False
    assert s["final_success_source"] == "check_success"
    assert s["env_steps"] > s["first_success_env_step"]
    assert tk.solved()                                            # the paper's (sticky) success is unchanged


def test_visible_free_reports_the_current_predicate(tmp_path):
    tk = toolkit(tmp_path, "visible", "free")
    assert tk.execute_tool("move_to", UP).result["terminated"] is True
    assert tk.execute_tool("move_to", DOWN).result["terminated"] is False
    assert tk.execute_tool("view_env_state", {"step": -1}).result["terminated"] is False


def test_pi0_doubled_runs_on_when_free(tmp_path):
    tk = toolkit(tmp_path, "hidden", "free")
    r = tk.execute_tool("pi0_doubled", {"prompt": "push", "max_chunks": 8}).result["log"]["result"]
    assert r["chunks_used"] == 8                                  # no early stop at success (z passes SUCCESS_Z)
    assert "success" not in r and "task_success" not in r


def test_no_finish_records_the_end_state(tmp_path):
    tk = toolkit(tmp_path, "hidden", "free")
    tk.execute_tool("move_to", UP)
    s = tk.oracle_summary()
    assert s["finish_called"] is False and s["final_success"] is True
    assert s["final_success_source"].startswith("end_of_episode")


# ---------------------------------------------------------------------------
# information
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("info,dyn", [("hidden", "coupled"), ("hidden", "free"), ("fields", "coupled")])
def test_hidden_fields_never_reach_the_planner(tmp_path, info, dyn):
    tk = toolkit(tmp_path, info, dyn)
    results = [tk.execute_tool("move_to", UP).result, tk.execute_tool("move_to", DOWN).result,
               tk.execute_tool("pi0_pick", {"prompt": "pick", "max_chunks": 3}).result,
               tk.execute_tool("pi0_doubled", {"prompt": "push", "max_chunks": 2}).result,
               tk.execute_tool("view_env_state", {"step": -1}).result,
               tk.execute_tool("view_env_state", {"step": 0}).result]
    for r in results:
        assert not keys(no_images(r)) & OF.ORACLE_KEYS
        assert not OF.has_oracle_terms(no_images(r))
    if dyn == "coupled":
        assert results[1]["error"] == OF.REJECTED                 # refused, but without naming the termination
    states = json.loads((tmp_path / "states.json").read_text())
    assert not keys(states) & OF.ORACLE_KEYS
    denied = tk.execute_tool("read_text_file", {"path": str(tmp_path / "states.json")}).result
    assert "not readable" in denied["error"]
    tk.execute_tool("finish", {"status": "failure", "summary": "s"})
    s = tk.oracle_summary()
    assert s["scrub_residual"] == 0 and s["sticky_success"] is True


def test_pick_success_is_its_own_test_when_hidden(tmp_path):
    tk = toolkit(tmp_path, "hidden", "coupled")
    tk.execute_tool("move_to", {"xyz": [0.0, 0.0, SUCCESS_Z - 0.06], "max_steps": 40})
    r = tk.execute_tool("pi0_pick", {"prompt": "pick", "max_chunks": 3}).result["log"]["result"]
    assert r["success"] is False                                  # predicate fired mid-pick, but no lift/grip happened


def test_prompt_only_keeps_fields(tmp_path):
    tk = toolkit(tmp_path, "prompt", "coupled")
    assert tk.execute_tool("move_to", UP).result["terminated"] is True
    assert not OF.has_oracle_terms(tk.get_tools_spec())


@pytest.mark.parametrize("info", ["hidden", "prompt"])
def test_text_rewrite_tool_specs_and_guides(tmp_path, info):
    tk = toolkit(tmp_path, info, "coupled")
    assert not OF.has_oracle_terms(tk.get_tools_spec())
    for name in ("strict_hybrid_guide.md", "pro_hybrid_guide.md", "env_calibration.md"):
        r = tk.execute_tool("read_text_file", {"path": f"robots/libero/guides/{name}"}).result
        assert r.get("content") and not OF.has_oracle_terms(r["content"]), name


def test_visible_text_is_unchanged(tmp_path):
    tk = toolkit(tmp_path, "visible", "coupled")
    assert OF.has_oracle_terms(tk.get_tools_spec())
    r = tk.execute_tool("read_text_file", {"path": "robots/libero/guides/strict_hybrid_guide.md"}).result
    assert OF.has_oracle_terms(r["content"])


def _vars(info: str) -> dict:
    return {"knowledge_profile": "proven", "mode": "eval", "suite": "libero_object_swap", "task": 3, "seed": 45,
            "recipe_tag": "object_swap_t3_s45", "memory_profile": "hf", "memory_dir": "/m", "reference_tag": "r",
            "memory_inbox": "/i", "session_number": 1, "session_max": 1, "output_dir": "/out", "oracle_info": info}


def test_prompts():
    spec = get_robot_spec()
    for info in ("hidden", "prompt"):
        for variant in ("system", "user"):
            assert not OF.has_oracle_terms(spec.prompts.render(variant, variables=_vars(info)))
    for info in ("visible", "fields"):
        assert OF.has_oracle_terms(spec.prompts.render("system", variables=_vars(info)))


def test_guides_rewrite_strictly():
    guides = Path(os.environ["RPENT_DIR"]) / "robots" / "libero" / "guides"
    for g in guides.glob("*.md"):
        OF.rewrite_guide(g.name, g.read_text(), strict=True)


def test_config_needs_proven_profile():
    import argparse
    ns = argparse.Namespace(suite="libero_object_swap", task=3, seed=45, explore=False, memory_profile=None,
                            explore_sessions=1, knowledge_profile="mem-proven", memory_dir=None, output_dir="/tmp/x",
                            oracle_info="hidden", oracle_dynamics="coupled")
    with pytest.raises(ValueError):
        _parse_config(ns)
    ns.knowledge_profile = "proven"
    assert _parse_config(ns).prompt_vars["oracle_info"] == "hidden"

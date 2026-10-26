# Changes to RPent

`rpent_f70d720.patch` applies to RPent at commit `f70d720` and gives the RPent code all evaluations of the paper used
(8 files, +303 / -9 lines). The planner's tools and RPent's control loop are not changed.

```bash
git clone https://github.com/RLinf/RPent.git && cd RPent
git checkout f70d720
git apply --check /path/to/AgenticRoboRL/patches/rpent/rpent_f70d720.patch
git apply /path/to/AgenticRoboRL/patches/rpent/rpent_f70d720.patch
git add -A && git write-tree        # 11c44fcdedab126962be13ed3f477e0212932323
```

| Change | Files |
|---|---|
| `--knowledge-profile proven`: the zero-shot setting of Harness VLA. The planner gets the cross-task "proven levers" text whole and the guides, but no task memory and no global memory; its file tools can read only the guides and the episode's own output directory | `robots/libero/prompts/knowledge_eval.py` (new), `robots/libero/prompt_bundle.py`, `robots/libero/robot_spec.py`, `robots/libero/toolkit.py`, `rpent/cli/main.py`, `rpent/cli/dashboard.py` |
| `--no-send-back-thinking`: the planner's earlier reasoning is not sent back to the model in later turns; `--temperature`, `--top-p`, `--top-k`, `--presence-penalty`, `--generation-seed` passed to the model server | `rpent/planner/api_loop.py`, `rpent/planner/base.py`, `rpent/cli/main.py`, `rpent/cli/dashboard.py` |
| The transcript records `solved` (the simulator's success flag) and `agent_error`; `rl/adapters/rpent_libero.py` reads the reward from them | `rpent/cli/main.py` |

The RL rollouts of the paper ran on a later version of these files that additionally recorded simulator state for
an experiment not reported in the paper (off unless an environment variable is set; nothing of it reaches the planner).
That recording is not part of this patch.

sha256 of `rpent_f70d720.patch`: `74380a09251aa42f516457b1f410415d7d7cd1aee84d661886c164e18b0860b7`

## `rpent_oracle_c.patch` (research plan C)

Applies on top of `rpent_f70d720.patch` (`research/c/setup_rpent.sh` does both; tree after both:
`10a16d6ff9c04d3cbf14433ee7fe6728b9ec41e6`). With the defaults (`--oracle-info visible --oracle-dynamics coupled`)
the planner sees exactly what it sees with the first patch alone; the transcript additionally gets an `oracle` block.

| Change | Files |
|---|---|
| `--oracle-info visible\|hidden\|fields\|prompt`: what the planner learns about LIBERO's success predicate. `fields` removes `terminated` / `truncated` from every tool result, state view and `states.json`, drops `pi0_doubled`'s success flag, makes `pi0_pick` report only its own lift/grip test, replaces the refused-step error text, and blocks reads of `states.json`, logs, transcripts and proxy records; `prompt` rewrites the system prompt, the tool descriptions and the guides so that they no longer mention a success flag (exact fragments, checked for leftovers); `hidden` = both. Needs `--knowledge-profile proven` | `robots/libero/prompts/oracle_free.py` (new), `robots/libero/toolkit.py`, `robots/libero/tools.py`, `robots/libero/prompt_bundle.py`, `robots/libero/robot_spec.py`, `rpent/tools/state.py` |
| `--oracle-dynamics coupled\|free`: `free` keeps the environment running after success (primitives no longer stop at success and later env steps are no longer refused); success is only logged | `robots/libero/env_client.py`, `robots/libero/toolkit.py` |
| Success bookkeeping for the experimenter: per-step `success_now` / `success_ever`, first success step, an `env.check_success` RPC queried when the planner calls `finish`; the transcript's `oracle` block (`sticky_success`, `final_success`, `finish_status`, step log) | `robots/libero/env_client.py`, `robots/libero/env_server.py`, `robots/libero/toolkit.py`, `rpent/cli/main.py` |

Tests without a simulator: `research/c/tests/test_rpent_oracle.py`.

sha256 of `rpent_oracle_c.patch`: `55d323513b5d12dbc78b4087a446968d4fd843cdf591d88f9ebaad0b85bc1d11`

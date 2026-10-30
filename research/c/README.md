# 方案 C 实验代码：运行手册

方案 C（`research/PLAN_C_then_A.md` v2）研究的问题是：在 RPent/LIBERO-PRO 上，屏蔽 planner 在回合内能看到的仿真器成功信号之后，planner 的行为和成功率会怎样变化。效应会被拆成信息通道和动力学通道两部分，并比较 RL 与 SFT 两个 planner。同一批回合还附带 A0 自评探针，为方案 A 提供试点数据。

本目录包含从安装到出判定所需的全部代码。如果在服务器上用 Claude Code 执行，按 `CLAUDE_CODE_PROMPTS.md` 里的 prompt 逐周进行。服务器上只需要搭好环境、下载权重、填写 `c.env`，然后按本文的顺序执行命令。

## 1. 文件一览

| 文件 | 作用 |
|---|---|
| `../../patches/rpent/rpent_oracle_c.patch` | RPent 补丁（叠加在论文 patch 之上），提供 `--oracle-info {visible,hidden,fields,prompt}` 和 `--oracle-dynamics {coupled,free}` 两个开关，并在 transcript 中写入 `oracle` 字段，记录 sticky 成功、`finish` 时的成功、`finish` 的 status、逐步日志 |
| `../../harnesses/rpent_libero_c/harness.yaml` | C 的评测配置：沿用论文的评测采样（top-p 0.95 / top-k 20 / presence 1.5 / seed 3407），录制代理用 passthrough 模式（不改采样，只录制），每个回合结束后自动运行自评探针 |
| `setup_rpent.sh` | 克隆 RPent `f70d720`，依次打两个 patch，并校验 tree hash |
| `c.env.example` | 路径、端口、显存比例等配置，复制为 `c.env` 后填写 |
| `preflight.sh` | 开跑前的静态检查（约 1 分钟，不跑任何回合） |
| `smoke.sh` | 第 2 周的泄漏自检：SFT × 4 个任务 × 4 个单元，共 16 个回合，然后跑 `leak_check` |
| `run_cell.sh` | 一个单元（阶段 × planner × 条件），分到 `$GPUS` 上并行，支持断点续跑 |
| `run_stage.sh` | 一个阶段的所有单元，跑完后依次运行 VLM 判官、泄漏检查和分析 |
| `run_judge.sh` / `vlm_judge.py` | 外部 VLM 判官：读 `finish` 时的第三人称图和腕部图，输出 f_vlm |
| `self_probe.py` | A0 自评探针：回合结束后用 planner 的完整上下文问"成功概率是多少"，得到 p_probe |
| `leak_check.py` | 补丁是否生效、录制条数是否完整，以及 (a) 文本、(b) 字段、(c) 解耦、(d) 耦合四项自检 |
| `analyze.py` / `stats.py` | 回合级数据表、各单元汇总、C 判定、A0 判定、RL 放大效应、信息/动力学分解 |
| `tasks/` | 各阶段的初始状态子集（`make_tasks.py` 可重新生成） |
| `lit_audit/` | 文献审计的预注册与编码表（读论文之前填写） |
| `tests/` | 无 GPU 测试（见第 9 节） |

条件的命名：

| 名称 | `--oracle-info` | `--oracle-dynamics` | 含义 |
|---|---|---|---|
| V | visible | coupled | 原始 harness |
| M-all | hidden | free | 完全无 oracle |
| M-info | hidden | coupled | 信息藏了，但成功时仍自动停手、成功后动作仍被拒 |
| M-dyn | visible | free | 看得到成功，但环境不再在成功时停 |
| M-field / M-prompt | fields / prompt | coupled | 只删字段 / 只改写文本 |

`hidden` 屏蔽的内容：
- 所有工具返回值和状态视图中的 `terminated` / `truncated`；
- `pi0_doubled` 的成功标志（它就是 LIBERO 官方谓词本身）；`pi0_pick` 只报告它自己的抬升和夹爪判据；
- 成功后动作被拒时的报错原文（改为不含成功字样的通用信息）；
- 输出目录里的 `states.json`、日志、transcript、proxy 录制（planner 的文件工具读不到）；
- system prompt、工具说明、guides 中所有提到 `terminated` 的文本。所有改写都按原文精确匹配，改写后会检查是否还有残留，有残留就报错。

`free` 的含义：环境在成功后照常运行，成功只写进实验者的日志；`finish` 时会额外查询一次 LIBERO 成功谓词。

## 2. 服务器准备（只需做一次）

```bash
# 2.1 本仓库
git clone https://github.com/tongbsh/AgenticRoboRL.git && cd AgenticRoboRL
git checkout claude/practical-hamilton-9atn7u
export AGENTICROBORL_DIR=$PWD

# 2.2 RPent：f70d720 + 论文 patch + 本 patch（脚本会校验 tree hash）
research/c/setup_rpent.sh /data/RPent-c

# 2.3 环境（三个，可以和论文复现共用）
#   RPENT_ENV：Python 3.11
conda create -y -p /data/envs/rpent python=3.11 && /data/envs/rpent/bin/pip install -e "/data/RPent-c[full]" pytest
/data/envs/rpent/bin/liberopro-download-assets --skip-existing
#   PLANNER_ENV：vLLM 0.28（同时用于 planner、VLM 判官和控制脚本）
conda create -y -p /data/envs/vllm python=3.11 && /data/envs/vllm/bin/pip install vllm==0.28.0 pyyaml fastapi uvicorn httpx numpy

# 2.4 权重
hf download ciciciciliu/AgenticRoboRL-Qwen3.5-9B --include "sft/*" "rl-v006/*" --local-dir /data/ckpt/AgenticRoboRL-Qwen3.5-9B
hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT --exclude optimizer.pt --local-dir /data/ckpt/RLinf-Pi05-LIBERO-130-fullshot-SFT
modelscope download facebook/sam3 --local-dir /data/ckpt/sam3
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/ckpt/Qwen3-VL-8B-Instruct
# 只有 INCLUDE_BASE=1 时才需要：hf download Qwen/Qwen3.5-9B --local-dir /data/ckpt/Qwen3.5-9B

# 2.5 配置
cp research/c/c.env.example research/c/c.env   # 按上面的路径填写
```

## 3. 第 1 周：静态检查 + 单回合冒烟

```bash
research/c/preflight.sh
```

`preflight.sh` 检查以下内容，全部为 ok 才继续：
- 路径和权重是否存在；
- RPent 的 tree hash 是否为 `10a16d6…`；
- `RPENT_ENV` 实际导入的是不是这份 RPent（必须是 `pip install -e` 安装）；
- 补丁的单元测试：用假的 LIBERO 后端跑真实的 toolkit，覆盖全部条件；
- vLLM 版本、控制脚本的依赖、harness 配置能否渲染、各 GPU 是否可见、端口是否空闲。

单回合手动冒烟（可选，用来确认三个服务都能起来、看一眼回合目录）：

```bash
source research/c/c.env
$CTL_PYTHON rl/local_ctl.py eval --config harnesses/rpent_libero_c/harness.yaml --run-dir $RUNS_DIR/try \
  --weights $SFT_WEIGHTS --tag try --tasks research/c/tasks/smoke.jsonl --init-key val_init_ids --shard 0/4 \
  --var gpu=0 --var model_port=18100 --var proxy_port=18200 --var vla_port=18300 --var sam3_port=18400 \
  --var oracle_info=hidden --var oracle_dynamics=free --var vllm_mem=$VLLM_MEM \
  --var max_model_len=$MAX_MODEL_LEN --var max_turns=$MAX_TURNS
```

这一周要实测并记下来的数据：
- 单回合耗时（`$RUNS_DIR/try/eval/try/eps/.../transcript_*.json` 中的 `elapsed_s`）；
- 每卡显存（`nvidia-smi`）；
- vLLM 是否能以 `MAX_MODEL_LEN=262144` 启动。

如果显存不够或 vLLM 起不来，先调整 `c.env` 里的 `VLLM_MEM`（vLLM 占整卡的比例）和 `harness.yaml` 里的 `episodes_parallel`（每卡并发回合数），最后才降 `MAX_MODEL_LEN`，并在结果里如实报告改动。

## 4. 第 2 周：泄漏自检（必须全部通过才能跑 C0）

```bash
research/c/smoke.sh      # 只用第一张 GPU，16 个回合，约 2–3 小时
```

`leak_check` 必须对 16 个回合全部报告 0 failure。重点看两项：
- `hidden/free` 和 `visible/free` 下有回合在成功之后还继续动作，并且没有被拒绝。这就是计划里的头号风险"成功后能否继续运行"，在真实 LIBERO 上第一次得到验证。
- `hidden/*` 下，planner 看到的工具结果、prompt、工具说明里没有任何 oracle 字样。

coupled 条件下的 (d) 警告不算失败：π0.5 的一个动作块（5 步）在一次 RPC 里执行完，成功如果发生在块的中间，块内剩下的动作照样执行，可能把成功撤销，`finish` 时 `final_success` 为 false。原版 RPent 就是这样，V 条件就是论文协议，所以照实记录。只有成功之后又走了 5 步以上（进入了下一个动作块），才说明 coupled 没起作用，记为失败。

如果 (c) 失败（free 条件下成功后的动作被拒绝），说明这个 LIBERO 环境版本在成功后不能继续运行。此时 C 只能报告信息通道效应（V vs M-info），按计划 1.5 节降低因果表述。

## 5. 第 3 周：C0 试点与决策

```bash
research/c/run_stage.sh c0      # {sft, rl} × {V, M-all} × 80 任务 × 验证 init 45–46 = 640 回合
```

`run_stage.sh` 依次执行：4 个单元 → VLM 判官 → `leak_check` → `analyze`。结果在 `$RUNS_DIR/analysis/c0/report.md`，其中两行决定下一步：

| `C decision` | 下一步 |
|---|---|
| `C_HOLDS` | C1 前半 `research/c/run_stage.sh c1a`，然后 C2，最后 C1 后半（见第 6 节） |
| `EXPAND`（中间档） | `research/c/run_stage.sh c0_ext`，在合并后的 `analysis/c0+c0_ext/report.md` 上按同一规则再判 |
| `C_WEAK` | C 缩为 A 论文中的一节；照常发布无泄漏协议 |

| `A0 decision` | 下一步 |
|---|---|
| `A_GATE` / `A_GATE_LOW_ROOM` | 进入 A 的在线小 gate（计划 2.6 节；代码在 A 阶段写） |
| `A_WEAK_SIGNAL` | 先在 M-all 下加 verify 工具或改善 `finish` 前的观测，补测 160 回合 |
| `A_TOO_EASY` | 换 swap 套件或 LIBERO-Plus 等更难的设定重测 |

判定规则与计划 2.5 节一致，`analyze.py` 的文件头有完整定义。按 12 路并发、单回合 20 分钟估算，C0 约 18 小时。

## 6. 后续阶段

```bash
research/c/run_stage.sh c0_ext   # 仅中间档：验证 init 47–49，960 回合
research/c/run_stage.sh c1a      # C1 前半：{sft, rl} × {V, M-all}，评测 init 1–5，1600 回合 → analysis/c1a/
research/c/smoke.sh              # C2 之前：确认 smoke 的 4 个单元都已跑完并通过（续跑，已完成的单元会跳过）
research/c/run_stage.sh c2       # 2×2 的另外两格：M-info、M-dyn，评测 init 1–5，1600 回合 → analysis/c1a+c2/
                                 #   C2_SPLIT=1 时加入 M-field、M-prompt
research/c/run_stage.sh c1b      # C1 后半：评测 init 6–10，1600 回合 → analysis/c1a+c1b/（C1 的正式结果）
```

C1 拆成两半只改变执行顺序，不改分析：C2 的 2×2 分解只用评测 init 1–5，c1a 跑完就能和 C2 配对，分解结果因此提前约一个 C1 半程拿到。规则：
- C1 的结论（复现论文数字、C 的确认、RL 放大）只在 `analysis/c1a+c1b/` 上报告；`analysis/c1a/` 是中期结果。
- c1b 不论 c1a 和 C2 的结果如何都要跑（事先承诺，不按中期结果决定是否继续）。
- 卡多、时间够时也可以直接 `run_stage.sh c1`（评测 init 1–10 一次跑完，C2 的分析会自动合并 c1）。两种方式不要混用。
- `INCLUDE_BASE=1` 时 C1 加入 base 模型。

C1 的 V 条件就是论文的评测协议（800 回合 × 2 个 planner）。它的 `Y_sticky` 应接近论文的 65.1 / 71.0；偏差明显时，先排查环境，再看其他结果。

planner 的墙钟上限与论文相同（`--planner-timeout-s 7200`），我们的卡比论文慢，所以超时更多，尤其是 RL planner。`report.md` 的 timeouts 一列是每个单元的超时比例：同一对比的两个单元应该接近。服务器和别人共用时，超时会随别人的负载变化，请在 `SERVER_LOG.md` 里记录运行期间各卡上有没有别人的进程。

## 7. 运行管理

- **断点续跑**：重新执行同一条命令即可。已完成的回合（含 `episode_done.json`）会跳过；只跑了一半的回合目录会改名为 `*.stale_*` 后重跑。
- **暂停**：`touch $RUNS_DIR/<stage>/<cell>/g*/STOP`，当前回合结束后停止。
- **日志**：
  - 控制器日志：`$RUNS_DIR/<stage>/<cell>/g<gpu>.log`；
  - 服务日志：`.../g<gpu>/eval/<stage>/logs/{model,proxy,service_pi05,service_sam3}.log`；
  - 回合日志：`.../eps/local/<suite>/task_XX/seed_XXX/attempt_00/episode.log`。
- **单个回合目录里的文件**：
  - `transcript_*.json`：RPent 的记录，`oracle` 字段见下表；
  - `rl_turns/`：每次 planner 调用的录制，包括 prompt、输出 token、logprob，供 A 使用；
  - `self_probe.json`：p_probe；
  - `vlm_judge.json`：f_vlm；
  - `states.json` 与 `<artifact>/NN.png`：每一步的状态和图像。
- **磁盘**：每个回合约几十 MB（图像加录制），C 全部约 1 万回合，请预留约 0.5–1 TB。

`oracle` 字段：

| 字段 | 含义 |
|---|---|
| `info` / `dynamics` | 本回合的条件 |
| `sticky_success` | 某一步成功过（论文口径，与 `solved` 相同） |
| `final_success` / `final_success_source` | `finish` 时（没有 `finish` 则为回合结束时）成功谓词的值；来源通常是 `check_success` |
| `finish_called` / `finish_status` / `finish_step` | planner 是否调用了 `finish`、给出的 status、当时的步号 |
| `first_success_env_step` / `env_steps` | 第一次成功时的 env 步数 / 总步数 |
| `log` | 每次有状态的工具调用之后的 `success_now` / `success_ever` |
| `scrub_residual` | 脱敏后仍有 oracle 字样的工具结果数，必须为 0 |

## 8. 需要在服务器上第一次确认的事

1. **free 条件下 LIBERO 能否在成功后继续运行**。依据：RLinf 的 `LiberoEnv` 在 `ignore_terminations: False`、`auto_reset: False` 时只转发每一步的成功谓词；LIBERO 的 `done` 是即时成功谓词；robosuite 只在步数达到 horizon（10000）时才拒绝 step；唯一的拦截是 RPent 客户端的断言，补丁已经绕开。由第 4 节的 smoke 验证。
2. **VLM 判官返回的 logprobs**。`vlm_judge.json` 的 `p_vlm` 应该是 0–1 之间的连续值；如果全是 0/1，说明没拿到 logprobs，此时退化为按答案文本取值，需要在报告中说明。
3. **自评探针的解析率**。`report.md` 的 "probe cov." 一列应接近 1。如果偏低，看 `self_probe.json` 的 `raw` 字段，判断是不是模型仍在输出思考过程。

## 9. 不需要 GPU 的测试

```bash
# 统计、分析、泄漏检查、探针、判官（在 CTL_PYTHON 或任何装有 numpy/pytest 的环境中）
python -m pytest -q research/c/tests
# RPent 补丁（在 RPENT_ENV 中；preflight.sh 也会运行它）
cd $RPENT_DIR && RPENT_DIR=$RPENT_DIR $RPENT_ENV/bin/python -m pytest -q $AGENTICROBORL_DIR/research/c/tests/test_rpent_oracle.py
# 仓库原有测试 + 代理 passthrough 测试
python -m pytest -q tests
```

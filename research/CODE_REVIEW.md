# AgenticRoboRL 基线代码审查（2026-10-26）

审查对象：本仓库 `373a103`（initial release）至 `0f8a794`，配合论文 [arXiv:2610.28589v2](https://arxiv.org/abs/2610.28589) 与 RPent `f70d720` 源码交叉核对。

## 结论

- **RL 核心管线（proxy → build_round → train_rl）逻辑完整、自洽，没有缺失的步骤。** 逐项核对见下表。
- **测试：235 个里 234 个通过。** 唯一失败的 `test_vanished_worker_episode_runs_again_under_a_new_lease` 能稳定复现，根因是测试里连续两次运行共用同一组端口时的竞态（见问题 2），不是 RL 逻辑错误。
- **对后续研究影响最大的不是代码 bug，而是一个设定问题：RPent/LIBERO 中仿真器的成功信号 `terminated` 在回合内对 planner 可见**（问题 1）。如果要做"少标签/无 oracle"方向，必须先处理这一点。
- 若干复现所需的组件没有发布（SFT 数据、DeepSpeed 版 SFT 训练脚本、LIBERO-Plus 评测配置等），见"未发布部分"。

## 核心逻辑逐项核对

| 环节 | 文件 | 核对结果 |
|---|---|---|
| 行为策略 | `rl/rl_sampling.py`, `rl/proxy.py:154-157` | proxy 用纯 T=1 采样覆盖 harness 的采样参数，并去掉 `top_logprobs`，因此 vLLM 返回的 logprob 就是行为策略概率。保留客户端 `seed`。正确。 |
| 记录 | `rl/proxy.py:105-118` | 每次请求记录 prompt/output token ids 和逐 token logprob，图片按 sha1 去重，原子写入。返回给 harness 的响应去掉了附加字段。正确。 |
| 回合判定 | `rl/build_round.py:45-63` | 基础设施失败记为 invalid；记录数与 harness 报告的请求数不一致记为 invalid；首尾两条记录的 policy_version 必须等于本轮版本。正确（只查首尾两条是有意为之的折中）。 |
| 分组与优势 | `rl/build_round.py:126-142` | 超过一半无效的组丢弃；有效回合不足 2 个的组丢弃；`A_i = r_i − mean_{j≠i} r_j`，不除以标准差（Dr.GRPO/RLOO）。A=0 的回合不训练。正确。 |
| turn 采样 | `rl/build_round.py:148-177` | 每个回合无放回抽至多 16 个 turn，RNG 种子由 (round seed, episode key) 决定，可重启复现；排除规则（过长 turn、成功后 turn）用未抽中的 turn 替补，替补数不会越界（已推导 `k−|keep| ≤ |spare|`）。权重 1/k，使每个回合在 loss 中等权。正确。 |
| PPO 损失 | `rl/rl_loss.py` | dual-clip（A<0 时上限 `−A·c`）+ 截断 IS 权重 `min(π_old/π_vllm, 2)`（detach）。与论文公式一致。 |
| π_old 预计算 | `rl/train_rl.py:320-358` | 轮初权重下对所有样本做一次无梯度前向，存为 PPO 的 clip 参照；同时统计 gate-B（训练端与 vLLM 端 logprob 偏差）。mid-checkpoint 恢复时从文件读回同一张表，clip 参照始终是轮初权重。正确。 |
| gate A | `rl/train_rl.py:132` | 每个样本用 HF processor 重渲染 prompt，与 vLLM 的 `prompt_token_ids` 逐 token 比对，不一致则权重置 0（不训练）并计数。正确。 |
| 梯度归一化 | `rl/train_rl.py:446-464` | 每步分母 D = 全局样本权重和（all-reduce）。FSDP2/HSDP 的 reduce-scatter + all-reduce 对 world 取平均，所以 loss 乘 `world`，最终梯度 = Σ loss / D。已推导，正确。 |
| 填充与尾批 | `rl/plan.py`, `rl/train_rl.py:268-276` | 每个 rank 每步前向次数相同（缺的用权重 0 的重复样本填），FSDP collective 不会错步；`--tail_batch merge` 把不足一个 batch 的尾巴并入最后一步。单测覆盖。正确。 |
| 跨轮续训 | `rl/local_ctl.py:311-319` | round>0 时 `--resume` 读上一轮 dcp（权重 + Adam 状态）；round 0 从 SFT 权重新建优化器。正确。 |
| 训练池 | `rl/local_ctl.py:257-285` | 用验证 init 上成功率 ∈ [0.1, 0.9] 的任务组成训练池，组在整个 run 内不重复。正确。 |
| RPent patch | `patches/rpent/rpent_f70d720.patch` | 能干净 apply 到 `f70d720`，sha256 与 README 一致。patch 新增的 `solved = False` 只是异常路径的默认值，原代码 `rpent/cli/main.py:477` 已有 `solved = toolkit.solved()`。正确。 |

论文自己承认的一处近似：一个优化步内不一定包含完整回合，所以 loss 不是严格的"回合平均"目标（论文 §RL 方法）。

## 发现的问题（按对后续研究的影响排序）

### 1. 仿真器成功信号在回合内对 planner 可见（设定问题，影响"少标签/真机"方向）

- RPent 的 LIBERO 系统 prompt 写明目标是 "produce top-level `terminated == true`"（RPent `robots/libero/prompts/system.py:181`），primitive 的返回值里带 `terminated` / `libero_terminated`（`robots/libero/tools.py:124,202`）。
- 也就是说，planner 在回合中途就能读到 LIBERO 的官方成功判定，再决定是否 `finish`。本仓库的 `--drop-post-success`（`rl/adapters/rpent_libero.py:447-479`）也正是靠在 tool 结果里搜索 `"terminated": true` 实现的。
- 报告的成功率按 `toolkit.solved()` 计算，只要回合中任意时刻 `terminated` 为真即算成功（`robots/libero/toolkit.py:155`，sticky）。
- **影响**：在这个设定下，planner 的"自报成功"几乎等价于 oracle，所以如果直接在现有 harness 上做"少标签 RL"，代理奖励其实就是真值，问题会变得平凡。真机没有这个信号。做 Idea A 必须加一个"无 oracle"变体：从 tool 返回值和 prompt 中去掉 `terminated`，并另外记录 planner 调 `finish` 时刻的真实状态（final-state success）。详见 `research/IDEAS.md` 方向一。

### 2. 多机测试失败：端口竞态 + `wait_ready` 会被旧服务"骗过"（低严重度）

- 复现：`python -m pytest tests/test_multi_node_ctl.py::test_vanished_worker_episode_runs_again_under_a_new_lease`，两次都在 300 s 超时。
- 过程：测试先跑完一次 `train`，紧接着用**同一组端口**启动第二个 controller。前一轮 worker 在 controller 返回后还在停止自己的 model server；新 worker 启动 fake model server 时报 `OSError: [Errno 98] Address already in use` 并退出。但 `local_ctl.wait_ready`（`rl/local_ctl.py:112-127`）先检查进程存活、再访问 ready URL，此时旧 server 还在应答，于是判定就绪。第一个回合结束时 `stack.check()` 发现 model 进程已死，worker 退出。controller 对启动不满 300 s 的 worker 不做重启（`rl/multi_node_ctl.py:369-371`），超过了测试本身的 300 s 超时。
- 生产中两轮之间隔着 build 和训练，旧进程早已退出，所以通常不触发。但如果在同一组端口上**背靠背**跑两个阶段（比如脚本里 eval 结束马上 train），短时间内有回合会打到仍在退出中的旧 vLLM（旧权重），而 proxy 给这些记录贴的是新的 policy_version，`build_round` 的版本检查拦不住。
- 修复建议：`wait_ready` 就绪后再确认一次进程存活；model server 额外校验 `/v1/models` 返回的模型 id 等于本轮 `{weights}`（两个 harness.yaml 都用了 `--served-model-name {weights}`，可以直接比对）。

### 3. 训练与评测的采样设置不同（有意为之，记录在案）

训练 rollout 用纯 T=1 采样，RPent 评测沿用 Harness VLA 协议（`top_p 0.95, top_k 20, presence_penalty 1.5, seed 3407`，`eval.direct: true` 不经 proxy）。这是论文的选择，但我们做新方法时最好两种采样下的结果都报告，避免增益来自采样设置的差异。

### 4. 生成种子在一个回合内对每个请求相同（轻微）

`--generation-seed $(((round + 1) * 100000 + attempt))` 对该回合的每次请求都传同一个 seed；不同组里同一个 attempt 序号也共用种子。prompt 不同所以输出不同，对 LOO 估计影响很小，但严格说不同组之间的随机数是相关的。做方差相关的论文（比如方向一）时建议把 task/init 也编码进种子。

## 未发布 / 无法从本仓库复现的部分

| 缺失项 | 证据 | 对我们的影响 |
|---|---|---|
| SFT 用的教师轨迹 | README "The teacher trajectories used for SFT are not released" | 不能重训 SFT；直接用 HF 上的 `sft/`（65.1%）和 `rl-v006/`（71.0%）作起点 |
| SFT 前若干步用的 DeepSpeed ZeRO-3 训练脚本 `train.py` | `sft/rpent/train_fsdp.py:6,13`（`--init_universal` 从 DeepSpeed universal checkpoint 接续） | 同上，只影响逐位复现 SFT |
| LIBERO-Plus 评测配置 | README 报告 LIBERO-Plus 72.2%（10,030 个配置），但 `harnesses/` 只有 LIBERO-PRO 的 80 个任务 | 要测 LIBERO-Plus 需要自己写 tasks.jsonl 和 `--libero-type` 配置 |
| CaP-X 的 SFT/RL 权重 | README 只发布了 RPent 的两个 checkpoint | CaP-X 只能从 base（2.7%）开始，太低，不适合做第二个 harness 的主实验 |
| 论文 RL rollout 实际用的 RPent 版本（额外记录仿真器状态） | `patches/rpent/README.md` 最后一段 | 该记录对 planner 不可见，不影响结论 |
| `audit_memory.py` | `sft/rpent/replay_build.py:49` 注释引用 | 无影响 |

## 适配 4×72GB 需要改的地方

- 两个 `harness.yaml` 的 `train.cmd` 写死了 `--nproc_per_node 8 --gpus_per_node 8`，要改成 4。`global_batch 128` 能被 4 整除，可以保留。
- `--ce_chunk`：RPent 配置用 0（一次算全部 logits）。输出长达 20k token 时，logits 本身就是 20k × 248k × 4 B ≈ 20 GB，再加梯度，72 GB 卡放不下，应设 512。
- `--max-total-tokens 160000`：9B 模型 fp32 主权重 + 梯度 + Adam 共约 144 GB，4 卡分片后每卡约 36 GB；剩下约 36 GB 给激活。160k token 的样本即使开了 gradient checkpointing 也很可能 OOM，建议先实测，大概率要降到 64k–96k。注意：丢弃长 turn 会系统性地偏向短回合和早期 turn，要在论文里报告被丢弃的比例。
- rollout 时每张卡同时跑 vLLM（9B，`--gpu-memory-utilization 0.62` ≈ 45 GB）、π0.5 和 SAM3，72 GB 卡需要实测是否放得下，必要时降低 `--max-num-seqs` / `episodes_parallel`。

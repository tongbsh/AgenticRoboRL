# 在服务器上用 Claude Code 执行方案 C：prompt 集

用法：
1. 服务器上拉取代码（不需要合并到 main）：
   ```bash
   git clone https://github.com/tongbsh/AgenticRoboRL.git && cd AgenticRoboRL   # 已有仓库则：git fetch origin
   git checkout claude/practical-hamilton-9atn7u && git pull
   ```
2. 在仓库根目录启动 Claude Code，最好在 tmux 里启动，断线也不影响。
3. 按顺序喂下面的 prompt，一周一段。每段做完，把 Claude Code 给出的总结（以及它写的 `research/c/SERVER_LOG.md`）发回给我。
4. `<尖括号>` 里的内容先替换成你们服务器的实际值。

长任务（单回合实测、smoke、各阶段）都要求 Claude Code 放进 tmux 后台运行。之后随时可以开一个新对话，用 Prompt 5 查看进度。

---

## Prompt 1：第 1 周，环境、权重、配置、静态检查、单回合实测

```text
你在一台 Linux GPU 服务器上（4 × NVIDIA RTX PRO 5000 Blackwell 72GB），当前目录是 AgenticRoboRL 仓库的
claude/practical-hamilton-9atn7u 分支。我们要运行"方案 C"（仿真器成功信号泄漏的因果审计）。实验代码和运行手册在
research/c/，研究背景在 research/PLAN_C_then_A.md。

这次对话只做第 1 周的工作：搭环境、下权重、写配置、静态检查、单回合实测。不要开始 smoke 或 C0。

先完整阅读 research/c/README.md、research/c/c.env.example、harnesses/rpent_libero_c/harness.yaml，然后按 README
第 2、3 节执行：

1. 盘点服务器：nvidia-smi（驱动和 CUDA 版本）、磁盘空间（实验输出约需 1 TB）、conda 或 uv 是否可用、能否访问
   huggingface.co（不行就设 HF_ENDPOINT=https://hf-mirror.com）、能否用 modelscope。
2. 数据根目录用 <数据根目录，例如 /data>，在其下建 envs/、ckpt/。
3. 运行 research/c/setup_rpent.sh <数据根目录>/RPent-c。它会校验 tree hash，不一致就停下报告。
4. 建两个 Python 3.11 环境：
   - RPENT_ENV：pip install -e "<数据根目录>/RPent-c[rlinf,openpi,libero-pro,sam3]" pytest，
     然后运行 liberopro-download-assets --skip-existing。
   - PLANNER_ENV：pip install vllm==0.28.0 pyyaml fastapi uvicorn httpx numpy pytest。
   这是 Blackwell 显卡（sm_120）。在两个环境里分别确认 torch.cuda.is_available()、
   torch.cuda.get_device_capability()，以及一次 GPU 矩阵乘法能正常运行。如果不支持，安装 CUDA 12.8 的 torch
   构建，并记录你做了什么。
   确认 MuJoCo 能在 MUJOCO_GL=egl 下离屏渲染：用 RPENT_ENV 的 python 渲染一帧即可。
5. 按 README 2.4 节下载权重：AgenticRoboRL 的 sft/ 和 rl-v006/、RLinf-Pi05-LIBERO-130-fullshot-SFT（不要
   optimizer.pt）、SAM3、Qwen3-VL-8B-Instruct。不需要 Qwen3.5-9B base。
6. 把 research/c/c.env.example 复制为 research/c/c.env，填好所有路径，GPUS="0 1 2 3"。
7. 运行 research/c/preflight.sh，直到全部为 ok。warn 可以保留，但要在日志里说明原因。
8. 单回合实测：用 README 第 3 节的命令，在 GPU 0 上跑 smoke 任务集（hidden/free 条件）。这一步要几十分钟，
   放在 tmux 会话里后台运行，定期查看，不要在前台阻塞等待。跑完后记录：
   - 每个回合的 elapsed_s、turns_used、transcript 里的 oracle 字段；
   - self_probe.json 里是否有 p_probe；
   - 运行期间每卡显存峰值（nvidia-smi）；
   - vLLM 是否以 MAX_MODEL_LEN=262144 正常启动（看 eval/try/logs/model.log）。

规则：
- 不要为了让检查通过而修改 research/c/、rl/、harnesses/rpent_libero_c/、patches/ 下的任何代码或阈值。
- 遇到以下情况先停下，把完整报错和你的分析告诉我，等我确认后再继续：任何检查失败、命令报错、需要改代码，
  或者需要改实验设置（比如降低 MAX_MODEL_LEN、改 episodes_parallel 或 VLLM_MEM）。
- 不要删除任何已下载的数据或运行目录。
- 所有执行过的命令、版本号、实测数字都追加到 research/c/SERVER_LOG.md。

最后给我一份总结：
- 环境版本：驱动、CUDA、两个环境的 torch、vLLM、MuJoCo；
- preflight 结果；
- 单回合实测：耗时、轮数、显存；
- 遇到的问题和你做过的处理。
```

## Prompt 2：第 2 周，泄漏自检（smoke）

```text
继续方案 C 第 2 周：泄漏自检。先读 research/c/README.md 第 4 节和 research/c/SERVER_LOG.md。

1. 确认 research/c/preflight.sh 仍然全部为 ok。
2. 在 tmux 会话 c-smoke 中运行 research/c/smoke.sh，输出用 tee 写到 runs/c/smoke/smoke.out。它只用第一张 GPU，
   跑 16 个回合，约 2–3 小时。不要在前台等待，每 20–30 分钟查看一次进度：runs/c/smoke/*/g*.log 和已完成的回合数。
3. 结束后报告以下内容：
   - leak_check 的汇总（各条件的 episodes / failed / warned）和所有 FAIL 行。
   - "成功后能否继续运行"的验证：在 sft__hidden-free 和 sft__visible-free 两个单元里，列出
     oracle.first_success_env_step < oracle.env_steps 的回合（即成功后还继续动作的回合），并确认这些回合在成功
     之后没有被拒绝的动作。
   - 改写后的文本：从一个 hidden 回合的 rl_turns 中取第一条记录，展示 system prompt 里 YOUR GOAL 那一段，以及
     pi0_doubled 的工具说明，让我确认。
   - runs/c/analysis/smoke/report.md 的 Cells 表。smoke 的数字没有统计意义，只看四个单元是否齐全。另外看 p_probe
     的覆盖率；smoke 不跑判官，f_vlm 为空是正常的。
4. 如果任何检查失败，不要改代码，停下来。把 runs/c/smoke/leak_check.json 中失败回合的路径、对应 transcript 的
   oracle 字段、相关工具结果片段整理给我。

把过程和结果追加到 research/c/SERVER_LOG.md。
```

## Prompt 3：第 3 周，C0 试点与决策

```text
继续方案 C 第 3 周：C0 试点。先读 research/c/README.md 第 5 节、research/PLAN_C_then_A.md 第 2.5 节，以及
research/c/SERVER_LOG.md。前提是 smoke 的 leak_check 已经全部通过；如果 SERVER_LOG 里没有这条记录，先停下问我。

1. 根据 SERVER_LOG 里的单回合实测，重新估算 640 个回合的总耗时，告诉我。
2. 在 tmux 会话 c-c0 中运行 research/c/run_stage.sh c0，输出用 tee 写到 runs/c/c0/stage.out。不要在前台等待。
3. 运行期间如果某个控制器失败退出，先看日志（runs/c/c0/<单元>/g*.log 和 g*/eval/c0/logs/*.log）定位原因：
   - 偶发问题（端口占用、服务启动超时）：重新执行同一条命令续跑，已完成的回合会保留；
   - 代码或环境问题：停下来报告。
4. 全部结束后（run_stage.sh 会自动跑 VLM 判官、leak_check 和分析）：
   - 确认 leak_check 为 0 failure；
   - 把 runs/c/analysis/c0/report.md 全文给我；
   - 用中文总结以下内容：
     - C decision 和 A0 decision 分别是什么；
     - 按 README 第 5 节的表，下一步该做什么；
     - 任何可疑的数字：
       - V 条件的 Y_sticky 与论文 65.1 / 71.0 不在同一量级（C0 用的是验证 init，不要求完全一致）；
       - 某个单元的无效回合过多；
       - 探针或判官覆盖率低，或者判官的 p_vlm 全是 0 或 1（说明没拿到 logprobs）。

不要修改任何代码、阈值或判定规则，结果不论好坏都如实报告。把过程追加到 research/c/SERVER_LOG.md。
```

## Prompt 3b：C0 收尾，开始 C1 前半（c1a）

C0 的 leak_check 有 5 个 (d) 失败，已确认是原版协议的行为而不是泄漏；检查 (d) 已修订。这段 prompt 先在已有数据上重跑检查和分析（不重跑回合），补齐 smoke，再开始 c1a。

```text
继续方案 C：C0 收尾，然后开始 C1 前半（c1a）。先拉取最新代码：git pull origin claude/practical-hamilton-9atn7u，
然后读 research/c/README.md 第 4–6 节和 research/c/SERVER_LOG.md。

背景：C0 的 leak_check 有 5 个 V 回合没通过检查 (d)。原因已经确认：coupled 条件下，env 要等成功所在的那个 5 步
π0.5 动作块执行完才停，块内剩下的动作可能把成功撤销。原版 RPent 就是这样（V 条件就是论文协议），不是泄漏。
新版 leak_check 把 (d) 改为：成功之后又走了 5 步以上才算失败，块内撤销记为警告。analyze.py 的 Cells 表新增了
timeouts 一列（planner 墙钟超时的比例）。

1. 在已有的 C0 数据上重跑检查和分析，不重跑任何回合：
   - source research/c/c.env，然后 $CTL_PYTHON research/c/leak_check.py runs/c/c0 --json runs/c/c0/leak_check.json
     预期：0 failure，原来那 5 个回合变成 (d) 警告。出现任何 failure 就停下报告。
   - 用 run_stage.sh 最后一行的命令重跑分析（--stages c0 --out runs/c/analysis/c0）。确认除了新增的 timeouts 列，
     其他数字与上次完全一致。把新的 Cells 表给我。
2. 补齐 smoke：在 tmux 会话 c-smoke 中运行 research/c/smoke.sh（续跑，已完成的单元会跳过）。确认：
   - 4 个单元都齐全，leak_check 为 0 failure；
   - 重点看 visible-free 和 hidden-coupled 两个单元（C2 要用）。visible-free 里成功之后还继续动作的回合，
     不能有被拒绝的动作。
   smoke 没通过就停下报告，不要开始 c1a。
3. 都通过后，在 tmux 会话 c-c1a 中运行 research/c/run_stage.sh c1a，输出用 tee 写到 runs/c/c1a/stage.out。
   - 先按 C0 的实际速度估算总耗时，告诉我；
   - 控制器失败的处理同 Prompt 3 第 3 步；
   - 运行期间每天记录一次各卡上有没有别人的进程（nvidia-smi），写进 SERVER_LOG。
   如果现在能用的卡比 c.env 里的 GPUS 多，先告诉我，不要自行修改。
4. c1a 结束后（run_stage.sh 会自动跑判官、leak_check 和分析）：确认 leak_check 为 0 failure，把
   runs/c/analysis/c1a/report.md 全文给我。这是中期结果，不据此做任何决策；下一步固定是 C2。

不要修改任何代码、阈值或判定规则。把过程追加到 research/c/SERVER_LOG.md。
```

## Prompt 4：后续阶段（c2 / c1b，或中间档时的 c0_ext）

顺序固定为 c1a → c2 → c1b。c1b 不论前面结果如何都要跑。

```text
继续方案 C：运行阶段 <c2 | c1b | c0_ext>。先拉取最新代码（git pull origin claude/practical-hamilton-9atn7u），
读 research/c/README.md 第 6 节，以及 research/c/SERVER_LOG.md 中上一阶段的记录，确认这个阶段确实是下一步
（顺序是 c1a → c2 → c1b；c0_ext 只在 C0 判为 EXPAND 时跑）。不是的话先停下问我。

按 Prompt 3 的第 1–4 步执行，把其中的 c0 换成本阶段名：
- tmux 会话名用 c-<阶段>；
- 分析结果的路径：c2 在 runs/c/analysis/c1a+c2/，c1b 在 runs/c/analysis/c1a+c1b/，c0_ext 在
  runs/c/analysis/c0+c0_ext/；
- 运行期间每天记录一次各卡上有没有别人的进程。

阶段特有的报告内容：
- c2：报告中 "Information / dynamics decomposition" 一节的数字，以及信息通道和动力学通道哪个效应更大。
- c1b：这是 C1 的正式结果（评测 init 1–10 全部）。
  - V 条件的 Y_sticky 与论文 65.1（sft）/ 71.0（rl）的对比。这是论文评测协议的复现，差距超过 4 个点要单独说明；
  - C decision 和 RL amplification 一节；
  - 各单元的 timeouts，同一对比的两个单元差别大时要指出。

不要修改任何代码、阈值或判定规则。把过程追加到 research/c/SERVER_LOG.md。
```

## Prompt 5：随时查看进度（新开对话时用）

```text
方案 C 的实验正在 tmux 里后台运行。先读 research/c/SERVER_LOG.md 了解当前阶段，然后报告：
- tmux 会话列表；
- 当前阶段每个单元的已完成回合数 / 总回合数，以及基础设施失败数；
- 根据最近完成的回合速度，估计剩余时间；
- g*.log 和服务日志里最近的错误。

只看不改：不要重启任何任务，除非控制器已经退出且原因明确是偶发问题。即便如此，也先告诉我你打算执行的命令。
```

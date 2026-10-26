# 执行计划：先做 C（仿真器成功信号泄漏的因果审计），再做 A（自评 planner）

日期：2026-10-26。硬件：4 × RTX PRO 5000 72GB。起点：开源的 `sft/` 与 `rl-v006/` 两个 planner，加上 base 模型 Qwen3.5-9B。

文中 RPent 的代码位置均指 `f70d720` 加上本仓库 patch 之后的版本。

---

## 0. 总览

| 周 | C | A | 产出 / 决策 |
|---|---|---|---|
| 1–2 | 环境搭建（W1）、无 oracle 补丁（W2） | （同左，共用） | 补丁通过泄漏自检 |
| 3 | **C0 试点**（第 2 节） | **A0 自评探针**（随 C0 一起跑，不加 rollout） | **继续 / 转向决策** |
| 4–5 | C1 主审计 + C2 因子消融 | 实现 A 的代码（`--mode sg`、标注、带置信度的 `finish`），写 T1–T4 的证明 | |
| 6–7 | C 分析与写作 | A-E1 离线方差实验 | C 初稿，可先挂 arXiv 占位 |
| 8–14 | （C 投稿：RSS 2027 约 1–2 月，或 CoRL 2027 / NeurIPS D&B 2027 约 5–6 月） | A-E2 在线实验；空档跑 A-E3（ALFWorld） | |
| 15–18 | | 分析与写作 | A 投 NeurIPS 2027（约 5 月中） |

会议日期都是往年惯例的估计，以官网为准。

两者如何衔接：C 的所有评测回合（无 oracle 条件下的 SFT/RL planner）同时是 A 的试点数据和离线实验数据。C 的核心结论"现有 harness 依赖回合内 oracle"也正是 A 的问题动机。A 只需要在 C 的基础上再加一个 `finish(confidence)` 字段。

---

## 1. 方案 C：仿真器成功信号泄漏的因果审计

暂定题目：*Who Told You It Worked? Simulator-Success Leakage in Agentic Robot Benchmarks*

### 1.1 动机

1. **领域现状**：agentic robotics 的 SOTA 数字（Harness VLA、AgenticRoboRL，以及一批 harness 自演化工作）大多在仿真里取得。planner 要决定"何时宣布完成、是否重试、是否换策略"，而在 RPent/LIBERO 中，这些决定可以直接读仿真器的官方成功判定。真机上不存在这个信号，所以这些数字有多少能迁移到真机，目前没人量化过。
2. **RL 可能放大这种依赖**：AgenticRoboRL 的 RL 是在 oracle 可见的条件下训练的。策略可能学到"一直试，直到看到 `terminated: true`"这种只在仿真里成立的行为。RL 比 SFT 多出的 5.9 个点里，有多少依赖这个信号？
3. **社区需要一个无泄漏协议**：已有工作零星提到这类捷径。[2601.20334](https://arxiv.org/abs/2601.20334) 把仿真器成功信号回灌进上下文，作者自己报告了 agent 读取仿真器内部状态的案例。但还没有人系统审计、做因果量化，并给出协议。
4. **成本低、结果风险低**：泄漏已经在代码层面确证，剩下的只是"测出有多大"。只需要评测，不需要训练。

### 1.2 RPent 中的特权通道清单（已逐条定位）

| 编号 | 通道 | 代码位置 | 类型 |
|---|---|---|---|
| P1 | 每个 primitive 的返回值和状态视图都带 `terminated` / `truncated` | `robots/libero/tools.py`：`pi0_pick`:215、`pi0_doubled`:261、`move_to`:336、`rotate_wrist`:412、`rotate_pitch`:494、`move_pose`:581、`release`:610、`set_gripper`:633；`view_env_state`:1613 | oracle 直接暴露 |
| P2 | prompt 以 `terminated == true` 为目标，并规定"看到 `terminated == true` 就写 audit 然后 `finish`" | `robots/libero/prompts/system.py`：ROLE:28、GOAL:181、Rule 4:277、WORKFLOW:492–500。学生用的 `proven` 配置（`knowledge_eval.py`）原样继承了这些段落 | 指示 planner 依赖 oracle |
| P3 | `pi0_doubled` 的成功判定就是 LIBERO 官方终止谓词（返回 `success` / `task_success`） | `tools.py`:249–260；prompt 中 "success := `terminated`"（`system.py`:456） | oracle 换了个名字暴露 |
| P4 | VLA 和解析运动在 `terminated` 时提前 `break`，因此 `chunks_used`、耗时、轨迹被截断 | `tools.py`:202（pick）、249（doubled）、326（move_to）、400（rotate_wrist） | 侧信道 |
| P5 | 成功后再调用任何动作会触发 `assert not (self.terminated ...)` 报错 | `robots/libero/env_client.py`:65、84 | 侧信道 |
| P6 | `reset_episode` 返回 `libero_terminated` | `tools.py`:124（prompt 禁止 reset，影响很小） | oracle 直接暴露 |
| R1 | 状态中的 `object_names` 来自仿真器的观测键名 | `tools.py`:908–916 | 特权但较弱（任务语言里通常也有） |
| R2 | 来自仿真器的完美深度、相机标定和世界坐标图（`agentview_world.npz` 等） | `_save_observation_artifacts` | 真机可实现（深度相机 + 标定），只是没有噪声 |
| R3 | pick 的启发式成功判据（下降、抬升、夹爪闭合） | `tools.py`:180–200 | 真机可实现，**不算泄漏** |

扩展（可选）：CaP-X 配置使用真值物体状态的"privileged" API（`harnesses/capx_libero/harness.yaml` 头注释）。

C 的主审计只干预 oracle 类通道 P1–P6。R1 和 R2 作为"可实现但理想化"的通道放在讨论部分，或者单独做一个加噪的消融。

### 1.3 因果框架与估计量

- **干预**：do(屏蔽通道集 S)，即在 harness 层面移除或改写 S 中的信息。其余一切（模型、采样参数、初始状态、工具集）保持不变。
- **结局 Y**（每个回合测量）：
  - `Y_sticky`：回合中任意时刻成功（论文口径）；
  - `Y_final`：planner 调用 `finish` 时，LIBERO 成功谓词是否成立（部署口径，主指标）；
  - `FC`：虚报成功，即 `status=success` 且 `Y_final = 0`；
  - `MS`：错过成功，即 `Y_sticky = 1` 但 `status ≠ success`，或者成功后又把状态破坏了（`Y_final = 0`）；
  - `NF`：没有调用 `finish`（耗尽 100 轮或超时）；
  - 轮数、成功之后的动作数、各套件分解。
- **依赖度（主估计量）**：ACE_S(Y; π) = E[Y | do(全可见), π] − E[Y | do(屏蔽 S), π]，π ∈ {base, SFT, RL}。
- **RL 放大效应**：Δ_RL = ACE_S(Y; RL) − ACE_S(Y; SFT)，即双重差分。大于 0 说明 RL 增加了对 oracle 的依赖。
- **oracle 缺口**：G = E[Y_sticky − Y_final | do(屏蔽)]。衡量"做成了但自己不知道，或者又搞砸了"的比例。
- **"信息 vs 指令"分解**：比较只屏蔽字段（P1）、只改写 prompt（P2）、全部屏蔽（P1–P6）三种条件。区分 planner 依赖的是 oracle 这个信息本身，还是 prompt 里"以 oracle 为目标"的指令。

### 1.4 实验条件

| 条件 | 内容 | 用途 |
|---|---|---|
| **V** | 原始 harness（P1–P6 全可见） | 对照；同时复现论文数字 |
| **M-all** | 屏蔽 P1–P6：删除 `terminated` / `truncated` / `libero_terminated` 字段；改写 P2 的 prompt（目标改为"按 task_language 完成任务，自行目视确认后写 audit 并 `finish(status)`"）；`pi0_doubled` 改为只返回执行信息、不返回成功判定；去掉提前 `break`（各 primitive 只按自身的停止条件结束）；成功后允许继续运行 | 完全无 oracle，主条件 |
| M-field | 只删字段（P1、P3、P6），prompt 不改 | 信息 vs 指令 |
| M-prompt | 只改写 prompt，字段保留 | 信息 vs 指令 |
| M-all + verifier（可选） | M-all，再给 planner 一个可在真机实现的 `verify` 工具（开源 VLM 看当前图像回答"任务是否完成"） | 展示"可实现的替代方案"能挽回多少；与 A 衔接 |

实现要点：
- 所有开关做成 RPent 的一个 CLI 参数（如 `--oracle-channels all|none|fields|prompt`）。
- prompt 改写沿用 `knowledge_eval.py` 的 `_replace_once` 风格（片段不存在就报错），保证可审计。
- **成功后继续运行**：RPent 用的是 RLinf 的 LIBERO 环境（`robots/libero/env_server.py:74` 中 `ignore_terminations: False`、`auto_reset: False`）。屏蔽条件下需要验证：设成 `ignore_terminations: True` 能否让环境在成功后继续 step，同时仍能从 info 或成功谓词拿到成功标志供内部记录（`Y_sticky`、`Y_final`）。如果不行，备选方案是在 env server 端绕过终止处理，这需要读 RLinf 环境的代码。这是 W2 最大的不确定点，第 1 周就要验证。
- 记录 `Y_final`：在 `finish` 被调用时（`rpent/planner/api_loop.py:432` 附近检测到 `finish`）立即查询成功谓词，写进 transcript；同时保存这一刻的 agent-view 和 wrist 图像，供 VLM 判官和 A 使用。
- 采样设置：C 的所有评测都用论文的评测协议（`eval.direct`，top-p 0.95 / top-k 20 / presence 1.5 / seed 3407），保证与 65.1% / 71.0% 可比。

### 1.5 实验阶段

| 阶段 | 内容 | 回合数 | GPU 时间（单回合按 20 分钟、16 路并发，即每小时 48 回合） |
|---|---|---|---|
| **C0 试点** | {SFT, RL} × {V, M-all}，验证 init 45–49 中取 2 个 × 80 任务 | 4 × 160 = 640 | ≈ 13 h |
| C0 扩展（结论不清时） | 补全 5 个验证 init | +960 | ≈ 20 h |
| **C1 主审计** | {base, SFT, RL} × {V, M-all}，评测 init 1–10 × 80 任务（论文协议） | 6 × 800 = 4800 | ≈ 100 h |
| C2 因子消融 | {SFT, RL} × {M-field, M-prompt}，评测 init 1–5 | 4 × 400 = 1600 | ≈ 33 h |
| C3 扰动（可选） | LIBERO-Plus 抽 500 个配置 × {SFT, RL} × {V, M-all} | 2000 | ≈ 42 h |
| C4 可实现的替代（可选） | {SFT, RL} × M-all + verifier，评测 init 1–5 | 800 | ≈ 17 h |
| **合计** | | ≈ 1 万回合 | **约 7–10 个 GPU 天** |

另外还有一个零算力的文献审计：逐篇检查已发表 harness 工作的评测协议是否暴露仿真器成功信号或真值状态（Harness VLA、AgenticRoboRL、CaP-X、2601.20334、World Action Agent、SERA、BATON 等），做成一张表。这能把 C 从"一个 harness 的发现"提升为"领域级的问题"。

### 1.6 假设（C1/C2 开始前预注册）

- **HC1**：SFT 和 RL 两个 planner 在 M-all 下的 `Y_final` 都显著低于 V（ACE > 0）。
- **HC2（RL 放大）**：Δ_RL > 0，即 RL 比 SFT 更依赖 oracle。
- **HC3**：M-all 下 oracle 缺口 G 显著大于 0，主要来源是虚报成功和成功后又破坏状态。
- **HC4**：M-field 的效应接近 M-all，说明依赖主要来自信息本身而非 prompt 指令。这条是探索性的，不预设方向。
- **HC5（重估基线）**：在无泄漏协议下，RL 相对 SFT 的增益会改变。缩小说明增益部分来自对 oracle 的利用，扩大说明 RL 学到了更强的自主判断。两种结果都值得报告。

统计方法：
- 回合按 (task, init) 配对；
- ACE 用配对差和以 task 为聚类单位的 bootstrap 给 95% CI；
- Δ_RL 用 logistic 回归 `Y ~ planner × condition`，标准误按 task 聚类；
- 二值主比较另报 McNemar 检验；
- HC1–HC3 用 Holm 校正。

### 1.7 论文结构（C）

1. 问题：agentic robotics 基准里的仿真器成功泄漏（通道清单 + 文献审计表）；
2. 因果审计方法（干预、估计量）；
3. 结果：依赖度、RL 放大、oracle 缺口、信息 vs 指令；
4. 行为分析：虚报成功、错过成功、成功后破坏状态的典型案例；
5. 无泄漏协议与重估后的基线；可实现的替代方案（verifier）能挽回多少；
6. 讨论：对仿真到真机迁移的含义，以及 R1/R2 这类"可实现但理想化"的通道。

---

## 2. 第一步：验证实验（C0 + A0），约 3 周，决定后面所有事情

### 2.1 目标：用最少的 GPU 时间回答四个问题

| 问题 | 用于 | 怎么测 |
|---|---|---|
| Q1 环境能否在 4 卡上跑通？实际单回合耗时、显存、吞吐是多少？ | 所有算力估计 | 冒烟测试 + C0 自带的计时 |
| Q2 屏蔽 oracle 后，`Y_final` 掉多少？虚报成功率是多少？RL 是否掉得比 SFT 多？ | C 是否成立 | C0 的 2×2 设计 |
| Q3 planner 的自评能不能区分成功和失败？校准得如何？ | A 是否成立 | 回合结束后的自评探针 + 自报 status + VLM 判官 |
| Q4 用这些自评做控制变量，标签噪声带来的方差能降多少？ | A 的标签效率主张 | 在 C0 数据上离线计算 |

### 2.2 第 1–2 周：工程（W1 + W2）

**第 1 周（W1，环境）**

1. 按 README 准备 `RPENT_ENV`、`PLANNER_ENV`（vLLM 0.28）和 `TRAIN_ENV`，下载 `sft/`、`rl-v006/`、`PI05_CHECKPOINT`、`SAM3_CHECKPOINT`。
2. 把 `harnesses/rpent_libero/harness.yaml` 拷贝为 4 卡版本：`--var gpu=0..3`，每卡一套端口，`episodes_parallel` 先设 3（72GB 上同时放 vLLM、π0.5、SAM3，需要实测）。
3. 冒烟测试：SFT 权重在 V 条件下跑 8 个任务 × 1 个验证 init，确认 transcript、`solved`、成功率与预期量级一致。
4. **同一周验证"成功后继续运行"的可行性**（见 1.4 实现要点）。这决定 M-all 能否按设计实现，是整个计划的头号技术风险。
5. 记录：单回合耗时分布、每卡显存峰值、vLLM 解码吞吐。可选：试 FP8 权重服务，看采样端和训练端的 logprob 偏差。这个只影响 A 的训练 rollout，C 的评测始终用 bf16。

**第 2 周（W2，无 oracle 补丁）**

1. 在 RPent 上实现 `--oracle-channels {none, fields, prompt, all}`，覆盖 P1–P6，并在 `finish` 时记录 `Y_final` 和最终图像。
2. 新 patch 放进 `patches/rpent/`，不改动原 patch，以便原样复现论文。
3. **泄漏自检**（必须全部通过才能开跑）：
   - (a) 用 M-all 跑 5 个回合，在 transcript 和 proxy 记录的全部 prompt / tool 内容里搜索 `terminat`、`task_success`、`solved`，不允许出现；
   - (b) 对比 V 和 M-all 下同一个 primitive 在成功前后的返回结构，确认字段集合不随成功与否变化；
   - (c) 刻意构造一个"成功后继续调用动作"的回合，确认不再报错；
   - (d) 在 V 条件下核对：planner 在成功后立刻 `finish` 的回合里，`Y_final` 应与 `Y_sticky` 一致，用来检验记录逻辑。
4. 加入 A0 探针所需的钩子（下一节）。

### 2.3 第 3 周：C0 + A0 运行

**设计**：{SFT, RL} × {V, M-all} × 80 任务 × 验证 init {45, 46}，共 640 个回合，约 13 GPU 小时。每个 (planner, task, init) 在两种条件下使用相同的生成 seed（论文评测协议固定 seed 3407），形成配对。

**A0 自评探针（零额外 rollout）**：每个回合结束后，在 planner 的完整上下文末尾追加一条用户消息，例如："仅基于你在本回合看到的信息，给出任务目标此刻已满足的概率（0 到 1 之间的一个数），只输出数字。" 读出 p_probe。这个查询不影响回合本身，每个回合只多一次 prefill 加几个 token。

同时收集三种自评：
- `f_status`：`finish` 里的 status 是否为 success（M-all 下有意义；V 下等价于读 oracle）；
- `f_probe`：上面的探针概率（V 和 M-all 下都测）；
- `f_vlm`：开源 VLM（如 Qwen3-VL-8B）看 `finish` 时的 agent-view 和 wrist 图像加任务语言，输出成功概率。回合结束后离线批量跑。

### 2.4 分析（第 3 周末，半天到一天）

**C 部分**（每个 planner 都算）：
- `Y_final` 和 `Y_sticky` 在 V 与 M-all 下的配对差、95% CI；
- 虚报成功率 FC、错过成功率 MS、未结束率 NF；轮数分布；
- RL 与 SFT 的双重差分（试点阶段只看点估计和 CI，不做正式检验）。

**A 部分**（M-all 回合，以 `Y_final` 为真值）：
- 每种自评的 AUROC、ECE、Brier 分数、Murphy 分解（可靠性 / 分辨率 / 不确定性）；
- **PP 方差缩减潜力**：对 q ∈ {5%, 10%, 20%}，在 640 个回合上反复重抽标注掩码，计算 r̃ 的优势相对全标签优势的 MSE，比较"仅标签"、"PP 用 `f_probe`"、"PP 用 `f_vlm`"三者，并画出方差缩减与 Brier 分数的关系。这一张图就是 T3（方差恒等式）的第一个实证。

### 2.5 决策规则（试点前写定）

| 结果 | 判定 | 动作 |
|---|---|---|
| 任一 planner 的 `Y_final` 降幅 ≥ 5 个点，或 FC ≥ 10%，或 NF 增加 ≥ 10 个点 | **C 成立** | 进入 C1/C2 |
| `Y_final` 降幅 < 3 个点，且 FC < 5%，且 NF 增加 < 5 个点 | C 偏弱：泄漏存在但没被利用 | C 缩成一节放进 A 的论文，或者写短文；A 照常推进（它不依赖 C 的效应大小） |
| `f_probe` 或 `f_vlm` 的 AUROC 在 [0.65, 0.97] 之间，且 ECE > 0.05 | **A 成立**：有信号、有改进空间 | 进入 A 的实现 |
| 所有自评的 AUROC 都 < 0.65 | A 在 LIBERO 上信号不足（planner 从自己的观测里看不出成败） | A 的主实验移到 ALFWorld/WebShop，机器人降为次要；或者在 A 中加入 verifier 工具 |
| `f_probe` 的 AUROC > 0.97 且 ECE < 0.03 | 自评已近完美 | A 的标签效率会很高，但"校准改进"和"现象"主张变弱；改用 swap 套件或 LIBERO-Plus 等更难的设定重测 |
| 单回合耗时 > 30 分钟 | 算力超预算 | 按优先级砍掉可选阶段；优先尝试 FP8 服务、调低最大轮数，或降低评测重复数 |

---

## 3. 方案 A：自评 planner（Self-Grading），详细设计

### 3.1 动机

1. 真机没有成功 oracle。planner 必须自己判断"做成了没有"，而训练所需的成功标签要靠人工确认，很贵。C 的结果会给出这个缺口的量级。
2. 直接用 agent 的自报（或 VLM 判官）当奖励会被钻空子：agent 控制着评判看到的证据。用固定误差率去校正（PGBC 一类）又无法应对误差随策略漂移。
3. 关键观察：在 agentic 设定里，用来降低方差的控制变量可以由 agent 自己产出，而且可以通过激励设计让它如实报告、并主动变准。这把"标签稀缺"转化成了一个机制设计问题。

### 3.2 方法

**接口**：`finish` 增加 `confidence` 参数（0–1 的数，或从 {0.05, 0.2, 0.4, 0.6, 0.8, 0.95} 中选；Brier 对离散报告同样是恰当的）。prompt 中说明："confidence 是你认为任务目标此刻已满足的概率；你的奖励会因如实且准确的判断而提高。" 回合结束时没有调用 `finish` 的，记 p = 0。

**奖励**（每个回合一个标量）：
- ξ ~ Bernoulli(q)，在回合结束后抽取。被抽中时揭示 r = `Y_final`。
- r̃ = λp + (ξ/q)(r − λp)
- S̃ = −p² + (2p − 1) r̃
- R = r̃ + βS̃，β ∈ {0.5, 1}（T2 要求 β ≤ 1）

**标注概率**：q_i = clip(c · √(p_i(1 − p_i)), q_min, 1)，其中 c 使 Σq_i 等于每轮预算（期望 10%）。T4 表明，在 p 校准时这是方差最优的分配。

**λ**：按组划分两折交叉拟合，λ̂ = Σ w ξ r p / Σ w ξ p²（IPW 权重），截断到 [0, 1]。

**优势**：在 R 上做 LOO（与论文一致），或者用 E1 选出的 V̂(s₀) 基线；再乘一个常数，使优势的均方根与全标签 RLOO 一致（只改变步长，不改变方向）。

**落地位置**：
- `build_round.py` 新增 `--mode sg`；
- 新增 `rl/label_select.py`；
- RPent 补丁中 `finish` 增加 `confidence` 字段；
- `train_rl.py` 不改。

### 3.3 要证明的性质

详见 `IDEAS_v2.md` 1.4。

- **T1 策略防操纵**：对任意策略都有 E[r̃ | τ] = r。
- **T2 激励对齐**：如实报告是唯一最优；β ≤ 1 时期望价值关于后验成功概率单调；由凸性，agent 有动力在结束前收集验证证据，信息价值等于 β × 后验方差的减少量。
- **T3 方差恒等式**：标签校正项的方差 = (1 − q)/q × E[(r − λp)²]；λ = 1 时等于 (1 − q)/q × Brier(p)。
- **T4 最优求助**：q ∝ √(p(1 − p))，加 IPW 后保持无偏。

另外，用 Bayesian persuasion 刻画"只用代理奖励时 agent 操纵证据"的上界（判官理性时），作为现象部分的理论解释。写作时要说清前提：agent 只能选择展示什么，不能伪造证据。

### 3.4 实验

**A-E1 离线（第 6–7 周，约 2 GPU 天）**
- 数据：C1 的 M-all 回合（有 `f_probe` 和 `f_vlm`），加上 A-E2 第 0 轮（SFT 权重在 M-all + `confidence` 下的 384 个回合，同时作为各 arm 共享的第 0 轮）。
- 内容：
  - 各估计量（仅标签、只用代理、PGBC、PP、SG）在 q ∈ {5, 10, 20, 40}% 下的优势 MSE；
  - 每个 turn 样本做一次反向传播并存随机投影后的梯度草图，计算梯度余弦（梯度对优势线性，可以无限次重抽标注掩码）；
  - 方差与 Brier 的关系图。
- 冻结 E2 用的超参数：基线类型、λ 的截断方式、q_min、β、优势尺度。

**A-E2 在线（第 8–14 周）**

| arm | 内容 | 种子数 |
|---|---|---|
| R0 | 全标签 RLOO（M-all harness） | 2 |
| L10 | 仅标签 10% | 2 |
| S-self | reward = p | 1 |
| P10 | PP 用 p 作控制变量，不加 Brier 项 | 1 |
| SG10 | 本文方法，均匀标注 | 1 |
| **SG10-ask** | 本文方法，按 √(p(1−p)) 标注 | 2 |

每个 run 6 轮，每轮 32 组 × 4 回合，每个回合抽 8 个 turn，其余训练超参与论文一致。

**A-E3 通用性（ALFWorld / WebShop，Qwen3-4B）**
- 把 ALFWorld 包装成写 `result.json` 的 harness，直接复用本仓库管线。
- 同样要屏蔽环境的 "won" 信号；`finish` 带 `confidence`。
- 同一组 arm，每组 3 个种子（便宜）。

**A-E4 行为分析**：`finish` 前验证类动作的频率（看 wrist、调 `view_env_state`、`segment` 目标物体）随轮次的变化，用来检验 T2-iii 的预测"学会先验证再交卷"。

**A-E5 迁移**：LIBERO-Plus 子集上的校准和成功率。

### 3.5 指标与统计

| 主张 | 主指标 | 检验 |
|---|---|---|
| 只用代理会被钻空子，SG 不会 | S-self 的 p 均值、FC 率逐轮变化；SG 的 FC 率 | 趋势检验；效应预期很大 |
| SG 校准，RLOO 失准 | 每轮的 ECE、Brier、AUROC | 配对 bootstrap |
| 标签效率 | E1 中的梯度方差与余弦；E2 中达到同样成功率所需的标签数 | 离线重抽，功效高 |
| 10% 标签不劣于全标签 | `Y_final`，1600 回合配对 | 非劣界 3 个点，单侧 α = 0.05 |
| 优于仅标签 | 同上 | McNemar |
| 通用性 | ALFWorld/WebShop 上的同样排序 | 3 个种子 |

### 3.6 否决线（A）

| 编号 | 时间点 | 条件 | 处置 |
|---|---|---|---|
| KA1 | C0 结束 | 所有自评的 AUROC < 0.65 | 以 ALFWorld/WebShop 为主线，机器人降为次要，或加 verifier 工具 |
| KA2 | E1 结束 | q = 10% 时 SG 的优势 MSE 相对仅标签降低不到 20% | 撤下标签效率主张，转为"防操纵 + 校准"的方法文 |
| KA3 | E2 第 3 轮 | S-self 没有出现 p 上升或 FC 上升；SG 的 Brier 也没有改善 | 现象和校准主张都弱，重新评估 |
| KA4 | E2 结束 | SG10-ask 对 R0 的非劣性不成立 | 成功率主张撤下，保留其余主张 |

### 3.7 算力（单回合按 20 分钟）

| 项 | GPU 天 |
|---|---|
| A-E2：9 个 run × 约 1.6 天 | ≈ 14 |
| A-E2 最终评测（3 个 arm 各 1600 回合，其余各 800 回合） | ≈ 8 |
| A-E1 | ≈ 2 |
| A-E3（ALFWorld） | ≈ 4 |
| 合计（C 已承担试点） | ≈ 28；加 40% 余量约 39 |

C 约 7–10 个 GPU 天，两者合计约 7–8 周的 4 卡独占时间，与第 0 节的周计划一致。

---

## 4. 风险登记

| 风险 | 影响 | 应对 |
|---|---|---|
| RLinf LIBERO 环境在成功后无法继续运行 | M-all 无法按设计实现 | 第 1 周验证；备选在 env server 端绕过终止处理，或者成功后让动作照常执行但只冻结成功谓词的记录 |
| 72GB 卡同时放不下 vLLM、π0.5、SAM3 | 并发降低，耗时增加 | `episodes_parallel` 设为 2–3；SAM3 单独放一张卡，按卡分工 |
| 屏蔽 oracle 后 SFT/RL planner 大量耗尽 100 轮（一直等 `terminated`） | 评测耗时暴涨 | 本身就是 C 的发现；按 NF 记录；必要时把最大轮数降到 60，并在两个条件下保持一致 |
| C 的效应很小 | C 只能写成短文 | A 不依赖 C 的效应大小；C 仍提供无泄漏协议 |
| "RLCR + PPI"的审稿质疑 | A 的新意被低估 | 把 T3（agent 降低自己梯度的方差）、T1（agent 控制证据下的防操纵性）、多轮验证行为放在最前面 |
| AgenticRoboRL / Harness VLA 组自己发现泄漏 | C 被抢先 | C 的工程量小，尽快完成；第 7 周前挂 arXiv |

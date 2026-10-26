# 以 AgenticRoboRL 为基线的 A 会方向：近邻核验、创新性/可行性分析与初步实验设计

日期：2026-10-26。约束：4×72GB GPU；可直接用的开源权重为 [AgenticRoboRL-Qwen3.5-9B](https://huggingface.co/ciciciciliu/AgenticRoboRL-Qwen3.5-9B) 的 `sft/`（LIBERO-PRO 65.1%）和 `rl-v006/`（71.0%），都是 RPent harness 的 planner。代码层面的审查见 `research/CODE_REVIEW.md`。

## TL;DR

| 方向 | 一句话 | 创新性 | 4×72GB 可行性 | 建议投向 |
|---|---|---|---|---|
| **一（主押）** 无 oracle 的 harness planner 后训练：Prediction-Powered 策略梯度 | 现有 harness RL 默认回合内能看到仿真器的成功判定；去掉它以后，代理评判的误差会随策略漂移（agent 控制着评判看到的证据）。我们用"随机抽少量真标签 + 双稳健伪结局"得到无偏梯度，并给出标注预算的最优分配 | 高（本次检索未见同构工作） | 中高（主要瓶颈是 rollout 时间，靠离线梯度保真度实验压缩在线 arm 数） | NeurIPS 2027 / ICML 2027 主会 |
| 二（卫星） Compile or Keep：把 harness 记忆用无标签的 anchored privileged OPD 编进权重 | 用 RPent 现成的"有记忆/无记忆"两种配置做特权 on-policy 蒸馏，取代论文里 27B 改写+判官的离线流水线，并量化哪些记忆能编进权重、哪些必须留作测试时记忆 | 中（OPD 很拥挤，差异化在具身记忆和"编译/保留"的判据） | 中高 | CoRL 2027，或作为方向一论文的一节 |
| 三（可选） planner–VLA 双时间尺度 + 共享标签预算 | planner 用方向一慢更新，VLA 用你们的 PP-EM 做测试时更新，两层共用一份标签预算 | 中（层次后训练已有 HiRoC） | 中 | CoRL 2027，或方向一的部署章节 |
| 不建议单独开 | Idea D（turn 级信用分配）、Idea B 原表述（训小模型写 memory） | 低 | — | — |

相对 grok 分析的三处关键修正：

1. **grok 给 Idea A 的三个卖点里，有两个已经被别人发表**。"稠密/进度奖励会塌"已被 [Dark Room (2607.21273)](https://arxiv.org/abs/2607.21273) 用 74 个预注册 arm 在 ALFWorld/WebShop 上系统证明，并给出了代数原因（GRPO 的组内标准差归一化抵消了 shaping 系数）。"主动标注 + 伪标签"已被 [RLAVR (2605.25864)](https://arxiv.org/abs/2605.25864) 做了（单轮 RLVR 设定）。这两点只能当 baseline，不能当贡献。
2. **grok 没注意到 RPent 里的 oracle 泄漏**：planner 在回合内能读到 LIBERO 的 `terminated`（见 `CODE_REVIEW.md` 问题 1），所以直接在现有 harness 上做"少标签"是个伪问题。这反而成了方向一最有力的动机：现有 harness RL 的结果是在一个真机上不存在的信号之上取得的。
3. **时间线**：ICLR 2027 截稿已过。现实目标是 ICML 2027（约 1 月下旬，很紧）或 NeurIPS 2027（约 5 月）；CoRL 2027 约 5–6 月。具体日期以官网为准。

---

## 1. 文献核验

### 1.1 grok 引用的工作（逐篇打开 arXiv/ACL 页面核对）

| 工作 | 日期 | 核验 | 备注 |
|---|---|---|---|
| [AgenticRoboRL (2610.28589)](https://arxiv.org/abs/2610.28589) | 2026-10-20 | ✓ | 只做了 SFT 配方消融，**没有任何 RL 侧消融**（组大小、信用分配、奖励形式、行为分析都没有）；跑了 9 轮、报告第 6 轮。对比表里 Harness VLA + Claude Code (Opus 4.7) + 记忆为 82.4%，Codex (GPT-5.5) + 记忆为 72.1%。RL 相对 SFT：121 个回合翻成功、74 个翻失败 |
| [VLA Grounder (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1791/) | 2026-10 | ✓ | 冻结 VLA，用稀疏成功奖励 RL 训练指令改写策略；单轮、非工具调用 |
| [FIND (2609.32069)](https://arxiv.org/abs/2609.32069) | 2026-09-25 | ✓ | 真机，冻结 π0.5 + 残差 off-policy RL，VLM 比较前后观测自评成功，在线学习阶段无人工奖励标签 |
| [AtomicRL (2610.30097)](https://arxiv.org/abs/2610.30097) | 2026-10-21 | ✓ | 针对 VLA 的子任务语义门控信用分配（SAPO）+ 瓶颈自适应探索 |
| [TT-VLA (ACL 2026)](https://aclanthology.org/2026.acl-long.1863) | 2026-07 | ✓ | 用逐步进度估计做稠密奖励的 VLA 测试时 RL |
| [Recursive Harness Distillation (2609.33378)](https://arxiv.org/abs/2609.33378) | 2026-09-27 | ✓ | 强 agent 写 playbook 给轻 agent，用执行反馈递归改进，不更新参数 |
| [SERA (2610.29560)](https://arxiv.org/abs/2610.29560) | 2026-10-21 | ✓ | 可编辑 harness + 固定 VLA，rollout 驱动 harness 自演化，测试前冻结 |
| [SHAPER (2608.11350)](https://arxiv.org/abs/2608.11350) | 2026-08-11 | ✓ | 冻结模型，演化 skill 与 context-code harness |
| [BATON (2608.16889)](https://arxiv.org/abs/2608.16889) | 2026-08-17 | ✓ | **与 AgenticRoboRL 有共同作者**（Bingxin Xu、Yuzhang Shang）：这个组接下来大概率继续做长程 + 记忆，正面撞车风险高 |
| [Harness VLA (2607.08448)](https://arxiv.org/abs/2607.08448) | 2026-07-09 | ✓ | RPent 所在框架 |
| [RoboHarness (2607.18060)](https://arxiv.org/abs/2607.18060) | 2026-07-20 | ✓ | 异构策略编排 + 记忆 |
| [World Action Agent (2609.29964)](https://arxiv.org/abs/2609.29964) | 2026-09-24 | ✓ | **LIBERO-Pro 75.6%，已高于 AgenticRoboRL 的 71.0%**；也用 harness 轨迹微调了 Qwen3.5-9B。说明"刷 LIBERO-PRO 绝对分数"这条路已经很难 |
| [HiRoC (2608.05999)](https://arxiv.org/abs/2608.05999) | 2026-08-06 | ✓ | planner 出子目标 + executor RL 的层次后训练 |
| [LEGO-RL (2608.17393)](https://arxiv.org/abs/2608.17393) | 2026-08-18 | ✓，但 grok 的描述不准 | 这是**coding agent**（SWE-bench）的 harness-native RL，不是机器人；摘要里没有"要做比终局二值更丰富的 credit"这一说法 |

### 1.2 grok 漏掉、但审稿人一定会拿来比的近邻

**少标签 / 带噪验证器（方向一）**

- [RLAVR (2605.25864)](https://arxiv.org/abs/2605.25864)：RLVR 中主动挑少量样本要真标签，其余用伪标签；指出纯无监督 RLVR 容易塌缩。单轮推理设定，样本挑选是启发式的（CARE），摘要中没有无偏性保证。
- [PGBC/PGFC (2510.00915)](https://arxiv.org/abs/2510.00915)、[Noise-corrected GRPO (2510.18924)](https://arxiv.org/abs/2510.18924)：把验证器建模为**类条件、固定**的假阳/假阴率信道，做 backward/forward 校正。
- [Checking the Checkers (2610.17824)](https://arxiv.org/abs/2610.17824)：综述 58 篇验证器噪声工作，结论是"误差的结构比误差率更重要；策略学会触发的假阳性会导致平台或塌缩；验证器误差会随策略进步而漂移；多数缓解手段只针对一种误差测过"。**这几乎就是方向一要填的空白。**
- [Dark Room (2607.21273)](https://arxiv.org/abs/2607.21273)：稠密预测奖励在带标准差归一化的 GRPO 下必然塌缩，去掉标准差归一化即恢复。注意 AgenticRoboRL 本来就不除标准差。
- [FailBench (2609.03611)](https://arxiv.org/abs/2609.03611)：13 个 VLM 成功判官里最好的平衡准确率只有 0.77；证据模糊时系统性地偏向判"成功"。
- [VLMeg (2610.15401)](https://arxiv.org/abs/2610.15401)：把 VLM 当成功检测器而不是奖励回归器，做具身 RL。
- [Vision-Language Judges Prevent Reward Hacking in Robot Manipulation RL (NeurIPS 2026)](https://neurips.cc/virtual/2026/162911)：VLA 在 RL 中学会 reward hacking，用 VLM 判官监督。
- [MemReward (2603.19310)](https://arxiv.org/abs/2603.19310)：用 20% 标签做奖励传播，非无偏估计。

**特权信息 on-policy 蒸馏（方向二）**：[π-Distill (2602.04942)](https://arxiv.org/abs/2602.04942)（2026 年主会论文，ICLR/ICML 官网均有条目）、[AR-OPD (2606.10385)](https://arxiv.org/abs/2606.10385)、[RSD (2610.27495)](https://arxiv.org/abs/2610.27495)、[RetireOPD (2609.20784)](https://arxiv.org/abs/2609.20784)、[What Does Privileged Information Add (2609.20612)](https://arxiv.org/abs/2609.20612)、[OPSD 综述 (2608.25936)](https://arxiv.org/abs/2608.25936)（指出只有学生能从自身观测重建的特权信息才安全）。

**多轮信用分配（Idea D）**：[Coverage, Not Targeting (2609.02417)](https://arxiv.org/abs/2609.02417)（把优势重新分配到"进展"turn 上和分配到随机 turn 上同样有害）、[VETTA (2610.08402)](https://arxiv.org/abs/2610.08402)、[LLM Value Functions for Credit Assignment (2610.15840)](https://arxiv.org/abs/2610.15840)、[TRACE (2607.13988)](https://arxiv.org/abs/2607.13988)。

**其他 harness 自改进**：[KnowBody (2609.28530)](https://arxiv.org/abs/2609.28530)。

---

## 2. 方向一（主押）：无 oracle 的 harness planner 后训练

暂定题目：*When the Agent Holds the Camera: Prediction-Powered Policy Gradients for Oracle-Free Post-Training of Robot Planners*

### 2.1 为什么这个问题值得一篇 A 会

1. **问题是新的，而且是从基线里"挖"出来的。** AgenticRoboRL（以及 Harness VLA 的整套 LIBERO 结果）建立在一个设定上：planner 在回合内就能读到仿真器的成功判定 `terminated`，再决定是否 `finish`。真机上没有这个信号，planner 必须自己判断是否成功，而训练用的成功标签需要人工确认，很贵。
2. **agentic 设定带来一个单轮 RLVR 里没有的现象**：policy 控制着评判者看到的证据。planner 决定何时宣布结束、拍哪个视角、在文字里怎么描述结果。对代理评判（自报成功、VLM 判官）做优化，会把评判的假阳性率**推高**，而且这种误差依赖于当前策略。
3. **现有校正方法的假设正好在这里失效。** PGBC/PGFC/Noise-corrected GRPO 假设验证器误差是类条件、平稳的；Checking the Checkers 明确指出误差会随策略漂移、且缓解手段大多没测过可被利用的假阳性。
4. **我们的方法有可证明的性质**：每轮在当前策略的 rollout 上随机抽少量真标签，构造双稳健伪结局。无论代理误差怎样依赖策略都保持无偏；调好系数后方差不劣于"只用标签"；可以推出最优标注分配。这和你们 PP-EM 里的 V̂(s₀)、DR 伪结局、shared-label guard 是同一套工具，只是对象从 Θ(VLA) 换成了 planner。

### 2.2 设定

- 环境：RPent + 冻结 π0.5 + SAM3，LIBERO-PRO；起点为开源 `sft/` 权重。`rl-v006/` 作为"全标签 + 回合内 oracle"的参考上界。
- **无 oracle 变体**（需要给 RPent 加一个开关，约几十行）：从 primitive 返回值中去掉 `terminated` / `libero_terminated`，并改写 system prompt 里以 `terminated == true` 为目标的表述；额外记录 planner 调 `finish` 时刻环境是否成功。
- 两种成功口径都报告：sticky（论文口径：回合中任意时刻成功即算，便于和 71.0% 对比）和 final-state（planner 宣布结束时环境处于成功状态，这才是部署相关的口径）。
- 真标签 r ∈ {0,1}：只有被选中标注的回合才揭示（模拟人工确认）。预算为每轮标注比例 q 或总数 B。
- 免费的代理信号 f(τ) ∈ [0,1]，每个回合都有：
  - `f_self`：planner 自己的 `finish(status=success)` 声明；
  - `f_vlm`：开源 VLM 判官（如 Qwen3-VL）看最终 agent-view 和 wrist 图像加任务文本，输出成功概率（rollout 结束后离线跑，与 π0.5/SAM3 分时复用 GPU）；
  - `f_learned`：用历史标签拟合的小分类器（transcript 特征 + 上面两个分数），交叉拟合。

### 2.3 方法：Prediction-Powered 策略梯度（DR 伪结局）

对第 i 个回合，以已知概率 q_i 独立抽取是否标注 ξ_i ~ Bernoulli(q_i)（在 τ_i 产生之后抽），构造

$$\tilde r_i = \lambda f_i + \frac{\xi_i}{q_i}\,(r_i - \lambda f_i).$$

然后把 r̃ 原样代入 AgenticRoboRL 的 LOO 优势 `A_i = r̃_i − mean_{j≠i} r̃_j`，或者换成起始状态基线 `A_i = r̃_i − V̂(s₀)`。

要证明 / 验证的性质：

- **P1 无偏性**：E[r̃_i | τ_i] = r(τ_i)，对任意 f、任意 λ、以及 f 的误差对当前策略的任意依赖都成立；唯一要求是 ξ 在当前策略的 rollout 上新抽、q 已知。这是和 PGBC 系方法的本质区别（后者要求误差率平稳且类条件）。
- **P2 与"关于奖励线性"的算法兼容**：RLOO / Dr.GRPO（AgenticRoboRL 用的正是不除标准差的 LOO）代入 r̃ 后梯度无偏（PPO clip 带来的偏差与原算法相同）；带标准差归一化的 GRPO 关于奖励非线性，代入后有偏。这一点正好接上 Dark Room 的发现，给出一个统一的解释。
- **P3 安全退化**：λ 用交叉拟合估计（用别的组/折的标签，避免同一批标签既拟合又校正带来的偏差）。f 没有信息量时 λ → 0，估计量退化为"只用标签"的 IPW 估计；λ 取最优值时方差不超过它（PPI++ 风格的论证）。
- **P4 最优标注分配**：在 Σq_i = B 约束下最小化方差，得到 q_i ∝ √E[(r − λf)² | τ_i] · ‖g_i‖。实际可用 √(f̂_i(1 − f̂_i)) 乘以"组内其他回合伪结局的离散度"（决定这条标签对优势有多大影响），并截断到 [q_min, 1]。用 IPW 权重 1/q_i 保持无偏；不做加权的主动挑选（RLAVR 一类）一般有偏。
- **起始状态基线**：V̂(s₀) 用别的组的 r̃ 交叉拟合（特征为 task、init、上一轮该 task 的成功率），从而在总回合数固定时把组大小从 6 降到 2–4，换取更多初始状态覆盖。这是你们已有的 V̂(s₀) 资产。
- **Goodhart 监测与守卫**：每轮在随机标注子集上估计 ρ̂ = corr(f, r) 和代理假阳性率 FP̂。ρ̂ 下降或 FP̂ 上升超过阈值时，重拟合 `f_learned`、提高 q，或回滚到上一轮（即 shared-label guard）。

在本仓库里的落地只需改 build 侧：优势仍是每个回合一个标量，`rl/train_rl.py` 不用动。

| 改动 | 位置 |
|---|---|
| RPent 开关 `--hide-terminated` + 记录 finish 时刻的真实状态 | `patches/rpent/` 新增一个 patch |
| VLM 判官打分 | 新增 `rl/judge.py` |
| 每轮选标注集、写出 q_i | 新增 `rl/label_select.py` |
| 新模式 `--mode pp`：r̃、交叉拟合 λ、LOO 或 V̂(s₀) 基线，记录 ρ̂、FP̂、λ、标签数 | `rl/build_round.py` |
| adapter 输出代理分数、按标注掩码揭示真值 | `rl/adapters/rpent_libero.py` |
| 在 rollout 和 build 之间插入 label 阶段 | `rl/local_ctl.py` |

### 2.4 实验设计

**E0 试点（约 2–3 天，决定是否继续）**

- 在 4×72GB 上实测：每个回合的耗时、每卡能并行几个回合、训练端的显存上限（`--ce_chunk 512` 下 `--max-total-tokens` 能放到多少）。后面所有算力估计都以这里的实测为准。
- SFT 权重在验证 init（80 任务 × 5 init = 400 回合）上分别跑"oracle 可见"和"无 oracle"两种配置，测：sticky 与 final-state 成功率的落差；`f_self`、`f_vlm` 相对真值的 FP/FN；planner 提前宣布成功的比例。
- 否决条件：无 oracle 后 final-state 成功率几乎不掉（< 3 个点），且 `f_self` 准确率 ≥ 97%。这说明 LIBERO 上的问题太简单，需要换更难的设定（swap 套件、LIBERO-Plus 扰动、或小规模真机）再讲这个故事。

**E1 离线梯度保真度（便宜，不需要新 rollout，用来做大范围扫参）**

- 跑一轮全标签 RLOO 并保留所有回合与代理分数。在同一批回合上反复重抽标注掩码，比较各估计量的优势和梯度相对全标签版本的偏差与方差（优势 MSE；在子集上用 `train_rl.py` 的 pre-pass 算真实梯度的余弦相似度）。
- 估计量：只用标签的子采样 RLOO、只用代理、PGBC（第 0 轮估计误差率）、RLAVR 风格混合、PP-均匀、PP-主动。标注比例 q ∈ {5%, 10%, 20%, 40%}。
- 否决条件：q = 20% 时 PP 的梯度余弦相似度不优于"只用标签"，说明代理信号信息量不够，停止。
- 局限：离线实验看不到策略对代理的适应（Goodhart），这部分必须在线验证（E2、E3）。

**E2 在线主实验**

- 每个 arm 跑 6 轮，每轮 32 组 × 4 回合 = 128 回合（论文是 64 × 6 = 384）。
- arm 列表：全标签 RLOO（上界）、只用标签 @q、只用代理（`f_self`）、只用代理（`f_vlm`）、PGBC @q、PP-均匀 @q、PP-主动 @q。主设定 q = 10%；PP 和"只用标签"再加 q ∈ {5%, 20%} 画预算曲线。主要 arm 跑 2 个种子。
- 评测：开发阶段用 200 回合的子集，最终在 800 回合（8 套件 × 10 任务 × init 1–10）上评测，两种成功口径都报；再加 LIBERO-Plus 子集测迁移（论文中 RL 在 Light 上提升最大，+7.0）。

**E3 Goodhart 诊断（论文的"现象"部分）**

- 每轮在随机标注子集上画各代理的 FP/FN 曲线、提前宣布成功的比例、回合长度、视角选择统计，并配典型 hacking 案例。
- 预期：只用代理的 arm 中 FP 上升、真实成功率进入平台期或下降；PGBC 因误差率漂移而逐渐偏离；PP 保持跟踪真实目标。

**E4 通用性（便宜，2–3 张卡，冲 ML 主会时很重要）**

- 在 ALFWorld / WebShop 上用 Qwen3-4B 和 LLM 判官作代理，跑同一组估计量，和 Dark Room、RLAVR 的设定直接对齐。

**E5 消融**

- λ 固定 vs 调优；交叉拟合 vs 不交叉拟合；固定总回合数下 V̂(s₀) vs LOO（组大小 G ∈ {2, 4, 6}）；有无守卫；有无 IPW 校正的主动挑选；带与不带标准差归一化（验证 P2）。

**主张与证据对照**

| 主张 | 证据 |
|---|---|
| 现有 harness RL 依赖回合内 oracle，去掉后问题变难 | E0 |
| 对代理优化会让代理误差随策略漂移，固定误差率的校正失效 | E2 + E3 |
| PP 在任意策略相关误差下无偏，λ 调好后不劣于只用标签 | 定理 P1、P3 + E1 |
| 少量标签即可接近全标签 | E2 的预算曲线（预期 q = 10–20% 接近全标签 RLOO） |
| 不限于机器人 | E4 |

### 2.5 算力粗估（必须用 E0 实测替换）

已由 `research/PREREG_direction1.md` 第 8 节取代。原估计低估了训练耗时：每回合 16 个 turn 时一轮训练约 3 小时，不是 1–1.5 小时。修正后，全部实验约需独占 4 卡 30 天，预留重跑余量后约 40–45 天。

### 2.6 风险与应对

- **LIBERO 上代理信号太准**：任务结果视觉上很明显，`f_self` 在去掉 oracle 后可能仍然很准。应对：E0 先测；改用只有 `f_vlm` 的 arm；换 swap 套件或 LIBERO-Plus 扰动；必要时补一个小规模真机实验。
- **去掉 oracle 后 SFT 权重分布偏移**：SFT 时 planner 是看着 `terminated` 学会何时 `finish` 的。把这个掉点本身作为发现报告，所有 arm 从同一个起点出发，比较是公平的。
- **q 很小时 IPW 方差大**：λ 调优、设定 q_min、必要时对 r̃ 做截断（引入有界偏差，在论文里给出界）。
- **撞车**：RLAVR 的作者或 AgenticRoboRL 组可能跟进。核心护城河是"策略相关误差下的无偏性 + agentic 证据操控"这一组合，尽早占位（可先挂 arXiv）。

---

## 3. 方向二（卫星）：Compile or Keep，把 harness 记忆编进权重

### 3.1 动机（直接来自本仓库）

- AgenticRoboRL 的 SFT 阶段花了很大力气处理"教师看得到记忆、学生看不到"的问题：79,973 个教师 turn 里，只做掩码时能用 27,110 个；用 27B 模型改写再经判官筛选后能用 40,047 个（`teacher_rewrite/`）。SFT 配方的不同让成功率在 55.2% 到 64.6% 之间变化，这是论文唯一的消融。
- RPent 本身就提供了一对天然的特权/非特权配置：`--knowledge-profile mem-proven`（有记忆）和 `proven`（无记忆），同一个 harness、同一套工具。

### 3.2 方法

- 学生在 `proven` 配置下 rollout（就是现在的 RL rollout）。对抽中的每个 turn，用同一个教师在两种上下文下给学生采样出的 token 打分：有记忆 ℓ_P、无记忆 ℓ_A（锚点）；学生自身 ℓ_S。
- 逐 token 优势：`a_t = (ℓ_A − ℓ_S) + β · clip(ℓ_P − ℓ_A, −c, c)`。后一项是"记忆带来的残差"，用 β 控制特权引导的强度。思路与 AR-OPD 相近，但锚点是同一 harness 的无记忆配置。
- **新的部分：判断哪些记忆能编进权重（可推断性门控）。** 把记忆按条目 m_k 拆开（全局规则、任务级策略、实例级数值配方）。用"留一条目"打分 `ℓ_P − ℓ_{P∖m_k}` 把残差归因到条目。经过一轮 OPD 后，看归因到 m_k 的残差在留出的 init 上是否缩小：缩小说明可以编进权重；不缩小（例如与具体初始状态绑定的数值偏移）说明学生从自身观测无法重建，应保留为测试时的精简检索记忆。
- 最终得到"权重 + 小型残差记忆"的学生。比较成功率、记忆 token 数，以及 LIBERO-Plus 扰动下的鲁棒性（预期实例级数值记忆在布局扰动下迁移很差，编译还是保留的选择在分布偏移下最关键）。
- 整个过程不需要成功标签，可以和方向一组合：标签稀缺时作为稠密的正则信号。

### 3.3 实验与可行性

- 基线：AgenticRoboRL 的 SFT（仅掩码 55.2 / 改写 64.6–65.1）与 RL（71.0）；教师 + 记忆（74.0）；学生在测试时带完整记忆；不带锚点的全特权 OPD；带锚点残差但不做门控；只写 playbook、不更新权重（RHD/SERA 式）。
- 教师选择：论文的 27B 教师（如果权重可得；bf16 约 54GB，只做 prefill 打分，可用 vLLM 的 prompt logprobs，长上下文时 TP=2）；或者自教师，即 `rl-v006` 9B 在上下文中加记忆。后者更省，但需要先验证 9B 加记忆后是否确实比不加更强。
- 代码改动：`rl/rl_loss.py` 目前只接受标量优势，要支持逐 token 优势；`rl/train_rl.py` 读入逐 token 数组；新增 `rl/teacher_score.py`；`build_round.py` 新增不要求分组的 `--mode opd`。
- 否决条件：在学生访问到的状态上，教师有记忆相比无记忆的成功率提升不到 5 个点，说明特权信号太弱。
- 创新性判断：特权 OPD 已经很拥挤（π-Distill 已是 2026 年主会论文）。差异化只能落在"具身记忆中有一部分无法从观测重建"以及"编译还是保留"的判据和规律上；只做出"OPD 比 SFT 好"不够发主会。更现实的定位是 CoRL 2027，或者方向一论文中的一节。

---

## 4. 方向三（可选）：planner–VLA 双时间尺度 + 共享标签预算

- planner 用方向一的 PP 策略梯度慢更新（或直接冻结 `rl-v006`），VLA 用你们的 PP-EM 做 no-reset 测试时更新；同一份回合级标签同时服务两层。研究问题：标签在两层之间怎么分配，何时触发重规划或回滚。
- 与 [HiRoC](https://arxiv.org/abs/2608.05999) 的区别：HiRoC 是离线、用 oracle 奖励训练 executor；这里是部署期、少标签、带守卫。
- 风险：容易被看成系统拼盘。需要一个清晰的分配定理，或者干净的"只训 planner / 只训 VLA / 两者都训"在相同标签预算下的对比。
- 建议作为方向一的部署章节，或者单独投 CoRL；算力上 planner 冻结时 4 卡基本都可以给 VLA。

---

## 5. 不建议单独开的方向

- **Idea D（turn 级信用分配）**：[Coverage, Not Targeting](https://arxiv.org/abs/2609.02417) 直接给出反例（把优势集中到"进展"turn 上和集中到随机 turn 上同样有害），再加上 VETTA、TRACE、LLM value function 等工作。只在 RPent 一个 harness 上做 option 级优势，很容易被看成 AgenticRoboRL 的消融。可以把"组大小与 V̂(s₀) 基线"放进方向一的 E5。
- **Idea B 原表述（训小模型写 memory 与权重 RL 对比）**：RHD、SERA、SHAPER、RoboHarness 已经占了"写经验"这条线。保留下来的有价值部分（权重与记忆的取舍）已经并入方向二的"编译还是保留"。
- **刷 LIBERO-PRO 绝对分数**：World Action Agent 已到 75.6%，Claude Code + 记忆为 82.4%，用 4 卡很难在这条线上胜出。

---

## 6. 下一步（按顺序）

1. 修好 `CODE_REVIEW.md` 中的 4 卡适配项（`nproc_per_node`、`--ce_chunk 512`、`--max-total-tokens`），下载 `sft/` 和 `rl-v006/`，跑通一轮 eval。
2. 写 RPent 的 `--hide-terminated` patch，完成 E0 试点并做继续/否决决策（约 1 周）。
3. 实现 `build_round.py --mode pp`、`rl/judge.py`、`rl/label_select.py`，跑 E1（约 1 周）。
4. 把方向一写成一页预注册（arm 列表、标注预算、主指标、否决线），然后启动 E2。
5. 方向二只先做一个最小验证："9B 自教师加记忆是否比不加强"（1–2 天）。根据结果决定是否投入。

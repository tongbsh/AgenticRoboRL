# 文献审计预注册（方案 C，`research/PLAN_C_then_A.md` 1.7 节）

**本文件必须在阅读任何候选论文的方法细节之前填写并提交（git 记录即时间戳）。** 之后只能追加"偏离记录"，不能改动已写的标准。

## 1. 研究问题

已发表的"LLM/VLM planner + 仿真器"机器人工作中，有多大比例在评测时把仿真器的成功判定提供给了 planner？

## 2. 纳入标准（同时满足）

1. 2025-01-01 至 2026-10-31 发表于 arXiv 或同行评审会议/期刊；
2. LLM 或 VLM 作为 planner/agent，在循环中做多步决策（调用技能、写代码或调用工具）；
3. 在带程序化成功判定的仿真器中评测（LIBERO 系列、RoboCasa、ManiSkill、MetaWorld、RoboTwin、CALVIN、SimplerEnv 等）；
4. 报告任务成功率。

排除：纯端到端 VLA（没有 planner 循环）；只做真机的工作；综述。

## 3. 检索（固定，填写后不再更改）

- arXiv 检索式（cs.RO 或 cs.AI，日期范围同上）：
  - `(LLM OR VLM OR "language model") AND (agent OR planner OR harness) AND (LIBERO OR RoboCasa OR ManiSkill OR MetaWorld OR RoboTwin OR CALVIN OR SimplerEnv)`
  - 检索执行日期：2026-10-30（执行细节见 `search_log.md`）
- 种子论文的引用与被引：Harness VLA (2607.08448)、AgenticRoboRL (2610.28589)、CaP-X、2601.20334。
- 记录：检索命中数 → 标题/摘要筛选后数 → 全文纳入数（PRISMA 式流程）。

## 4. 编码（每篇一行，`coding_sheet.csv`）

| 字段 | 取值 | 说明 |
|---|---|---|
| `success_visible_eval` | yes / no / unclear | 评测时 planner 能否在回合内读到仿真器成功判定（工具返回、状态、prompt 指示、环境自动终止并告知） |
| `success_visible_train` | yes / no / unclear / na | 训练或经验收集时是否可见（无训练填 na） |
| `success_metric_only` | yes / no | 成功判定只用于统计指标 |
| `privileged_state` | yes / no / unclear | planner 能否访问真值物体位姿等仿真特权状态 |
| `auto_terminate_on_success` | yes / no / unclear | 环境是否在成功时自动结束回合（动力学通道） |
| `evidence` | code / text | 判断依据：代码核实（给出文件和行号）或仅凭论文文字 |
| `evidence_ref` | 文本 | 代码位置或论文章节 |

## 5. 流程

- 两名编码者独立编码，各自填一份 `coding_sheet_<name>.csv`；
- 报告 `success_visible_eval` 的 Cohen's κ；
- 分歧讨论后形成 `coding_sheet.csv`，在偏离记录中写明讨论结果。

## 6. 事先写定的叙事门槛

- 纳入论文中 `success_visible_eval = yes` 的比例 **≥ 30%**：在 C 的论文里使用"领域级问题"的说法。
- 否则：C 定位为"一个 harness 的案例研究 + 无泄漏协议"，文献表作为背景。

## 7. 偏离记录

（日期 — 内容 — 原因）

- 2026-10-30 — arXiv 检索通过 API 的 `all:` 字段执行（标题、摘要、作者、注释，不含全文）— 预注册未指定字段，这是 API 的默认全字段检索；作者名造成的误命中在筛选中排除。
- 2026-10-30 — 种子论文的引用与被引用 Semantic Scholar 获取，只保留 2025 年及以后、有 arXiv 编号的论文；CaP-X 因 API 限流暂未获取，之后补上 — 预注册未指定数据源。
- 2026-10-30 — 编码者 1 为 LLM（Claude），编码者 2 为人工；筛选只由编码者 1 完成 — 人力限制；论文中如实报告，κ 衡量的就是 LLM 与人工编码的一致性。

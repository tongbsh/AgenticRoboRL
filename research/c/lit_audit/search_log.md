# 文献检索记录（方案 C 文献审计，预注册见 `PREREG.md`）

## 1. 检索执行（2026-10-30）

**arXiv**：通过 arXiv API（export.arxiv.org/api/query）执行预注册检索式。API 的 `all:` 字段覆盖标题、摘要、作者和注释，不含全文：

```
(cat:cs.RO OR cat:cs.AI) AND (all:LLM OR all:VLM OR all:"language model") AND (all:agent OR all:planner OR all:harness)
AND (all:LIBERO OR all:RoboCasa OR all:ManiSkill OR all:MetaWorld OR all:RoboTwin OR all:CALVIN OR all:SimplerEnv)
AND submittedDate:[202501010000 TO 202610312359]
```

命中 62 篇，原始结果见 `search_arxiv_2026-10-30.csv`。`all:` 也匹配作者名（例如名为 Calvin 的作者），因此有一批与机器人无关的命中，在筛选中排除。

**种子论文的引用与被引**：通过 Semantic Scholar Graph API 获取，只保留 2025 年及以后、有 arXiv 编号的论文：

| 种子 | 被引 | 参考文献 |
|---|---|---|
| Harness VLA (2607.08448) | 86 | 77 |
| AgenticRoboRL (2610.28589) | 0（刚发布） | 55 |
| 2601.20334 | 8 | 37 |
| CaP-X (2603.22435) | 尚未获取（API 限流） | 尚未获取 |

去掉与 arXiv 命中重复的，新增 133 篇，原始结果见 `search_seeds_2026-10-30.csv`。

## 2. 流程计数（PRISMA 式，随进展更新）

| 步骤 | 数量 |
|---|---|
| 识别：arXiv 检索 | 62 |
| 识别：种子引用与被引（去重后新增） | 133 |
| 合计（去重） | 195 |
| 第一轮筛选（标题；arXiv 命中另看了摘要）：纳入 / 待定 / 排除 | 46 / 69 / 80 |
| 第二轮筛选（摘要，处理待定） | 待做 |
| CaP-X 的引用与被引 | 待做 |
| 全文纳入并编码 | 待做 |

逐篇的筛选决定与理由见 `screening.csv`。排除理由的分类：非机器人；端到端 VLA 或策略模型（没有 planner 循环）；VLM 反馈奖励或成功检测器；benchmark、仿真器或基础设施；综述或基础模型报告；LLM 不在控制循环里。

## 3. 编码者

- 编码者 1：Claude（LLM），编码表为 `coding_sheet_claude.csv`。证据按预注册分两级：读过代码的写 `code` 并给出文件和行号；只看论文文字的写 `text`。
- 编码者 2：需要一位人工编码者，独立填写 `coding_sheet_<名字>.csv`。**在完成自己的编码之前，不要看编码者 1 的表**，否则 κ 没有意义。
- 筛选只由编码者 1 完成。预注册没有要求双人筛选；论文里如实写明。

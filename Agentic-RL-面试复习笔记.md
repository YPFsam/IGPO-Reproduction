# Agentic RL 面试与项目复习笔记

> 覆盖:验证集构建 / Reward Hacking / 长程搜索任务 / Repo-level Coding Agent / 数据合成 / 长轨迹信用分配与 PRM / 控制流训练 / **GRPO vs PPO 长程适配性(2025-2026 最新)** / 模拟面试 Q&A
>
> 用法:每节末尾有「速记金句」,面试直接用。模拟面试在第 11 节。

---

## 目录

0. [项目背景速查](#0-项目背景速查2wiki-agentic-search)
1. [验证集构建](#1-验证集构建)
2. [Reward Hacking 检测(剪刀差法)](#2-reward-hacking-检测剪刀差法)
3. [长程搜索任务:数据/指标/评估](#3-长程搜索任务数据指标评估)
4. [Repo-level Coding Agent 数据](#4-repo-level-coding-agent-数据)
5. [数据合成:搜索 + 代码](#5-数据合成搜索--代码)
6. [长轨迹信用分配与 PRM 全流程](#6-长轨迹信用分配与-prm-全流程)
7. [控制流训练:何时搜/停/验证](#7-控制流训练何时搜停验证)
8. [【重点】GRPO vs PPO:长程适配性深度分析](#8-grpo-vs-ppo长程适配性深度分析)
9. [Prompt 从哪来(in-context vs 训练)](#9-prompt-从哪来in-context-vs-训练)
10. [模拟面试 Q&A(9题 + 压轴)](#10-模拟面试qa9题--压轴)
11. [数据集清单](#11-数据集清单)
12. [速记金句 & 电梯陈述](#12-速记金句--电梯陈述)
13. [附录:关键论文与资源](#13-附录关键论文与资源)

---

## 0. 项目背景速查(2wiki agentic search)

| 维度 | 事实 |
|---|---|
| 任务 | 2WikiMultihotQA 多跳问答 + agentic 检索(Search-R1/R1-Searcher 风格) |
| 框架 | veRL(IGPO 变体,GRPO 系) |
| 模型 | Qwen2.5-1.7B + LoRA |
| 检索库 | tantivy,391694 passages,8890 端口 |
| 动作标签 | `<think>` / `<tool_call>` / `<answer>`,工具只有 `web_search` |
| 训练集 | `train_2wiki.parquet` 24923 条(同质:`data_source=2wiki`,`ability=search & QA`) |
| 验证集 | `dev.parquet` 875 条,smoke `dev_2wiki.parquet` 125 条 |
| 测试集 | `test_2wiki.parquet` 512 条 |
| 可检索过滤 | `answerable_indices.json`(15913/167k)+ `train_2wiki_filtered.parquet` |
| 当前验证配置 | `TEST_FREQ=-1`(关闭周期验证)、`val_before_train=false` |

**实测开销(tensorboard,step 40)**:每 step 226.8s,其中 generation 168.5s(占 74%),save_checkpoint 40.2s,`response_length/mean`≈2104 token。generation 占大头,因为多轮 agentic(MAX_TURNS=10)。

**原始 2WikiMultihotQA 字段**:`_id, type, question, context, supporting_facts, evidences, answer`。**预处理后训练/验证集都丢弃了 `supporting_facts`/`evidences`——这是做过程诊断的最大缺口,评估集应该补回来。**

> **速记**:数据极简(只存「一句问题 + ground-truth 短答案」),复杂度全在 system prompt 模板和 reward 侧。

---

## 1. 验证集构建

### 通用原则
1. **与训练严格隔离 + 同分布**。RL 极易 reward hacking,验证集是发现"reward 涨但泛化崩"的唯一手段。
2. **用 ground-truth/外部裁判,不用训练 reward 函数**。训练 reward 含各种 shaping(格式/长度/工具奖励),验证指标必须是最干净的"答案对不对"。
3. **规模几百~几千即可**,agentic 验证很贵。

### 三层验证概念(别混淆)
| 概念 | 作用 | 关键 |
|---|---|---|
| **dev** | 选 ckpt / 早停 / 调超参 | 训练全程可看 |
| **test** | 最终汇报 | **只跑一次,别偷看** |
| 离线评估 | ckpt → 完整多轮 rollout → F1/EM/MBE | 训练后独立流程 |

### 周期性验证 vs checkpoint(易混点)
- veRL 里 `test_freq`(验证频率)和 `save_freq`(存 ckpt 频率)**是两个独立参数**。
- 周期性验证**不需要 ckpt**:直接拿当前 actor in-memory 权重 rollout。
- 离线评估(`evaluate.sh` + `cacluate_metrics.py`)**才需要 ckpt**。

### 耗时估算
验证比训练单步贵:每条要跑完整 episode(greedy/低温度、不截断)。**最准的方法:先在 `smoke/dev_20`(20 条)计时,按 875/20 放大**。一次验证开销 ≈ dev 条数 / 训练单 step 样本数 × 226.8s。

### 指标体系(本项目三层)
- **训练中(tensorboard)**:`critic/score/mean`、`critic/rewards/mean`(训练奖励视角)
- **离线评估(`cacluate_metrics.py`)**:
  - **EM**:归一化后字面相等
  - **F1**:token 集合交集的 P/R(简化版 set-F1)
  - **MBE**:GPT-4o-mini 语义判断(pred 与 GT 语义一致),比 EM 宽松

> **速记**:dev 选 ckpt,test 只跑一次,离线评估靠 ckpt。三层指标:训练 reward / EM / F1 / MBE。

---

## 2. Reward Hacking 检测(剪刀差法)

### 核心矛盾
如果验证 100% 复用训练 reward,只能看**同分布泛化/过拟合**,对 hacking 盲目。检测 hacking 需要**与训练 reward 解耦的指标**。

### 本项目的解耦设计(代码已实现)
验证时同时算三套分数(`ray_trainer.py:805-808`):

| tensorboard tag | 含义 | 含 shaping? |
|---|---|---|
| `val/test_score/2wiki_f1` | **noformatf1**(纯答案匹配) | ❌ 无 |
| `val/test_score/2wiki_f1_format_penalty` | 带格式奖励/惩罚的 f1 | ✅ 有(≈训练 reward) |
| `val/test_score/2wiki_em` | Exact Match | ❌ 无 |

外加训练步的 `critic/rewards/mean`。验证时 `do_sample=False`(greedy 可复现)、`global_steps=-self.global_steps`(标记 validation)。

### 剪刀差判据(核心)
```
情况 A(真本事):   rewards↑   f1_format_penalty↑   f1↑        → 健康
情况 B(刷格式):   rewards↑   f1_format_penalty↑   f1 平/↓   → ⚠️ hacking
情况 C(格式也崩): rewards↑   f1_format_penalty 平  f1↓        → ⚠️ 严重 hacking
情况 D(过拟合):   训练 rewards↑   dev 的 f1 平/↓        → 泛化退化
```
**训练 reward / f1_format_penalty 涨,但 noformatf1 不涨甚至掉 = reward hacking。**

### noformatf1 仍不够独立
它和训练 reward 共享同一 `compute_score`(同 GT、同 set-F1 规则)。能抓**格式 hacking**,抓不了**针对答案匹配规则本身的 hacking**(短答案刷 set-F1)。完全脱钩的信号(独立性从强到弱):
1. **MBE**(LLM-judge,已内置)
2. **人工抽检**(30~50 条)
3. **退化指标**(response_length、工具调用次数、distinct-n)
4. **检索中间信号**(supporting fact 命中率,需补字段)

> **速记**:剪刀差 = 带格式分 vs 不带格式分。抓格式 hacking 靠剪刀差,抓答案规则 hacking 靠 MBE/人工/退化指标。

---

## 3. 长程搜索任务:数据/指标/评估

### 长程 vs 普通多跳的本质区别
| | 单跳/普通多跳 | 长程搜索 |
|---|---|---|
| 步数 | 1~2 hop | 3~10+ step |
| 规划 | 弱 | 强(拆解、回溯、纠错) |
| 错误传播 | 局部 | **级联放大** |
| 信用分配 | 容易 | 难(失败不知哪跳断) |
| 数据核心 | 答案对不对 | **还得标注 reasoning chain** |

### 数据构造三法
1. **模板合成**(2wiki 路线):知识图谱选 seed → 沿关系走 N hop → 模板生成问题+gold 链。hop 可控、答案必然可达、可批量;分布窄。
2. **真实查询挖掘**:搜索日志/问答社区。分布真实但 hop 不可控,链要人工标。
3. **LLM 反向生成**:给 gold answer+supporting docs 生成需多跳的问题。要过滤掉单跳就能答的。

**必标字段**:`question`, `answer_set`(多 reference), `rubric`, `gold_chain`(每跳桥接实体+支持文档), `hop_count`, `type`, `difficulty`。

### 保证可检索(核心工程问题)
混入"检索器找不到答案"的样本 → reward 恒为 0 → 模型学到放弃/乱答 → 奖励噪声污染训练。
1. **离线 recall 验证**:对每问题检索 top-k,查 gold supporting docs 是否在内,记 `recall@k`。
2. **逐跳验证**(更强):多跳拆成 bridging query 分别查,很多"问题检索不到"拆成每跳后能查到 → **保留而非剔除**。
3. **oracle recall 阈值过滤**:只留 oracle recall=1 的样本。
4. **gold passage 保证在库**:合成法天然满足。
5. **记录 retriever recall 元数据**:训练时对 recall<1 降权。

### 指标三层
- **结果层**:EM / F1 / Accuracy(MBE 语义兜底)
- **过程层**(长程灵魂):
  - 检索 recall(supporting_facts 命中率)
  - **断链率**(哪跳丢了,需 gold_chain)
  - 轨迹效率(平均搜索轮次、token)
  - 终止行为(及时 `<answer>` vs 撞 MAX_TURNS)
  - 检索 precision(返回噪音占比)
- **退化层**:response_length、工具调用次数、distinct-n

### 评估集构造 + 方法
- 分层抽样:按 `type`/`hop_count`/实体类型,每层 ≥50~100 条
- **必须保留 `supporting_facts`/`gold_chain`**(本项目丢了,要补)
- 离线 rollout(greedy)→ 分层算指标 → **hop-stratified 汇报**(2-hop vs 5-hop 准确率,信息量最大)→ LLM-judge → 人工抽检

### 判断与验证
- **能力归因**:hop-stratified 分离推理 vs 检索;trajectory dump 定位断点(搜不到 vs 搜到没读懂)
- **防 hacking**:final EM 涨但 per-hop recall 掉 → 幻觉蒙;yes/no 类异常高 → 瞎猜

> **速记**:长程数据必须带 gold_chain,否则过程诊断全废。指标三层:结果/过程/退化。hop-stratified 是归因神器。

---

## 4. Repo-level Coding Agent 数据

### 杀手锏:测试即免费 verifier
代码任务的 reward 几乎免费(`reward = 测试 pass`),且 `run_tests` 是**天然可验证的 verify 动作**——上一轮说的"verify 难涌现"在代码任务里基本消失。

| | 单函数(HumanEval) | repo-level(SWE-bench) |
|---|---|---|
| 上下文 | 几百 token | 几十万行 |
| 任务 | 填空 | **诊断 issue → 定位 → 跨文件改** |
| 验证 | 单 unit test | 整套 test suite + 不能破坏现有功能 |
| Reward | 标准答案 | **测试 pass = 免费 reward** |

### SWE-bench pipeline(淘汰率极高,几万 PR → 2000+ 实例)
```
① 选仓库(有测试 + 能 docker 化)
② 选 PR(改代码 + 关联 issue + 改了测试)
③ 验证 FAIL_TO_PASS:改的测试【改前fail、改后pass】← 命脉
④ 验证 PASS_TO_PASS:gold patch 不破坏已有测试(防回归)
⑤ docker 能干净复现
```
**筛选做不对,RL 训出的是"绕过坏测试"的 hacking 模型。**

### instance schema
```python
{
  "instance_id", "repo", "base_commit",      # PR 之前的版本
  "problem_statement",                         # agent 输入:issue
  "patch",                                     # gold diff(评估/可选 SFT)
  "FAIL_TO_PASS": [...],                       # 关键:改前fail改后pass
  "PASS_TO_PASS": [...],                       # 不能破坏
  "environment_setup"
}
```

### 训练框架(接控制流思路)
- 工具:`read_file` / `search_code` / `edit_file` / `run_tests` / `run_lint` / `bash`
- **冷启动 SFT 必做**(否则探索不动几万行 repo)
- **RL reward**:`+1·(FAIL_TO_PASS 全过) − α·(PASS_TO_PASS 退化) − β·步数/改动量`
- stop/verify 好训:`run_tests` 给天然停止信号 + verify 动作;credit assignment 仍难

### 质量难点
可验证率(<10% 存活)、环境复现率、跨文件改动比例(repo-level 程度)、**预训练污染**(用 base_commit 在截止前 / held-out 仓库)、PASS_TO_PASS 防回归、难度分层(base agent pass rate)。

> **速记**:QA 花力气造 reward,代码花力气筛数据。FAIL_TO_PASS 是 reward/评估/正确性的命脉。

---

## 5. 数据合成:搜索 + 代码

### 搜索任务合成
| 方法 | 做法 | 优劣 |
|---|---|---|
| 模板合成 | KG 选 seed → 走 N hop → 模板生成 | hop 可控/答案可达;分布窄 |
| LLM 反向生成 | 给 gold+docs 生成需多跳的问题 | 自然;过滤单跳可答的 |
| **轨迹合成** | 从 gold docs 反推理想搜索路径 | 产生专家级"何时搜/停"示范 |
| 扰动增强 | 改写/换关键词/扩 answer_set | 提鲁棒性 |

**关键**:合成完必做 recall 验证过滤——理想路径必须真能命中检索库。

### 代码任务合成
| 方法 | 做法 | 优劣 |
|---|---|---|
| 真实 PR 挖掘 | SWE-bench 路线(筛选非合成) | 最真实;存活率<10% |
| 种 bug/feature | ACES/SWE-Gym 合成:LLM 造 bug+patch+测试 | 可控难度;真实性差 |
| **轨迹合成** | 强 agent 跑真实 task 记录全过程 | **代码独有:能反复 run_tests 标中间步好坏** |
| 变异对 | 正确 vs buggy 代码对 | 学定位/区分 |

**代码合成杀手锏**:每步 edit 都能立刻 `run_tests` 验证 → 过程标签几乎免费(QA 没有)。

> **速记**:轨迹合成是教控制流的关键;代码轨迹的过程标签靠 run_tests 白送。

---

## 6. 长轨迹信用分配与 PRM 全流程

### 6.1 评估中间轨迹——四条路(从弱到强)
| 方法 | 原理 | 成本 | 何时用 |
|---|---|---|---|
| Outcome 回溯 | 最终对错折扣分给每步 | 极低 | 起步 |
| **规则/环境信号** | 每步从环境拿可验证分(搜索recall/测试增量) | 低 | **首选,无偏不可hack** |
| MC rollout 估计 | 从每步继续 rollout K 次,成功率估 value | 高(K倍) | 自动标 PRM 数据 |
| PRM(学习型) | 训模型给每步打分 | 中 | 精修,有 hack 风险 |

**层次判断**:**优先榨环境信号 → 不够再上 PRM → PRM 数据靠 MC rollout 自动标,不靠人。**

### 6.2 信用分配机制
- **MC return**:`R_t = Σγ^k r_{t+k}`,无偏方差大,长轨迹 γ=0.9~0.95
- **GAE**:value 平滑 bias/variance,但 agentic 长 trajectory 里 value 极难学准
- **GRPO group-relative**:`advantage = (return − 组均值)/组std`,**不用 value,避开 value 不准的坑**(这是 GRPO 流行的真正原因)

**早期步 vs 晚期步不对称**:晚期步信号密;早期步(第一次搜索方向)是致命但难归因。处理:γ 别太小 + 早期步加单独过程奖励 + MC 标注天然放大早期错(V_i 会很低)。

### 6.3 训练 PRM 完整流程

**Step 1:定义"步"粒度(第一坑)**
- 对齐"可验证的环境反馈点":搜索以一次检索为单位,代码以一次 edit/run_tests 为单位
- **token 级标注是浪费**(太细,PRM 难学)

**Step 2:MC rollout 自动标注(Math-Shepherd/ReST-MCTS)**
```python
def label_steps(policy, problem, trajectory, K=8):
    labels = []
    for i, step in enumerate(trajectory.steps):
        prefix = trajectory[:i+1]
        outcomes = []
        for k in range(K):
            completion = policy.rollout(problem, prefix)  # 从 prefix 继续到终止
            outcomes.append(1.0 if is_correct(completion) else 0.0)
        V_i = mean(outcomes)   # 从这步继续的期望成功率 = 该步 value
        labels.append(V_i)
    return labels   # dense 序列 [V_1, ..., V_T]
```
**直觉**:`V_i` 高 = 走到这步大概率能成功 = 好策略;低 = 救不回来 = 有问题。把"哪步好"转化成"从这步开始还能不能赢"。

标签三种形态:回归(MSE 拟合 V_i)/分类(阈值化 good/bad/neutral)/偏好(V_i>V_j 的步对,Bradley-Terry)。

**省成本**:只标答错轨迹 + K 自适应(早期 K=4,后期调大)+ rollout 数据复用(当 SFT/RFT 回灌)。

**Step 3:输入输出**
- 输入:`(problem, trajectory_prefix, current_step)`——**必须带 prefix!**
- 同一 action 脱离上下文无法判断好坏(搜索 X 在缺口时好,找到后冗余)
- 架构:base LM + value head,或生成式打分(`<good>/<bad>` token)

**Step 4:训练目标**
```python
loss = CrossEntropy(PRM(prefix, step), label=good/bad/neutral)  # 分类(稳)
# 或 MSE(PRM(prefix, step), V_i)                                # 回归
```

**Step 5:验证 PRM(OpenAI 标准:Best-of-N)**
用 PRM 从 N 条轨迹挑分最高的,看最终正确率。画"N vs 正确率"曲线,**应在 ORM(outcome reward model)之上**——这是 PRM 价值的证据。

**Step 6:塞进 RL**
```
reward_step = w_o·outcome(终止步) + w_p·PRM(prefix,step) + w_e·env_signal(step)
```

**Step 7:PRM hacking(必处理)**
模型生成"PRM 喜欢但实际坏"的步(代码堆无意义 helper、搜索重复高质量 query 骗命中维度)。**四招防**:
1. **outcome 锚定**:outcome 永不撤,PRM 只是加速器
2. **PRM 在线重训**:policy 进化后用新 policy 重新 MC 标注(避免分布漂移)
3. **对抗喂 hard negative**:骗过 PRM 的轨迹当反例重训 PRM
4. **ensemble 投票**:降单点漏洞

> **速记**:PRM 数据靠 MC rollout 自动标(K 倍采样),输入必带 prefix,用 Best-of-N 验证,outcome 锚定防 hack。能不上 PRM 就不上——环境信号优先。

---

## 7. 控制流训练:何时搜/停/验证

### 三个本质难点(必点名)
1. **Credit assignment**:长程里"早该停却多搜3步"和"该验证却直接答"结果都错,outcome 无法区分
2. **Stop collapse**:模型天然倾向过搜(多搜更"安全"),停止极难训好
3. **Verify 难涌现**:自我验证几乎不会从纯 outcome reward 自然产生(代码任务除外,run_tests 是天然 verify)

### Reward 设计(按动作拆)
- **何时 search**:outcome 驱动 + **冗余惩罚**(重复 query、搜已知信息)
- **何时 stop(最难)**:① 步数惩罚 `−α·steps`(只在答对时给);② MAX_TURNS 截断=负奖励;③ oracle stop 引导(过早停/过晚停都惩罚)。制造"答对+少步 > 答对+多步"梯度
- **何时 verify**:几乎必须显式塑形——**verify-and-correct reward**(触发自检且改对→正奖励);"改对"靠 PRM/外部 verifier 判断,不能靠模型自评

### PRM vs ORM 取舍(加分点)
- ORM:标注便宜,长程信号稀疏,stop/verify 学得慢
- PRM:credit 精准,但标注贵 + PRM 可被 hack
- **实践**:ORM 打底 + 关键步(停止、冗余)用轻量规则做 process shaping,不上重 PRM

### 训练技巧
- **课程学习**:先 1-2 hop 再拉长(长程探索太难)
- **熵正则**:防早期 collapse 到单一策略(停止决策尤其要保熵)
- **GRPO group-relative**:同问题多 rollout 互比,优势信号稳
- **RFT/STaR**:答对轨迹回灌 SFT,比纯 RL 收敛快

### 数据构造(教"何时停")
1. **Rejection sampling**(你面试的思路,工业化):同问题采样 N 条 → 用"结果+过程质量"**可计算地**分好坏 → 好轨迹 SFT(STaR/ReST)、好坏对 DPO
2. **显式停止正负例**:正例(gold chain=k,第 k 步停且答对)、负例A(早停答错)、负例B(过搜浪费)
3. **反模式数据**:重复搜/搜已知/不验证就答 → DPO rejected 侧 / RL 低 reward
4. **课程数据**:短→长 horizon
5. **多样化策略采样**:不同温度/prompt 变体产生各种停止模式

**判定标准必须可算**:不是"我觉得好",而是"答对 ∧ 步数≤oracle×1.5 ∧ 无重复query"。

> **速记**:教停止靠显式停止正负例 + 反模式 + 课程。判定好坏要可计算。DPO 偏好和 RL 过程奖励是互补不是替代。

---

## 8. GRPO vs PPO:长程适配性深度分析

> **你的直觉对**:GRPO 只适合短程、可验证、reward 相对 dense 的场景。长程长轨迹是 GRPO 的灾区。下面是精确化 + 2025-2026 论文证据。

### 8.1 GRPO 的机制与甜区
- **核心**:去掉 critic,用 **group-relative advantage**。同 prompt 采样 G 条 rollout,`advantage = (return − 组均值)/组std`。
- **为什么流行**:agentic/长序列里 **critic(value function)极难训准**,GRPO 干脆不要它,省一半显存。这是它取代 PPO 的真正原因(不是 PPO 不行,是 critic 不行)。
- **甜区**:数学竞赛——**轨迹短**、**reward 可验证且相对 dense**(对/错)、**G 条 rollout 便宜**、**group 内有方差**(有对有错)。

### 8.2 长程下 GRPO 的四个结构性失效

**① Reward Miscalibration(校准失真)**
> [arXiv:2509.23870《Rethinking Reward Miscalibration of GRPO in Agentic RL》](https://arxiv.org/html/2509.23870v1)
group-relative 的 advantage 在多轮 agentic 场景下 reward 校准失真,导致学习信号差。

**② Group 方差崩溃(最致命)**
长程 + 稀疏 reward 下,group 内大量 rollout **全错**(advantage 全 0,无梯度)或**全对**(同样全 0)。GRPO 的信号**依赖 group 内有方差**,而长程稀疏恰恰消灭方差。这就是为什么 DAPO 要加 **Dynamic Sampling**(丢弃全对/全错的 group 重新采样)——是给 GRPO 打补丁。

**③ 多采样成本爆炸**
长轨迹每条 rollout 几千 token,GRPO 要 G 条(4~16)× N prompts,采样成本 = PPO 的 G 倍。长程下这是实打实的算力墙。

**④ Credit assignment 粗糙**
GRPO 把 trajectory-level advantage 摊到所有 token/turn。长程下"早期方向错误"和"晚期执行错误"无法区分——**而长程最需要的恰恰是精细的 per-turn 归因**。

### 8.3 "GRPO is Secretly a PRM"(关键洞察)
> [arXiv:2509.21154《GRPO is Secretly a Process Reward Model》](https://arxiv.org/html/2509.21154v1)

GRPO 的 group-relative 隐式提供了一种**粗糙的过程信号**(同 prompt 不同 rollout 对比,变相在比"哪条路径更好")。但它精度远不如显式 PRM/critic,且**强依赖 group 有方差**。这篇还指出:learned PRM 在 RL 训练中采用有限(印证"能不上 PRM 就不上")。

### 8.4 PPO 在长程的优势
1. **critic 提供 dense baseline**:即使单条 rollout 也能算 advantage,**不依赖 group 有方差**——直接破解 GRPO 的方差崩溃
2. **GAE 的 bias-variance 平衡**:per-step advantage 更精细
3. **配合 PRM/turn-level reward**:能做精细 per-turn credit assignment
4. **实证**: [arXiv:2512.17008 Turn-Level Advantage PPO](https://arxiv.org/html/2512.17008v2) **证明 turn-level PPO 在复杂长程多轮任务上胜过 GRPO**

### 8.5 PPO 的长程新问题 + 解法
- **off-policy drift**:长 episode 里 clipped objective 越来越 off-policy → 不稳定。
  → [arXiv:2511.20718《Stabilizing Off-Policy Training for Long-Horizon LLM Agent》](https://arxiv.org/abs/2511.20718) 的 **turn-level 修正**
- **critic 难训** → 用 **PRM warm-start critic**(绝妙结合:PRM 既是 dense reward 又能初始化 critic)
- **显存翻倍**(actor+critic) → LoRA/共享 backbone 缓解(你项目就是 Qwen2.5 + LoRA)

### 8.6 DAPO = GRPO 在长上下文的打补丁
> [DAPO(ByteDance)](https://openreview.net/forum?id=2a36EMSSTp)

四项改进,**全是治 GRPO 的长上下文病**:
| 技术 | 治什么病 |
|---|---|
| **Dynamic Sampling** | group 全对/全错(方差崩溃) |
| **Overlong Reward Shaping** | 长输出重复/超长 |
| **Token-Level PG Loss** | token 级 credit |
| **Clip-Higher** | 探索不足 |

**注意**:DAPO 仍是 GRPO 系,是"打补丁"而非换框架。长程极致场景下,PPO + turn-level + PRM 更彻底。

### 8.7 选型决策表

| 场景 | 推荐 | 理由 |
|---|---|---|
| 数学竞赛(短轨迹、可验证) | **GRPO / DAPO** | 甜区,critic 没必要 |
| 单轮推理、reward dense | GRPO | 省 critic |
| **长程 agentic、多轮、稀疏 reward** | **PPO + turn-level reward + PRM** | GRPO 方差崩溃 + credit 粗糙 |
| 超长 horizon + 预算紧 | DAPO(GRPO 打补丁) | 比 PPO 省,比 GRPO 稳 |
| 混合 | GRPO 探索打底 → PPO 精修 | 两阶段 |

**核心判断(面试金句)**:**长程任务需要 dense credit assignment,dense credit assignment 需要 value function(critic 或 PRM)。GRPO 的"无 critic"是短程的省事,长程下"无 critic = 无 dense baseline = 信号崩溃"。这就是业界(GLM/Qwen 系)在长程任务上回归 PPO 的根本原因。**

### 8.8 给本项目(2wiki agentic)的建议
你当前用 IGPO(GRPO 变体)。因 2wiki 是多轮 agentic + reward 较稀疏(只有最终答案 EM/F1),属 GRPO 的**边界甚至灾区**。可考虑:
1. **加 dense 过程信号**:每次搜索的 recall / info_gain(项目已有 `info_gain_rewards`)作为 turn-level reward → 缓解 group 方差崩溃
2. **评估 turn-level credit**:若 GRPO 效果停滞,考虑切 PPO + PRM(或 DAPO 的 Dynamic Sampling 先试,改动小)
3. **监控 group 方差**:如果大量 step 的 advantage 全 0,就是方差崩溃的铁证,该上 Dynamic Sampling 或换 PPO

> **速记**:GRPO 甜区=短程可验证;长程灾区=reward miscalibration + group 方差崩溃 + 多采样贵 + credit 粗糙。长程要 dense credit → 需要 value(PRM/critic)→ 这就是 PPO 回归的根本原因。DAPO 是 GRPO 的长上下文补丁。

---

## 9. Prompt 从哪来(in-context vs 训练)

### 先澄清层次(面试官想听的)
- **训练阶段(RL/SFT)**:停止决策靠**奖励信号内化进权重**,prompt 只给起点
- **部署阶段**:模型已学会,prompt 提供任务框架和格式
- **prompt 不是训练信号本身**——说清这层就比多数候选人强

### Prompt 来源
| 来源 | 做法 |
|---|---|
| 人工设计+迭代 | 基于任务理解手写→跑→改(本项目 `tools_server/config.yaml`) |
| 借鉴公开 agent | ReAct / Reflexion / Search-R1 / SWE-agent |
| 从好轨迹提炼 few-shot | 专家轨迹里"何时搜/停"片段摘成 in-context 示例 |
| 自动优化 | DSPy / 自动 prompt engineering |
| 结构化强制决策 | 要求每步输出 `<action>search/stop/verify</action>` |

### "何时搜/停"的 prompt 策略
- 规则注入:"信息够了立即 `<answer>`,不浪费轮次验证"
- 置信度触发:"不确定先搜,有把握直接答"
- 步数先验:"多跳通常需 N 次搜索"

> **金句**:prompt 给先验和格式,真正的停止能力是训练数据喂出来的。**prompt 决定下限,训练数据决定上限。**

---

## 10. 模拟面试 Q&A(9题 + 压轴)

格式:Q → 高分答 → 💡点评(面试官想听什么)。

---

**Q1(开场):"长轨迹最终答错,怎么定位是哪步的问题?"**
> "三层。最粗 outcome 回溯(负 reward 折扣分步,不准)。准一点用环境信号:搜索看每跳 recall,代码看测试增量。最细训 PRM,MC rollout 从每步继续采样估 value。实操**环境信号优先,PRM 补强**。"
💡 别只答"训个 PRM"——那是没考虑成本。给三层+取舍=层次感。

---

**Q2:"MC rollout 标注 K 倍采样成本爆炸,怎么办?"**
> "三招。① 只标答错轨迹(答对的每步默认正);② K 自适应(早期 K=4,后期调大);③ rollout 数据复用当 SFT/RFT 回灌。另外只在关键决策步标,token 级是浪费。"
💡 K 倍采样是 PRM 命门,必问。能答选择性标注+自适应+复用=真做过。

---

**Q3:"PRM 打分必须看 prefix 吗?为什么?"**
> "必须。同 action 不同上下文好坏不同——'搜索X'在缺口时好,找到后冗余;代码'删一行'修 bug 时对、功能正常时破坏。孤立打分会学表面模式被 hack。输入是 (problem,prefix,step) 三元组。"
💡 考 PRM 本质。**杀手答=举具体例子**。

---

**Q4:"GRPO 不用 value,PRM 在 GRPO 里干什么?"**
> "PRM 不是估 value,是直接当**每步 reward source**。GRPO 的 group-relative 算 advantage(不需要 value);PRM 把稀疏 outcome 变 dense per-step reward 塞进 return。两者正交:GRPO 解决'多轨迹怎么比',PRM 解决'单条内每步怎么给信号'。"
💡 高级题。澄清"PRM 是 reward source 不是 value estimator"=懂 RL 底层。

---

**Q5:"PRM 会被 hack 吗?具体怎么 hack,怎么防?"**
> "会。典型:代码堆无意义但语法漂亮的 helper 骗 PRM;搜索重复高质量 query 骗'检索命中'维度。四防:① outcome 锚定永不撤;② PRM 在线重训(policy 进化后重标);③ 骗过 PRM 的轨迹当 hard negative;④ ensemble 投票。"
💡 **举不出具体 hack 样例=没做过**。

---

**Q6:"过程奖励和结果奖励权重怎么调?冲突怎么办?"**
> "三原则。① outcome 是地基 w_o≠0;② w_p 从小调大(初期 PRM 不准);③ 冲突看剪刀差趋势——outcome 涨但 PRM 涨更猛/剪刀差扩大=PRM 被 hack,**降 w_p**。调时看两条曲线剪刀差,不单看一条。"
💡 不给数字给"看剪刀差"=有监控意识。

---

**Q7:"搜索和代码的 PRM 本质不同?"**
> "数据来源不同。搜索 dense 信号=检索 recall/info gain;代码=测试通过增量。**两者都该先吃环境信号,MC 标的 PRM 是补强**。差异在 verify 维度:代码 run_tests 是天然 verify,PRM 直接学;搜索没硬 verify,更依赖 MC 标。"
💡 考迁移。统一框架+指出差异=senior。

---

**Q8:"早期步和晚期步 credit 怎么分?"**
> "不对称。早期步方向性,错了全废但 outcome 难归因(折扣淹没)。① γ 别太小(0.95);② 早期步加单独过程奖励(query 方向质量/定位奖励);③ MC 标注天然缓解——早期 V_i 会很低,放大早期错信号。"
💡 答出"MC 标注天然缓解早期归因"=懂为什么 dense label 强。

---

**Q9(压轴):"算力只够选一个,outcome+环境信号 vs 上 PRM?"**
> "看 horizon 和信号密度。短轨迹+环境信号密(代码有测试增量、搜索有 recall)→ **outcome+环境信号就够,不上 PRM**。长轨迹+环境稀疏(纯推理)→ 才上 PRM,优先 MC 自动标。判断标准:**环境能不能每步给无偏反馈?能就别上 PRM。** PRM 是'环境信号不够'的补丁,不是默认。"
💡 考会不会过度工程。**能不上就不上=克制=senior**。

---

**GRPO vs PPO 专项追问(新增)**

**Q10:"为什么 GRPO 不适合长程?业界为什么回归 PPO?"**
> "GRPO 去掉 critic 用 group-relative,甜区是短程可验证(数学竞赛)。长程有四个结构性失效:① reward miscalibration(arXiv:2509.23870);② group 方差崩溃——稀疏 reward 下 group 内全错/全对,advantage 全 0,DAPO 的 Dynamic Sampling 就在治这个;③ 多采样成本爆炸;④ credit 粗糙。根本原因:**长程需要 dense credit assignment,需要 value(critic/PRM),而 GRPO 的'无 critic'在长程=无 dense baseline=信号崩溃**。turn-level PPO(arXiv:2512.17008)实证胜过 GRPO,这就是 GLM/Qwen 系回归 PPO 的原因。"

**Q11:"但 PPO 的 critic 长序列也难训啊?"**
> "对,这正是当年放弃 PPO 的原因。现在三招缓解:① PRM warm-start critic;② turn-level 优势估计(不依赖 token 级 value 精确);③ off-policy 修正(arXiv:2511.20718 治长 episode 的 off-policy drift)。本质是 PPO 的 critic 问题可工程缓解,而 GRPO 的方差崩溃是结构性的——所以权衡下来长程选 PPO。"

---

**最后三个'压垮型'追问,提前准备**:
1. **"PRM 怎么和 policy 共同进化?"** → on-policy 重训 + 分布漂移检测
2. **"PRM 标注 policy 和训练 policy 不一致(off-policy),偏差怎么办?"** → importance sampling 或定期重标
3. **"怎么证明是 PRM 起作用而非 outcome 本身?"** → Best-of-N 消融 + 去 PRM 对照

---

## 11. 数据集清单

### 长程搜索/多跳 QA
- **2WikiMultihotQA**(本项目)、**HotpotQA**(2-hop)、**MuSiQue**(2~4 hop,强 reasoning chain 标注,**过程诊断首选**)、**Bamboogle**(小而精,适合验证集)

### 长程 agentic 搜索 benchmark
- **GAIA**(多步工具+检索)、**BrowseComp**(DeepMind,极长程网页)、**HLE**(高难)、**MultiHop-RAG**

### Repo-level coding agent
- **SWE-bench** / **SWE-bench Verified**(500,去污染)/ **SWE-bench Lite**(300):评估金标准
- **SWE-Gym**:**可训练版** 2.6k instance,RL 主力数据
- **Multi-SWE-bench / SWE-bench-Java/JS**:多语言
- **ACES**:合成 task 工具

### 单函数/补全(冷启动)
- HumanEval / MBPP / LiveCodeBench(防污染)/ CommitPack(BigCode)
- RepoBench / CrossCodeEval / DevEval / RepoEval

### 获取途径
HuggingFace `datasets.load_dataset(...)`;SWE-bench 官方 `swebench` 包(含 docker 环境 + FAIL_TO_PASS 判定,开箱即用)。

---

## 12. 速记金句 & 电梯陈述

### 速记金句(面试直接用)
1. **验证集**:dev 选 ckpt,test 只跑一次,离线评估靠 ckpt。
2. **Hacking 检测**:剪刀差 = 带格式分 vs 不带格式分;完全脱钩靠 MBE/人工/退化指标。
3. **信用分配优先级**:环境信号(搜索 recall/测试增量)> PRM > outcome+MC。
4. **PRM 数据**:MC rollout 自动标(K 倍采样),输入必带 prefix,Best-of-N 验证,outcome 锚定防 hack。
5. **PRM 克制原则**:能不上就不上,环境能给无偏反馈就别上 PRM。
6. **DPO vs RL**:偏好学相对排序,RL 做细粒度 credit,**互补不替代**。
7. **Prompt**:给先验和格式,停止能力是训练数据喂出来的。prompt 定下限,数据定上限。
8. **GRPO vs PPO**:长程需要 dense credit → 需要 value → 这就是 PPO 回归的根本原因。GRPO 甜区=短程可验证。
9. **代码任务**:QA 花力气造 reward,代码花力气筛数据。FAIL_TO_PASS 是命脉。
10. **停止决策**:显式构造停止正负例 + 反模式 + 课程学习,判定好坏要可计算。

### 30秒电梯陈述(整场面试开场)
> "agentic RL 长程任务的核心是 dense credit assignment。数据上,任务数据走模板合成(搜索)/真实 PR 挖掘(代码),必须过可验证性筛选;轨迹数据用 rejection sampling 按'结果+过程质量'可计算地分好坏,好轨迹 SFT 回灌、好坏对 DPO、再进 RL。
>
> 信用分配我优先榨环境信号(搜索 recall/info gain,代码测试增量)——免费、无偏、不可 hack;不够才上 PRM,数据靠 MC rollout 自动标而非人工。
>
> 教'何时停'靠显式停止正负例 + 反模式 + 课程学习。**算法上,短程可验证用 GRPO,长程 agentic 因为 group 方差崩溃和 credit 粗糙,应该上 PPO + turn-level reward + PRM**——这是业界回归 PPO 的根本原因。
>
> 整个过程我优先做可验证、可监控、克制的工程,不为炫技上重模型。"

---

## 13. 附录:关键论文与资源

### GRPO vs PPO / 长程(2025-2026 核心)
- [Rethinking Reward Miscalibration of GRPO in Agentic RL (arXiv:2509.23870)](https://arxiv.org/html/2509.23870v1) — GRPO 多轮 reward 校准失真
- [Turn-Level Advantage PPO (arXiv:2512.17008)](https://arxiv.org/html/2512.17008v2) — turn-level PPO 胜过 GRPO
- [Stabilizing Off-Policy Training for Long-Horizon LLM Agent (arXiv:2511.20718)](https://arxiv.org/abs/2511.20718) — 长 episode off-policy 修正
- [GRPO is Secretly a Process Reward Model (arXiv:2509.21154)](https://arxiv.org/html/2509.21154v1) — GRPO 隐式 PRM 洞察
- [DAPO (ByteDance, OpenReview)](https://openreview.net/forum?id=2a36EMSSTp) — GRPO 长上下文四项改进
- [Comparative Analysis: PPO/GRPO/DAPO (arXiv:2512.07611)](https://arxiv.org/html/2512.07611v1) — 三算法系统对比
- [Demystifying RL in Agentic Reasoning (arXiv:2510.11701)](https://arxiv.org/html/2510.11701v1)
- [Verlog: Multi-turn RL Framework (CMU)](https://blog.ml.cmu.edu/2025/09/15/verlog-a-multi-turn-rl-framework-for-llm-agents/)
- [Best Practices for Multi-Turn RL (Fireworks AI)](https://fireworks.ai/blog/best-practices-for-multi-turn-RL)
- [The State of RL for LLM Reasoning (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [GRPO++ Tricks (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [Guide to LLM Post-Training Algorithms (HuggingFace)](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms)

### PRM / Credit Assignment
- OpenAI "Let's Verify Step by Step"(人工 PRM 标注先驱)
- Math-Shepherd / ReST-MCTS(MC rollout 自动标注)

### 项目相关
- SWE-bench / SWE-Gym / Search-R1 / R1-Searcher / ReAct / Reflexion

---

*文档生成于 2026-06-22,基于 agentic-rl 项目(2wiki agentic search)实战 + 面试方法论整理。GRPO/PPO 章节含 2025-2026 最新论文证据。*

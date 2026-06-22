# IGPO 本地复现项目 — 代码阅读指南

> 本文档帮助新人快速读懂整个项目流程。按训练数据流顺序阅读，每个文件标注了「是什么 / 为什么改 / 关键函数」。

---

## 0. 先建立全局认知

**IGPO 做什么**：用 turn-level 信息增益（info gain）作为内在奖励，教 LLM Agent 学会"多轮搜索时每一轮该搜什么、何时停止"。

**本项目相对原版 IGPO 的核心改动**：
1. **模型**：Qwen2.5-7B-Instruct（全参微调）→ **Qwen3-4B-Thinking-2507（LoRA）**
2. **搜索**：在线 Google/Bing API → **本地 tantivy BM25 服务（port 8890）**
3. **数据**：80K 三源混合 → **2Wiki 单源，BM25 过滤后 12.7K**
4. **新增 SFT warmup** 阶段（原版没有）
5. **Qwen3 chat template 适配**（debug 重点，见第 6 节）

---

## 1. 推荐阅读顺序（按数据流）

```
① 入口脚本
   train_1gpu_mock.sh        ← 从这里开始，看配置参数
        │
② 主调度
   verl/trainer/main_ppo.py  ← hydra 配置加载 + reward 函数装配
        │
③ 训练主循环
   verl/trainer/ppo/ray_trainer.py  ← 最长(1431行)，PPO 训练循环
        │
④ Agent 多轮生成（IGPO 核心！）
   scrl/llm_agent/generation.py  ← 最重要(1012行)，多轮搜索 loop
        │
⑤ Reward 计算
   verl/utils/reward_score/info_gain.py  ← F1 + info_gain 奖励
   verl/workers/reward_manager/naive.py  ← reward 装配/调度
        │
⑥ Advantage 计算
   verl/trainer/ppo/core_algos.py  ← GRPO 优势函数 + turn-level 累积
        │
⑦ 搜索服务端
   tools_server/config.yaml / tools.py / search_api.py  ← 本地 BM25
```

**阅读策略**：先读 ①② 理解"怎么启动"，再读 ④⑤⑥ 理解"算法核心"，最后读 ③ 理解"训练循环怎么串起来"。`ray_trainer.py` 最长最绕，建议最后读。

---

## 2. 各文件详解

### ① `train_1gpu_mock.sh`（入口）
- **是什么**：训练启动脚本，设置所有超参数，调用 `verl.trainer.main_ppo`
- **关键环境变量**：`MODE`（2wiki/grpo/mock）、`ROLLOUT_N`、`MAX_TURNS`、`IGPO_SEARCH_ENGINE=local`
- **注意**：`MODE=grpo` 时 `MAX_TURNS=1`（退化为单轮，只用 F1 reward，作为消融对照）

### ② `verl/trainer/main_ppo.py`（主调度，207行）
- **关键函数**：
  - `main(config)`（L79）：hydra 配置入口，把搜索配置转发给 Ray workers（L89 注释）
  - `TaskRunner`（L109）：在线程里跑训练，处理 reward 函数装配
- **为什么改**：L89 需要把本地搜索配置（engine=local、port=8890）转发给分布式 worker

### ③ `verl/trainer/ppo/ray_trainer.py`（训练主循环，1431行）
- **是什么**：veRL 框架的 PPO 训练器，IGPO 在此基础上接入 agent loop
- **关键流程**：rollout 采样 → 算 reward → 算 advantage → PPO 更新
- **注意**：这是框架代码，大部分逻辑是 veRL 原生的，只需关注 IGPO 改动的部分

### ④ `scrl/llm_agent/generation.py`（Agent Loop，1012行）⭐ 最核心
- **是什么**：多轮搜索生成循环。这是 IGPO 区别于普通 GRPO 的关键
- **核心循环**（L414）：
  ```python
  for step in range(self.config.max_turns):  # 最多 5 轮
      # 1. 算 pseudo ground truth log_prob（用于 info gain）
      # 2. 模型生成 response
      # 3. 解析 response（<answer> 结束 / <tool_call> 继续搜）
      # 4. 执行搜索 → 结果追加到 messages → 下一轮
  ```
- **关键函数**：
  - `parse_response`（L200）：解析模型输出，判断是给答案还是调工具
  - `execute_predictions`（L140）：执行 web_search，调本地 BM25 服务
  - `_compute_turn_level_advantage`（在 core_algos.py）：info gain 累积
- **⚠️ Qwen3 debug 点**（L436-439）：见第 6 节，最重要的一处改动

### ⑤ `verl/utils/reward_score/info_gain.py`（Reward，298行）
- **关键函数**：
  - `check_tags_balance`（L8）：检查 `<think></think>`/`<answer></answer>` 是否闭合（格式校验）
  - `compute_f1`（L77）：token 级 F1（最终答案的外在奖励）
  - `compute_score`（L165）：核心！把 F1 放在最后 token，info_gain 放在每轮 token
- **reward 结构**：中间轮 = info_gain（log_prob_diff），最后一轮 = F1

### ⑤' `verl/workers/reward_manager/naive.py`（252行）
- **是什么**：reward 调度器，把 info_gain 的 turn reward 喂给 reward 函数
- **为什么改**：原版只处理单次 outcome reward，IGPO 需要处理多轮 turn-level reward

### ⑥ `verl/trainer/ppo/core_algos.py`（Advantage，668行）
- **核心函数**：`compute_grpo_outcome_advantage`（L189）
- **流程**：组内归一化（F1 和 IG 分开）→ turn-level 累积（γ=1.0 不折扣）→ 广播到每轮所有 token
- **辅助**：`_compute_turn_level_advantage`（L28）实现 turn-level 折扣累加

### ⑦ `tools_server/`（搜索服务）
- `config.yaml`：搜索引擎配置（local/tantivy）+ system prompt（改成 BM25 关键词风格）
- `tools.py`：web_search 工具定义（参数是 query 数组）
- `search_api.py`：HTTP 服务，POST /search → 调 tantivy 返回 top-k

---

## 3. 训练一次完整的数据流

```
1. 读取 train parquet (prompt + ground_truth)
   ↓
2. Repeat n=8 次 → 8 条独立 rollout
   ↓
3. 每条 rollout 跑 agent loop（generation.py）：
   for turn in range(5):
     ├─ 用 ref 模型算 P(ground_truth | 当前上下文)  ← log_prob_old
     ├─ actor 模型生成 response
     ├─ 解析：<answer> 结束 / <tool_call> 搜索
     └─ 搜索结果追加 → 算 P(ground_truth | 新上下文)  ← log_prob_new
        info_gain[turn] = log_prob_new - log_prob_old
   ↓
4. reward（info_gain.py）：中间轮放 IG，最后轮放 F1
   ↓
5. advantage（core_algos.py）：8 条 rollout 组内归一化 + turn 累积
   ↓
6. PPO loss 更新 LoRA 参数 + KL 约束
```

---

## 4. SFT 相关文件（warmup 阶段）

- `run_sft_warmup.sh`：SFT 入口脚本
- `verl/utils/dataset/sft_dataset.py`：SFT 数据加载（修过 2 个 bug，见下）
- `data/sft_warmup.parquet`：81 条 SFT 数据（已清洗）

### sft_dataset.py 的两个 bug 修复
- **Bug 1**（L41 附近）：`self.prompt_key` 被包成 list，导致 DataFrame 而非 Series，`.tolist()` 崩溃
- **Bug 2**（L119 附近）：对已含 `<|im_start|>` 的预格式化 prompt 又套一次 `apply_chat_template`，双重渲染。加了 `preformatted` 检测跳过

---

## 5. 数据构建链（面试常问）

```
2WikiMultihopQA 原始 167K（含 context Wikipedia 段落）
   ├─→ 提取唯一 Wikipedia passage → passages.jsonl (391K) → tantivy 索引
   └─→ 筛 compositional 类型 → train_2wiki.parquet (24.9K)
                ↓ BM25 可回答性过滤（director 通过率 80%，family 25%，是/否 0%）
         train_2wiki_filtered.parquet (12.7K)
                ↓ 分层均衡采样
         train_2wiki_1k_balanced.parquet (863)
```

---

## 6. ⚠️ Qwen3 适配的 debug 点（最值得讲）

原版为 Qwen2.5 设计。换 Qwen3-4B-Thinking 后遇到的问题：

### 问题 1：think 标签双重注入（generation.py L436-439）
- **现象**：31/32 的 rollout 格式错误，`check_tags_balance` 失败
- **原因**：Qwen3-Thinking 的 chat template 在 `add_generation_prompt=True` 时**已经自动加了 `<think>` 开头**。原版 Qwen2.5 代码手动 append 了一个，导致 `<think>...<think>` 双开标签
- **修复**：删掉手动 append（L439 注释保留了旧代码 `# Old: rollings_active = [rolling + "..." ...]`）

### 问题 2：tool 结果序列化（generation.py L670 附近）
- **原因**：Qwen3 chat template 要求 message content 是 string，原版直接传 list/dict 会崩
- **修复**：`json.dumps` 序列化成字符串（注释 `# Serialize search results to string for Qwen3 chat template compatibility`）

### 问题 3：SFT 双重 chat template（见第 4 节 Bug 2）

---

## 7. 面试可重点讲的"我做了什么"

1. **本地化检索引擎**：tantivy BM25 替代在线搜索，全流程离线
2. **数据工程**：BM25 过滤 + 发现类型失衡（80% director）+ 均衡采样
3. **Qwen3 适配 debug**：think 标签双重注入、tool 序列化、chat template 兼容
4. **双路并行检索改进方案**（BM25 + 向量 RRF，简历亮点）
5. **结果**：Baseline 21 → GRPO 26.4 → IGPO 34.1（+62.4%）

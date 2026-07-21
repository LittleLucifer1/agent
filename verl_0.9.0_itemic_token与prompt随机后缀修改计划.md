# VeRL 0.9.0：Itemic Token 与 Prompt 随机后缀修改计划

## 1. 目标与版本说明

本计划用于在 VeRL 0.9.0 代码基础上实现以下两项能力：

1. 在每个训练 prompt 后随机添加 ` /think`、` /no_think` 或空串。
2. 在 OPD rollout 中对 teacher 无法识别的 itemic token 做特殊处理。

目前 PyPI 最新正式版仍是 `0.8.0`，GitHub `main` 分支的版本文件标记为 `0.9.0.dev`，并非正式发布的 `0.9.0`。因此远端修改前必须固定具体 commit。下面涉及的代码入口以当前 `0.9.0.dev` 为参考。

参考：

- [VeRL 版本文件](https://github.com/verl-project/verl/blob/main/verl/version/version)
- [VeRL PyPI 发布记录](https://pypi.org/project/verl/)
- [VeRL OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)

远端开始修改前记录：

```bash
git rev-parse HEAD
cat verl/version/version
git status --short
```

建议基于该 commit 新建独立开发分支，并将 commit SHA 写入实验记录。

## 2. 目标数据流

```text
原始 messages
  → 为最后一个 user prompt 分配 /think、/no_think 或空串
  → 学生高温 rollout
  → 检测 response 中第一个 itemic token
  → teacher 只计算有效前缀
  → itemic 位置注入极低 teacher logprob
  → itemic 之后关闭 distillation mask
  → reverse-KL token advantage
  → advantage clip
  → actor 更新
```

核心原则：

- itemic token 本身必须保留在学生 response 中，才能对它施加负梯度。
- itemic token 不能原样发送给 teacher，否则可能越过 teacher vocabulary。
- itemic token 之后的 token 不再参与 OPD。
- 不能直接丢掉包含 itemic token 的整条 trajectory，否则学生无法学会抑制该 token，还可能引入采样偏差。
- `response_mask` 和 `distill_response_mask` 必须分开，不能破坏 VeRL 原有 rollout、tool 或 PPO 语义。
- 优先复用 VeRL 0.9 原生 teacher loop 和 distillation loss，不移植 OpenOneRec 旧 fork 的完整 trainer。

## 3. 总体修改计划表

| 阶段 | 文件或模块 | 计划修改 | 完成标准 |
|---|---|---|---|
| 0. 固定版本 | 仓库根目录 | 记录 `git rev-parse HEAD`、`verl/version/version`；新建独立分支 | 后续实验都能对应到唯一 commit |
| 1. 建立基线 | VeRL 自带 OPD 示例 | 不做修改，先跑一个 8～32 prompt 的原生 OPD smoke test | teacher logprob、actor update、checkpoint 正常 |
| 2. 增加配置 | `verl/workers/config/distillation.py`、`verl/trainer/config/distillation/distillation.yaml` | 增加 `itemic_tokens` 和非对称 advantage clip 配置；prompt suffix 配置放在 `data` 下 | Hydra 能正确解析；关闭开关时行为和原版一致 |
| 3. Prompt 随机后缀 | 新建 `recipe/onpolicy_distill/onerec_opd_dataset.py` | 继承 `RLHFDataset`，覆盖 `__getitem__`，深拷贝 `raw_prompt` 后修改最后一个 user message | 每条训练 prompt 恰好获得一种后缀；不修改原始 dataset 对象 |
| 4. Tokenizer 审计 | 新建 `itemic_policy.py` 或启动检查函数 | 获取 student/teacher vocabulary，确定 itemic ID 范围及 teacher-safe token | teacher 收到的所有 token ID 都小于 teacher vocab size |
| 5. Rollout 检测 | `verl/experimental/agent_loop/agent_loop.py` | 在 `_compute_teacher_logprobs()` 前检测第一个 itemic token，构造 teacher-safe sequence 和 distillation mask | itemic 位置信息能随 batch 传入训练阶段 |
| 6. Teacher 计分 | `agent_loop.py`；必要时小改 `teacher_manager.py` | teacher 只计算到首个 itemic token 位置；用安全 token 占位；返回后覆盖该位置 teacher logprob | teacher 不报越界；itemic 位置获得预期的极低 teacher logprob |
| 7. 独立损失掩码 | `agent_loop.py`、`verl/trainer/distillation/losses.py` | 新增 `distill_response_mask`；distillation loss 优先使用它，其他 PPO/task reward 仍用原 `response_mask` | itemic 后 distillation loss 为零，原 rollout mask 不受影响 |
| 8. Advantage clip | `verl/trainer/distillation/losses.py` | PG OPD 中对 `advantages = -distillation_losses` 做非对称裁剪 | itemic token advantage 被裁到下界，没有 Inf/NaN |
| 9. 指标和日志 | agent loop、distillation metrics | 统计 suffix 分布、itemic response 比例、首次出现位置、截断比例 | 可从日志确认随机后缀和 itemic 抑制是否生效 |
| 10. 测试 | 新增 dataset、itemic、OPD 集成测试 | 单测、强制 itemic rollout、bf16 smoke test | 全部通过后才能扩大训练规模 |

VeRL 0.9 已经具有原生 teacher loop、`teacher_logprobs` 和 OPD loss，主要参考入口：

- [Agent loop](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)
- [Teacher manager](https://github.com/verl-project/verl/blob/main/verl/experimental/teacher_loop/teacher_manager.py)
- [Distillation losses](https://github.com/verl-project/verl/blob/main/verl/trainer/distillation/losses.py)
- [RLHFDataset](https://github.com/verl-project/verl/blob/main/verl/utils/dataset/rl_dataset.py)

## 4. Prompt 随机添加方案

### 4.1 随机选项

OpenOneRec 的公开实现中，`auto` 模式通过 `random.randint(0, 2)` 等概率选择：

- 空串
- `" /think"`
- `" /no_think"`

参考：[OpenOneRec OneRecDataset](https://github.com/Kuaishou-OneRec/OpenOneRec/blob/main/verl_distillation/verl/utils/dataset/onerec_dataset.py)

建议配置：

```yaml
data:
  custom_cls:
    path: recipe/onpolicy_distill/onerec_opd_dataset.py
    name: OneRecOPDDataset

  prompt_suffix:
    enabled: true
    choices: ["", " /think", " /no_think"]
    weights: [0.333333, 0.333333, 0.333334]
    seed: 42
    mode: stable_per_sample
    target: last_user
    validation_choice: ""
```

论文没有给出三种选择的明确概率，因此第一版可采用等概率，并保留 `weights` 配置以便后续实验调整。

### 4.2 Dataset 实现位置

建议新建：

```text
recipe/onpolicy_distill/onerec_opd_dataset.py
```

在其中实现：

```text
OneRecOPDDataset(RLHFDataset)
```

通过 VeRL 的 `data.custom_cls.path` 和 `data.custom_cls.name` 加载，尽量不直接修改核心 `rl_dataset.py`。

### 4.3 `__getitem__` 行为

实现要求：

1. 先调用父类 `__getitem__()`。
2. 对返回的 messages 使用 `copy.deepcopy()`。
3. 从后向前查找最后一个 `role == "user"` 的 message。
4. 只修改最后一个 user message。
5. 修改发生在 chat template 和 tokenization 之前。
6. 检查内容是否已经以 `/think` 或 `/no_think` 结尾，防止重复添加。
7. 将本次选择记录为 `prompt_suffix_mode`，供日志统计。
8. 验证集默认使用空后缀，不做随机化。

对于多模态或结构化 content：

- 如果 `content` 是字符串，直接追加后缀。
- 如果 `content` 是 segment list，则追加到最后一个 text segment。
- 不得改变 image、video segment 的顺序和内容。

### 4.4 随机性的可复现性

不要直接在 `__getitem__()` 中使用全局 `random.choice()`。多 worker、Ray 重试或 checkpoint resume 后，分配结果可能漂移。

推荐使用稳定映射：

```text
choice = categorical_hash(seed, sample_index)
```

这样在同一个 seed 下，同一个样本始终获得相同后缀。

如果确实要求每个 epoch 重新随机，可升级为：

```text
choice = categorical_hash(seed, epoch, sample_index)
```

但这要求 dataloader 在 checkpoint resume 后正确恢复 epoch，因此不建议第一版采用。

## 5. Itemic Token 的识别与启动检查

### 5.1 不直接写死 `151669`

OpenOneRec 当前示例把 `token_id >= 151669` 当作扩展 token，但这个值只适合其对应 tokenizer。

参考：[OpenOneRec OPD README](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/main/verl_distillation)

远端启动时应检查并记录：

```text
student_vocab_size
teacher_vocab_size
first_itemic_token_id
itemic_token_ranges
teacher_safe_token_id
```

优先使用明确的 itemic token ID 列表或区间。只有确认学生所有尾部扩展 token 都是 itemic token 时，才使用：

```text
token_id >= extend_vocab_start_token
```

### 5.2 Vocabulary 对齐检查

训练启动前必须验证：

- student 和 teacher 在共享 vocabulary 区域中，同一个 ID 对应同一个 token。
- 所有普通 prompt token 都能被 teacher 识别。
- `teacher_safe_token_id < teacher_vocab_size`。
- student vocabulary 确实包含额外 itemic token。
- 所有非 itemic 的 response token 都在 teacher vocabulary 内。

如果共享 ID 的 token 含义不同，应立即终止训练。这种情况无法通过简单替换或 mask 修复，因为 teacher/student 的逐 token logprob 已经无法对齐。

### 5.3 推荐配置结构

```yaml
distillation:
  itemic_tokens:
    enabled: true
    detection: explicit_ranges
    ranges: []
    start_id: null
    policy: truncate_after_first
    teacher_safe_token_id: null
    teacher_logprob_floor: -100.0
    keep_first_itemic_token_in_loss: true
```

建议支持以下策略，但默认只使用第一种：

| 策略 | 行为 | 建议 |
|---|---|---|
| `truncate_after_first` | 保留并惩罚首个 itemic token，其后全部关闭 distillation loss | 论文一致，推荐 |
| `mask_invalid_only` | 只 mask itemic token，其他位置继续训练 | 可作为工程对照实验 |
| `drop_response` | 丢弃或 mask 整条 response | 不推荐，可能引入采样偏差 |

## 6. Itemic Token 的论文一致处理算法

假设学生 response 为 `r[0:L]`，先只在模型生成位置检测 itemic token：

```text
itemic_mask[i] = response_mask[i] == 1 and is_itemic(r[i])
t = first_true(itemic_mask)
```

只在 `response_mask[i] == 1` 的位置检测，避免把 tool observation 中的 token 当成学生生成的 itemic token。

### 6.1 没有 itemic token

完全沿用 VeRL 原生 teacher scoring：

```text
teacher_query = prompt_ids + response_ids
distill_response_mask = response_mask
```

关闭新功能时，该路径必须与原生 VeRL 行为完全一致。

### 6.2 存在 itemic token

如果第一个 itemic token 位于 response 的位置 `t`，构造：

```text
teacher_query =
    prompt_ids
    + response_ids[:t]
    + [teacher_safe_token_id]
```

处理步骤：

1. teacher 只计算这个前缀。
2. safe token 只是占位，用来避免 vocabulary 越界。
3. 由于 safe token 前面的上下文仍然是原始有效前缀，teacher 在该位置使用的 state 是正确的。
4. teacher 返回后，将全序列位置 `prompt_length + t` 的 teacher logprob 覆盖为 `teacher_logprob_floor`。
5. teacher 输出右侧补齐到原始 response 长度。
6. `distill_response_mask[0:t+1]` 保留。
7. `distill_response_mask[t+1:] = 0`。
8. 学生原始 `response_ids[t]` 仍然保留为 itemic token，绝对不能替换。

最终效果：

```text
位置 0 ... t-1：正常 teacher/student reverse-KL
位置 t：itemic token 获得强负 advantage
位置 t+1 ... L-1：不参与 distillation loss
```

### 6.3 为什么不能替换学生 response 中的 itemic token

学生 actor 的 logprob 必须针对真实采样到的 itemic token 计算。如果把学生训练输入中的 itemic token 也替换为 safe token：

- actor 计算的是 safe token 的 logprob；
- itemic token 自身不会收到负梯度；
- 模型无法学习抑制 itemic token。

所以只能修改发送给 teacher 的副本，不能修改原始 student rollout。

## 7. Teacher Logprob Floor 与 Advantage Clip

论文描述可以使用类似 `-1e9` 的极小 teacher logprob，但在 bf16/fp16 和后续 KL 运算中可能产生 Inf/NaN。

建议配置：

```yaml
teacher_logprob_floor: -100.0
advantage_clip_min: -30.0
advantage_clip_max: 5.0
```

OpenOneRec 公开实现目前也使用 `[-30, 5]` 的 distillation advantage clip。

参考：[OpenOneRec 实现说明](https://github.com/Kuaishou-OneRec/OpenOneRec/tree/main/verl_distillation)

对于 reverse-KL PG OPD：

```text
distillation_loss_t = student_logprob_t - teacher_logprob_t
advantage_t = -distillation_loss_t
            = teacher_logprob_t - student_logprob_t
```

itemic token 的 teacher logprob 被设为 `-100` 后，会产生很强的负 advantage，再裁剪到下界：

```text
advantage_t = clamp(
    teacher_logprob_t - student_logprob_t,
    min=-30,
    max=5
)
```

当前 VeRL 的 `loss_max_clamp` 是对 loss 做对称裁剪，不完全等价于 `[-30, 5]` 的非对称 advantage clip。因此建议在 `DistillationLossConfig` 中增加：

```text
advantage_clip_min
advantage_clip_max
```

并在 PG OPD 分支生成 `advantages = -distillation_losses.detach()` 后再裁剪。启用新字段时，不要同时设置 `loss_max_clamp`，避免双重裁剪。

## 8. 独立的 Distillation Mask

不要直接覆盖原有 `response_mask`。建议新增以下字段：

```text
distill_response_mask
itemic_token_mask
has_itemic_token
first_itemic_position
```

最终蒸馏 mask：

```text
effective_distill_mask
    = response_mask
    × itemic_policy_mask
    × valid_response_attention_mask
```

其中：

- `response_mask`：VeRL 原有模型生成/tool/padding 语义。
- `itemic_policy_mask`：首个 itemic token 后为零。
- `valid_response_attention_mask`：排除 padding。

在 distillation loss 中采用逻辑：

```text
response_mask = data.get(
    "distill_response_mask",
    data["response_mask"],
)
```

该 mask 只用于 distillation loss 和相关指标：

- PPO/task reward 继续使用原始 `response_mask`。
- 关闭 itemic 功能时，没有 `distill_response_mask`，自动回退到原始行为。
- 如果 `use_task_rewards=false`，训练只受 distillation mask 影响。

## 9. 推荐 OPD 配置

论文式实现应使用 sampled-token reverse KL，而不是 `forward_kl_topk`：

```yaml
distillation:
  enabled: true

  itemic_tokens:
    enabled: true
    detection: explicit_ranges
    ranges: []
    start_id: null
    policy: truncate_after_first
    teacher_safe_token_id: null
    teacher_logprob_floor: -100.0
    keep_first_itemic_token_in_loss: true

  distillation_loss:
    loss_mode: k1
    use_policy_gradient: true
    use_task_rewards: false
    advantage_clip_min: -30.0
    advantage_clip_max: 5.0
    loss_max_clamp: null

actor_rollout_ref:
  actor:
    use_kl_loss: false

  rollout:
    temperature: 1.1
    top_p: 0.95
    top_k: 200

algorithm:
  use_kl_in_reward: false
```

注意：

- 高 rollout temperature 用于主动探索 itemic token，使模型有机会学习抑制它们。
- teacher 只做 prompt logprob forward，teacher inference temperature 应保持 `1.0`。
- 如果是纯 OPD，建议 `use_task_rewards=false`。
- 如果需要 OPD 与任务奖励混合，应单独验证 task reward 是否也需要截断；第一版不要改变 task reward mask。

### 9.1 Loss mode 保护

当以下配置同时成立时：

```text
itemic_tokens.policy = truncate_after_first
keep_first_itemic_token_in_loss = true
```

建议只允许单样本 KL estimator，例如：

```text
loss_mode = k1
```

如果误配成 `forward_kl_topk`，建议启动时直接报错。原因是 top-k teacher distribution 和“给采样到的 itemic token 设置 teacher logprob floor”不是同一种算法。

## 10. 具体字段传递方案

### 10.1 Agent loop 输出

在 `AgentLoopOutput.extra_fields` 或内部输出结构中保存：

```text
teacher_ids
teacher_logprobs
distill_response_mask
itemic_token_mask
has_itemic_token
first_itemic_position
```

在 `_postprocess()` 中将 tensor 字段 padding 并写入 `DataProto.batch`。

统计信息可保存在 `non_tensor_batch` 或 metrics 中，但训练 loss 使用的 mask 必须是 tensor。

### 10.2 Teacher 输出补齐

如果 teacher 只计算到 itemic 位置，应在返回后将：

```text
teacher_ids
teacher_logprobs
```

补齐到：

```text
prompt_length + original_response_length
```

补齐区域的 logprob 可使用 `0.0`，因为对应的 `distill_response_mask` 已经为零；不要使用 NaN 或 Inf。

### 10.3 多 teacher 场景

如果 VeRL 配置了多个 teacher，每个 teacher 可能具有不同 vocabulary。此时：

- itemic 检测可以基于 student tokenizer 统一完成。
- `teacher_safe_token_id` 和 `teacher_vocab_size` 必须按 routing key 分别配置或检查。
- 不得假设不同 teacher 的 `eos_token_id` 相同。

第一版如无多 teacher 需求，可以只支持单 teacher，并在检测到多个 teacher 时明确报错。

## 11. 测试计划

### 11.1 Prompt suffix 单元测试

| 测试场景 | 预期结果 |
|---|---|
| 单轮 user prompt | 正确添加一种后缀 |
| 多轮 messages | 只修改最后一个 user message |
| 已有 `/think` | 不重复添加 |
| 已有 `/no_think` | 不重复添加 |
| 同 seed、同 index | 分配结果完全一致 |
| 不同 seed | 分配结果发生变化 |
| 多 dataloader worker | 分配结果与单 worker 一致 |
| 原始 dataset 对象 | 内容未被原地修改 |
| 验证集 | 默认使用空后缀 |
| 多模态 content | image/video segment 不被破坏 |

在足够多样本上统计三类比例，应接近配置权重，但不要求小 batch 精确等于三分之一。

### 11.2 Itemic token 单元测试

| 测试场景 | 预期结果 |
|---|---|
| response 没有 itemic token | teacher 输入、mask、loss 与原版一致 |
| 第一个 response token 就是 itemic | 第一个 token 保留并受罚，后面全部 `distill_response_mask=0` |
| 中间出现 itemic | 前缀正常训练，itemic 位置受罚，后缀不训练 |
| 多个 itemic token | 第一个位置决定截断点 |
| itemic 位于 padding 边界 | 不污染 padding 区域 |
| EOS、PAD 等特殊 token | 不被误判成 itemic token |
| tool observation 含特殊 token | 只检测 `response_mask==1` 的模型生成 token |
| teacher 请求 | 最大 token ID 始终小于 teacher vocab size |
| teacher 输出补齐 | shape 与完整 student sequence 对齐 |
| bf16/fp16 | teacher floor 和 KL 计算无 Inf/NaN |

### 11.3 Loss 测试

需要直接构造一条已知 logprob 的样本，验证：

```text
student_logprob = -2
teacher_logprob_floor = -100
raw_advantage = -98
clipped_advantage = -30
```

同时验证：

- itemic token 之前 loss 正常。
- itemic token 本身参与 loss。
- itemic token 之后 loss 为零。
- `response_mask` 保持原值。
- `distill_response_mask` 只影响 distillation loss。

### 11.4 集成 smoke test

准备 8～32 条 prompt：

- 一部分正常 rollout。
- 一部分通过 mock 或强制 logits 让模型生成已知 itemic token。
- `rollout.n=1`。
- 只训练 1～2 step。

检查：

- teacher server 不出现 token ID 越界。
- itemic 位置 teacher logprob 被覆盖。
- itemic 位置 advantage 达到负裁剪下界。
- itemic 后 distillation mask 为零。
- actor backward 和 optimizer step 正常。
- checkpoint 可以保存并恢复。

## 12. 指标与日志

建议至少记录：

```text
prompt_suffix/think_count
prompt_suffix/no_think_count
prompt_suffix/empty_count
prompt_suffix/think_ratio
prompt_suffix/no_think_ratio
prompt_suffix/empty_ratio

itemic/response_rate
itemic/token_rate
itemic/first_position_mean
itemic/truncated_token_ratio
itemic/teacher_floor_count
itemic/teacher_oov_request_count

distillation/advantage_min
distillation/advantage_max
distillation/masked_token_ratio
distillation/nan_count
```

建议额外抽样打印少量 debug 样本：

```text
sample index
prompt suffix
first itemic position
original response token ids
teacher query token ids
distill response mask
itemic position student/teacher logprob
itemic position clipped advantage
```

debug 样本必须限量，避免把完整训练数据写入日志。

## 13. 验收条件

全部满足以下条件后，才进入正式大规模训练：

- suffix 三类比例接近配置值。
- 同 seed 重启后 suffix 分配完全一致。
- `itemic/teacher_oov_request_count == 0`。
- itemic token 位置 advantage 达到预期的负裁剪下界。
- itemic token 后面的 distillation loss 为零。
- 学生原始 itemic token 没有被 safe token 替换。
- 原始 `response_mask` 未被破坏。
- 无 NaN/Inf。
- 关闭 prompt suffix 和 itemic 功能后，结果与原生 VeRL OPD 基线一致。
- 在中等规模训练中，`itemic/response_rate` 随训练呈下降趋势。

## 14. 推荐实施顺序

1. 固定 VeRL commit，跑通未修改的原生 OPD。
2. 只实现 prompt suffix dataset，并完成确定性测试。
3. 完成 tokenizer/vocabulary 启动审计。
4. 单独实现 itemic 检测和 mask helper，并完成纯 CPU 单测。
5. 接入 agent loop 的 teacher 请求边界。
6. 接入 `distill_response_mask`。
7. 增加非对称 advantage clip。
8. 增加指标和 debug 日志。
9. 使用强制 itemic token 的小规模集成测试。
10. 进行 100～1000 prompt 的短训练，观察 itemic rate 和 suffix 分布。
11. 确认稳定后再启动正式训练。

## 15. 最终建议的文件范围

预计新增：

```text
recipe/onpolicy_distill/onerec_opd_dataset.py
verl/experimental/teacher_loop/itemic_policy.py
tests/utils/dataset/test_onerec_opd_dataset.py
tests/experimental/teacher_loop/test_itemic_policy.py
tests/special_e2e/run_itemic_opd_smoke.sh
```

预计小范围修改：

```text
verl/workers/config/distillation.py
verl/trainer/config/distillation/distillation.yaml
verl/experimental/agent_loop/agent_loop.py
verl/experimental/teacher_loop/teacher_manager.py  # 仅在需要扩展参数时修改
verl/trainer/distillation/losses.py
```

不建议直接复制或替换：

```text
OpenOneRec 旧版 RayOnPolicyDistillTrainer
OpenOneRec 旧版完整 ray_trainer.py
OpenOneRec 旧版完整 core_algos.py
```

OpenOneRec 的 OPD fork 基于较早的 VeRL commit，而 VeRL 0.9 已经具备原生 OPD 数据流。正确做法是保留 VeRL 0.9 原生 teacher loop 和 distillation loss，只在 dataset、teacher 请求边界及 distillation mask 三处增加扩展。


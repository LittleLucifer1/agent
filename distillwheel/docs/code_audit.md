# DistillWheel 代码审计报告

## 结论

本轮以仓库代码为准、设计 PDF 为背景资料，审查并修复了核心编排层、ms-swift 适配器和 VERL 适配器。2026-07-14 对用户新增修改再次做了整体复审；控制层测试在无 GPU 的本机完成，共 `192 passed`，Ruff 静态检查通过，wheel 构建和命令行入口也已验证。真实 CUDA、显存和后端 ABI 只能在目标 GPU 机器上通过一轮训练 step 验证，操作见[后端冒烟测试](backend_smoke_tests.md)。

## 已修复的主要问题

| 范围 | 原风险 | 当前处理 |
|---|---|---|
| 后端隔离 | 解释器路径依赖当前目录，父 `PYTHONPATH` 可能让健康检查误通过 | 支持绝对 Python 覆盖和统一环境根目录；健康检查、环境创建和 merger 使用 Python isolated mode；训练进程清理污染变量 |
| 进程管理 | 超时或异常时可能只终止父进程并留下 worker | 创建独立进程组，超时、消费中止和关闭时终止整个进程树；日志解码容错并保留尾部 |
| 输出安全 | 重用输出目录可能覆盖、误删用户文件或混入旧权重 | 每次运行独占目录并验证所有权标记/锁；覆盖仅清理已知管理产物；checkpoint 归一化拒绝非空 `final/` |
| IR 校验 | batch 语义、非有限数值、stage/样本不匹配和消息字段可能拖到后端才失败 | 统一 `global = micro × grad_accum × dp`；拒绝 `NaN/Inf`、未知字段和非法工具角色；统一包装 YAML/JSONL 编码错误 |
| 指标可靠性 | 后端 extra 中的 `NaN`、bytes、set 或循环对象可能写出非法 JSON、甚至中断训练 | 指标递归转换为标准 JSON-safe 数据，非有限值写为 `null`；日志异常时确定性关闭 launcher |
| ms-swift 命令 | 旧版参数、Agent 消息和 DPO 数据格式与 4.4 不一致 | 锁定 `>=4.4,<4.5`；使用位置 YAML、`swift rlhf`；将 OpenAI tool call 转成 `tool_call/tool_response` 并保留结构化 rejected continuation |
| ms-swift 分布式 | 继承父 `torchrun` 的 rank 或 rendezvous 变量可能加入错误进程组 | 单卡不注入 launcher 变量；清理 rank/world/master 变量；多卡由 `parallel.dp` 决定；多节点、TP、PP 明确拒绝 |
| ms-swift 产物 | 多个数字版本、分片模型和祖先目录资产可能选错或漏归一化 | 按数字版本选择最新有效 run，校验分片 index，并从 checkpoint/祖先复制 adapter、model、tokenizer 和 processor 资产 |
| VERL 配置 | Hydra 键、rollout 模式和 parquet schema 与 0.8 不一致 | 锁定 `verl[vllm]==0.8.0`；使用 0.8 仅支持的 `rollout.mode=async`；映射 GRPO/PPO/RLOO、FSDP dtype 和嵌套 Arrow prompt schema |
| VERL 日志 | `timing_s/step` 可能被误当成 global step | 只从明确的训练步数字段取 step，避免指标时间轴被耗时值污染 |
| VERL 奖励函数 | 相对路径在 Ray worker 中不可解析 | 配置生成时解析为绝对 `file.py:function`，同时支持合法 dotted module |
| VERL checkpoint | 使用过时 merger 或把 LoRA 当作完整模型 | 使用 0.8 `verl.model_merger`；LoRA 产物写入 base-model 依赖说明，不隐式下载资产 |
| 能力边界 | OPD、多模态、QLoRA 等可能被静默错误映射 | 仅声明已实现的 SFT/DPO/KTO、GRPO/PPO/RLOO；无法忠实表达的组合在启动前给出明确错误 |
| 安装与打包 | setup 健康检查过弱，环境 YAML/工具可能未进 wheel | 严格检查 Python 和后端版本/关键导入；环境清单、setup 工具和两个命令行入口进入 wheel |

## 建议的后续改进

这些不是本轮正确性修复的前置条件，可以按优先级逐项决定。

| 编号 | 建议 | 收益 | 工作量 | 建议优先级 |
|---|---|---|---|---|
| I-01 | 建立 GPU 契约 CI：固定卡型上 nightly 跑 Swift SFT/DPO 与 VERL GRPO 各 1 step | 第一时间发现后端升级、CUDA 或 vLLM ABI 回归 | 中 | P0 |
| I-02 | 发布经验证的后端容器/锁文件和 CUDA–PyTorch–vLLM 硬件矩阵 | 显著降低“pip 成功但运行失败”的环境问题 | 高 | P0 |
| I-03 | 将环境清单拆成 Swift 基础/QLoRA/DeepSpeed、VERL vLLM/SGLang 等能力 profile | 减少不需要的依赖冲突和安装体积 | 中 | P1 |
| I-04 | 扩展 Resource/Sequence IR：节点数、每节点 GPU、共享存储、prompt/response 长度 | 为多节点和更精确的 VERL 资源映射打基础 | 高 | P1 |
| I-05 | 增加独立 train/validation/eval 数据流 | 支持生产评估、早停和 VERL 验证奖励 | 中 | P1 |
| I-06 | 把 preflight 升级为后端原生配置解析：Swift 参数加载、VERL `--cfg job` | 在占用 GPU 前发现未知参数和 Hydra 键漂移 | 中 | P1 |
| I-07 | 为 OPD 增加 DistillationConfig（teacher、资源、loss、off-policy 数据）后再启用 | 让蒸馏算法可验证、可复现，而非依赖隐式 meta | 高 | P1 |
| I-08 | 为 VERL 实现多模态和 tool schema 的完整映射 | 将当前“明确拒绝”升级为真实能力 | 高 | P1 |
| I-09 | 使用结构化事件/指标通道替代日志正则，并记录后端完整版本清单 | 提高监控、故障定位和跨版本稳定性 | 中 | P2 |
| I-10 | 增加 run attempt、断点续训 manifest 和可审计的 stale-lock 恢复 | 改善抢占式集群和长任务恢复体验 | 中 | P2 |

建议先执行 I-01 与 I-02；这两项对减少异机环境 bug 的收益最大。之后再根据是否要支持多节点、OPD 或多模态选择对应 P1 项。

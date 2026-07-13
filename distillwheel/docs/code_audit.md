# DistillWheel 代码审计报告

## 结论

本轮以仓库代码为准、设计 PDF 为背景资料，审查并修复了核心编排层、ms-swift 适配器和 VERL 适配器。控制层测试在无 GPU 的本机完成，共 `133 passed`；wheel 构建和干净虚拟环境中的命令行入口也已验证。真实 CUDA、显存和后端 ABI 只能在目标 GPU 机器上通过一轮训练 step 验证，操作见[后端冒烟测试](backend_smoke_tests.md)。

## 已修复的主要问题

| 范围 | 原风险 | 当前处理 |
|---|---|---|
| 后端隔离 | 解释器路径依赖当前目录，环境变量可能污染子进程 | 支持绝对 Python 覆盖和统一环境根目录；清理分布式继承变量及 `PYTHONPATH`；保留必要 CUDA/HF 环境变量 |
| 进程管理 | 超时或异常时可能只终止父进程并留下 worker | 创建独立进程组，超时、消费中止和关闭时终止整个进程树；日志解码容错并保留尾部 |
| 输出安全 | 重用输出目录可能覆盖或误删用户文件 | 每次运行独占目录并写所有权标记/锁；默认拒绝非空目录；覆盖仅清理已知的框架管理产物 |
| IR 校验 | batch 语义、stage/样本不匹配和消息字段可能拖到后端才失败 | 统一 `global = micro × grad_accum × dp`；校验阶段、数值、最后一条 assistant、KTO label、工具调用和 JSONL 行号 |
| ms-swift 命令 | 旧版参数和 DPO 数据格式与 4.x 不一致 | 锁定 `>=4.4,<4.5`；使用位置 YAML、`swift rlhf`、标准 chosen/rejected 格式和 4.x LoRA/QLoRA 参数 |
| ms-swift 分布式 | 单卡继承 `NPROC_PER_NODE`，多节点行为不确定 | 单卡不注入 launcher 变量；多卡由 `parallel.dp` 决定；当前多节点、TP、PP 在启动前明确拒绝 |
| ms-swift 产物 | version 子目录、分片模型和 processor 资产可能漏归一化 | 递归发现 checkpoint，校验分片 index，并复制 adapter/model、tokenizer 和 processor 资产 |
| VERL 配置 | Hydra 键、算法选择和 parquet schema 与目标版本不一致 | 锁定 `verl[vllm]==0.8.0`；映射 GRPO/PPO/RLOO 的 0.8 配置键和 FSDP dtype；输出嵌套 Arrow prompt schema |
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

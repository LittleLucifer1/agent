# DistillWheel 后端冒烟测试

本文档针对当前代码锁定的后端契约：

- ms-swift：`>=4.4,<4.5`，支持 SFT、DPO、KTO。
- VERL：`==0.8.0`，支持 GRPO、PPO、RLOO；该版本已删除同步 rollout，适配器固定使用 `rollout.mode=async`。
- OPD 暂不宣称支持；当前 Recipe 还不能表达 teacher model、teacher resources 和蒸馏 loss。

ms-swift 4.x 的 YAML 是位置参数（`swift sft config.yaml`），DPO 使用标准
`messages + rejected_response` 数据格式。参考
[ms-swift 4.4 命令参数](https://github.com/modelscope/ms-swift/blob/v4.4.0/docs/source_en/Instruction/Command-line-parameters.md)
和[自定义数据集格式](https://github.com/modelscope/ms-swift/blob/v4.4.0/docs/source_en/Customization/Custom-dataset.md)。
VERL 的 CUDA、PyTorch、vLLM 组合非常敏感，生产测试优先使用
[VERL 官方安装方案](https://verl.readthedocs.io/en/latest/start/install.html)。
[VERL 0.8 RolloutConfig](https://github.com/verl-project/verl/blob/v0.8.0/verl/workers/config/rollout.py)
会直接拒绝旧的 `sync` 值，因此不要通过 `meta.verl.rollout_mode` 或自定义 override 改回同步模式。

以下命令均从仓库根目录执行。不要从 `examples/smoke` 子目录启动，因为奖励函数的相对路径在配置生成时按当前目录解析为绝对路径。

## 1. 安装控制环境

```bash
cd /path/to/distillwheel

python3.10 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[verl]'

export DISTILLWHEEL_ENV_ROOT="$PWD/.venvs"
distillwheel list-backends
```

预期最后两行包含 `swift` 和 `verl`。控制环境只负责 Recipe、数据转换、日志和产物归一化；训练框架运行在单独解释器中。

## 2. ms-swift：SFT 与 DPO

### 2.1 创建后端环境

完整托管环境：

```bash
distillwheel-setup-backends swift --python "$(command -v python)"

.venvs/swift/bin/python -c \
  "import importlib.metadata as m, torch, trl; print('ms-swift', m.version('ms-swift')); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('trl', trl.__version__)"
```

如果机器已经有按官方 CUDA/PyTorch 矩阵配置好的 Conda 环境，建议直接复用，并使用绝对路径：

```bash
export DISTILLWHEEL_SWIFT_PYTHON=/absolute/path/to/swift-env/bin/python
distillwheel-setup-backends swift
```

此时 setup 命令只做 Python 版本和健康检查，不会修改外部环境。只有明确加
`--install-into-external` 才会向该环境安装包。

若 GPU 不支持 BF16（例如部分旧卡），先把两个 Swift Recipe 中的
`precision: bf16` 改成 `precision: fp16`。

### 2.2 路由检查与真实训练

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

distillwheel show-route --recipe examples/smoke/swift_sft.yaml
distillwheel show-route --recipe examples/smoke/swift_dpo.yaml

distillwheel run \
  --recipe examples/smoke/swift_sft.yaml \
  --data examples/smoke/swift_sft.jsonl

distillwheel run \
  --recipe examples/smoke/swift_dpo.yaml \
  --data examples/smoke/swift_dpo.jsonl
```

两个 Recipe 都通过 `meta.swift.overrides.max_steps: 1` 限制为一个优化 step。
第二次使用同一输出目录时，改 `io.output_dir`，或对确认属于 DistillWheel 的旧目录加
`--overwrite-output`。该开关只清理带所有权标记的框架管理文件。

### 2.3 验收

```bash
python -c "import json; p='outputs/smoke-swift-sft'; print(json.load(open(p+'/.distillwheel-run.json'))); print(json.load(open(p+'/metadata.json')))"
python -c "import json; p='outputs/smoke-swift-dpo'; print(json.load(open(p+'/.distillwheel-run.json'))); print(json.load(open(p+'/metadata.json')))"

test -s outputs/smoke-swift-sft/metrics.jsonl
test -s outputs/smoke-swift-dpo/metrics.jsonl
test -s outputs/smoke-swift-sft/final/adapter_model.safetensors
test -s outputs/smoke-swift-dpo/final/adapter_model.safetensors
```

`status` 应为 `succeeded`，metadata 的 framework 应为 `swift`，且 final 中应有 LoRA adapter、adapter config、Recipe 和 tokenizer/processor 资产。若 ms-swift 保存 `.bin` 而不是 safetensors，将最后两个文件名改为 `adapter_model.bin` 检查。

两卡 DDP 时同时修改 Recipe：`parallel.dp: 2`，并保持
`global_batch = micro_batch * grad_accum * dp`；例如 micro=1、grad_accum=1、global=2。
然后使用 `CUDA_VISIBLE_DEVICES=0,1`。当前 IR 明确只支持单节点；设置 `NNODES>1` 会在启动前报错。

## 3. VERL：GRPO（主测试）以及 PPO/RLOO

### 3.1 准备 VERL 环境

首选官方 Docker/镜像内已经验证过的 VERL 0.8.0 环境，然后让控制进程指向其中的绝对 Python：

```bash
export DISTILLWHEEL_VERL_PYTHON=/absolute/path/to/verl-env/bin/python
distillwheel-setup-backends verl

"$DISTILLWHEEL_VERL_PYTHON" -c \
  "import importlib.metadata as m, torch, ray, vllm, pyarrow; print('verl', m.version('verl')); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('ray', ray.__version__, 'vllm', vllm.__version__, 'pyarrow', pyarrow.__version__)"
```

若只是临时单机试验，也可以让 DistillWheel 创建环境：

```bash
unset DISTILLWHEEL_VERL_PYTHON
distillwheel-setup-backends verl --python "$(command -v python)"
```

该方式安装 `verl[vllm]==0.8.0`，但 pip 成功并不等于 CUDA ABI 一定匹配；导入检查和下方真实 step 都必须通过。

### 3.2 真实 GRPO step

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

distillwheel show-route --recipe examples/smoke/verl_grpo.yaml

distillwheel run \
  --recipe examples/smoke/verl_grpo.yaml \
  --data examples/smoke/verl_grpo.jsonl
```

示例有 2 个 prompt、train batch=2、rollout n=2，因此一个 epoch 只有一个训练 iteration。奖励函数来自 `examples/smoke/reward.py:compute_score`；框架会把文件路径转换为绝对路径再传给 Ray worker。

PPO 和 RLOO 的映射也有独立契约测试；若要在目标机器做真实验证：

```bash
distillwheel run \
  --recipe examples/smoke/verl_ppo.yaml \
  --data examples/smoke/verl_grpo.jsonl

distillwheel run \
  --recipe examples/smoke/verl_rloo.yaml \
  --data examples/smoke/verl_grpo.jsonl
```

PPO 会额外创建 critic，显存需求高于 GRPO/RLOO。当前 VERL 适配器不支持 QLoRA、FP8、pipeline parallel、DeepSpeed zero_stage 或多节点；VERL 0.8 的 FSDP LoRA 路径也不会读取本框架 IR 中的非零 LoRA dropout，因此 `peft.dropout` 必须为 `0`。这些组合会在启动前报出明确错误，避免静默生成与 Recipe 不一致的训练配置。

当前 VERL 数据适配器只接受纯文本 prompt。含 `images`、`tools`、`tool_calls` 或 `tool_call_id` 的样本会被明确拒绝，而不会静默丢字段。ms-swift 适配器仍会保留这些字段。

### 3.3 验收与 Hydra 配置诊断

```bash
python -c "import json; p='outputs/smoke-verl-grpo'; print(json.load(open(p+'/.distillwheel-run.json'))); print(json.load(open(p+'/metadata.json')))"
test -s outputs/smoke-verl-grpo/metrics.jsonl
find outputs/smoke-verl-grpo/final -maxdepth 1 -type f -print
```

预期 status=`succeeded`、framework=`verl`，并存在 `global_step_1/actor` 的归一化产物。FSDP shard 会通过 VERL 0.8 的官方 merger 命令合并。LoRA 产物仍需要 metadata 中记录的 base model；框架不会在 checkpoint 归一化阶段隐式联网下载 tokenizer。

若训练在 Hydra 组合阶段失败，可用已生成的 overrides 做“只组合配置”检查：

```bash
mapfile -t VERL_OVERRIDES < outputs/smoke-verl-grpo/workdir/verl_overrides.txt
"$DISTILLWHEEL_VERL_PYTHON" -m verl.trainer.main_ppo \
  --cfg job "${VERL_OVERRIDES[@]}" > /tmp/distillwheel-verl-config.yaml
```

这一步应无 unknown-key / missing-key 错误；它不会替代真实 GPU step。

## 4. 失败时先收集这些信息

```bash
nvidia-smi
python --version
distillwheel --version

tail -n 200 outputs/<run>/raw_logs/stdout.log
cat outputs/<run>/workdir/swift_config.yaml
cat outputs/<run>/workdir/verl_overrides.txt
cat outputs/<run>/.distillwheel-run.json
```

Swift 只会生成 `swift_config.yaml`，VERL 只会生成 `verl_overrides.txt`，因此不存在的那个文件可以忽略。提交问题时同时提供 Recipe、上述版本输出和完整 traceback；不要只提供最后一行 CUDA 错误。

## 5. 官方契约参考

- [ms-swift 4.4 README](https://github.com/modelscope/ms-swift/blob/v4.4.0/README.md)
- [ms-swift 4.4 命令参数](https://github.com/modelscope/ms-swift/blob/v4.4.0/docs/source_en/Instruction/Command-line-parameters.md)
- [ms-swift 4.4 自定义数据集](https://github.com/modelscope/ms-swift/blob/v4.4.0/docs/source_en/Customization/Custom-dataset.md)
- [VERL 0.8 RolloutConfig](https://github.com/verl-project/verl/blob/v0.8.0/verl/workers/config/rollout.py)
- [VERL 安装](https://verl.readthedocs.io/en/latest/start/install.html)
- [VERL 数据准备](https://verl.readthedocs.io/en/v0.4.x/preparation/prepare_data.html)
- [VERL 配置](https://verl.readthedocs.io/en/latest/examples/config.html)
- [VERL checkpoint 合并](https://verl.readthedocs.io/en/latest/advance/checkpoint.html)

# vlm-planning

本仓库用于在 RTX 5090 服务器上运行 EmbodiedBench 的 EB-ALFRED（`eb-alf`）和
EB-Habitat（`eb-hab`）。三个顶层分支用途不同：`master` 保存可复现的基础运行配置，
`transformers-profile` 保存延时、KV 和 attention profiling 实验，`kv-reuse` 从
`master` 独立开发跨调用 KV 复用。后两者都是研究分支，不要把未验证改动直接合入 baseline。

模型与测评分开运行：

| 推理方式 | 模型服务环境 | 示例模型 | 测评环境 |
| --- | --- | --- | --- |
| LMDeploy | `lmdeploy` | Qwen3-VL-8B-Instruct（FP16） | `embench` |
| Transformers | `vlm-profile` | Qwen3-VL-8B-Instruct（BF16 / FP8） | `embench` |

以下说明面向第一次克隆仓库、尚未配置环境和数据的用户。所有 `/YOUR/...` 路径都要替换成自己的绝对路径。

## 1. 克隆仓库

```bash
git clone --branch master --recurse-submodules https://github.com/lucy0217/vlm-planning.git
cd vlm-planning
git submodule status --recursive
```

仓库会同时检出固定版本的 `third_party/EmbodiedBench`、`third_party/transformers`，
以及 EmbodiedBench 内的 `habitat-lab`。子模块显示 detached HEAD 是正常现象；不要执行
`git submodule update --remote`。

如果要复现本文第 8 节的 KV 实验，切换顶层分支后必须同步它记录的子模块提交：

```bash
git switch kv-reuse
git submodule update --init --recursive
git branch --show-current
git submodule status --recursive
```

仅运行实验时，Transformers 子模块处于 detached HEAD 是正常的；只有需要修改模型侧 KV
代码时，才在子模块仓库中切换或创建开发分支。

## 2. 配置环境

需要 Conda、Git LFS、可用的 NVIDIA 驱动。三个环境相互独立，下面命令均在仓库根目录即/vlm-planning执行。

注：环境文件针对 Linux x86_64：`.conda-explicit.txt` 固定 Python 和 conda 原生库，
`requirements.txt` 固定 pip 包。Habitat-Sim 和 OpenGL 依赖不能只靠 pip 安装，因此两类文件都需要。

### 2.1 EmbodiedBench 测评环境

```bash
conda create -n embench --file environments/embench.conda-explicit.txt -y
conda activate embench
python -m pip install -r environments/embench/requirements.txt
python -m pip install --no-deps --no-build-isolation \
  -e ./third_party/EmbodiedBench \
  -e ./third_party/EmbodiedBench/habitat-lab/habitat-lab
python -m pip check
```

该环境包含 AI2-THOR 2.1.0、Habitat-Sim/Lab 0.3.0，以及运行测评所需的旧版依赖。

### 2.2 LMDeploy 推理环境

本机配置使用 Python 3.11、LMDeploy 0.13.0、PyTorch 2.10.0 和 CUDA 12.8，
替代 EmbodiedBench 原始的旧 LMDeploy 环境，以支持 RTX 5090。

```bash
conda create -n lmdeploy --file environments/lmdeploy.conda-explicit.txt -y
conda activate lmdeploy
python -m pip install -r environments/lmdeploy/requirements.txt
python -m pip check
```

### 2.3 Transformers 推理环境

```bash
conda create -n vlm-profile --file environments/vlm-profile.conda-explicit.txt -y
conda activate vlm-profile
python -m pip install --no-deps --no-build-isolation -e ./third_party/transformers
python -m pip install -r environments/vlm-profile/requirements.txt
python -m pip check
python -c 'import transformers; print(transformers.__file__)'
```

最后一条命令应指向当前仓库中的
`third_party/transformers/src/transformers/__init__.py`。

## 3. 准备测评数据

```bash
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
git lfs install --local
git lfs pull
```

### EB-ALFRED

```bash
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED \
  embodiedbench/envs/eb_alfred/data/json_2.1.0
git -C embodiedbench/envs/eb_alfred/data/json_2.1.0 lfs pull
```

AI2-THOR 首次启动时还会下载 Unity 仿真程序到当前用户的缓存目录。

### EB-Habitat

```bash
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets \
  --data-path embodiedbench/envs/eb_habitat/data
```

完成后，`embodiedbench/envs/eb_habitat/data/` 下应包含 `replica_cad`、`objects` 等资源。

## 4. 配置 EB-ALFRED 显示服务

EB-ALFRED 需要 GPU Xorg，默认使用 `X_DISPLAY=:1`。用户需要具有 sudo 权限，
并在单独的 tmux 会话中运行：

```bash
tmux new -s xorg
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
sudo "$(which python)" -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

该命令会占用当前窗口，测评期间保持运行。按 `Ctrl-b`，松开后按 `d` 离开 tmux。
如果 `:1` 已经运行，不要重复启动。EB-Habitat 使用 headless Habitat-Sim，不需要 Xorg。

## 5. 下载 Qwen3-VL 模型

模型不保存在 Git 仓库中。下面的 `local_dir` 可以换成自己的目录或可读的共享目录。

### 5.1 原始模型（LMDeploy FP16 / Transformers BF16）
local_dir修改为自己的保存路径。

```bash
conda activate lmdeploy
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir="/YOUR/MODELS/Qwen3-VL-8B-Instruct")'
```

### 5.2 FP8 模型（Transformers）

```bash
conda activate vlm-profile
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-VL-8B-Instruct-FP8", local_dir="/YOUR/MODELS/Qwen3-VL-8B-Instruct-FP8")'
python -c 'from huggingface_hub import snapshot_download; snapshot_download("kernels-community/finegrained-fp8", revision="0e5764aa5c27d0f005117a92645156481540950a")'
```

FP8 服务依赖 `kernels==0.17.0` 和上面固定 revision 的完整 kernel snapshot。

## 6. 使用 LMDeploy 运行 Qwen3-VL FP16 测评

LMDeploy 0.13.0 的 Qwen3-VL 使用 PyTorch backend。模型服务和测评分别放在两个 tmux 会话中（因为测评运行时间较长，推荐放在tmux中避免模型服务中断）。

### 6.1 启动模型服务
需要将/YOUR/MODELS/Qwen3-VL-8B-Instruct \修改成自己保存模型的路径。
```bash
tmux new -s lmdeploy-qwen3-server
conda activate lmdeploy
cd /YOUR/PATH/vlm-planning
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
lmdeploy serve api_server /YOUR/MODELS/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --backend pytorch \
  --dtype float16 \
  --tp 1 \
  --session-len 12000 \
  --cache-max-entry-count 0.4 \
  --max-batch-size 1 \
  --server-name 127.0.0.1 \
  --server-port 23333
```

等待模型加载完成后，在新终端检查图片推理：

```bash
conda activate lmdeploy
cd /YOUR/PATH/vlm-planning
curl -fsS http://127.0.0.1:23333/v1/models
python scripts/smoke/vlm_image.py \
  --base-url http://127.0.0.1:23333/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct
```

### 6.2 运行测评脚本

```bash
tmux new -s lmdeploy-qwen3-eval
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
export DISPLAY=:1
export remote_url=http://127.0.0.1:23333/v1
export OPENAI_API_KEY=EMPTY
```

（1）运行 EB-ALFRED 的完整 `base` 子集：运行结果会保存在third_party/EmbodiedBench/running文件夹下。

```bash
python -m embodiedbench.main \
  env=eb-alf \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct \
  exp_name=lmdeploy_qwen3vl_fp16 \
  eval_sets='[base]' \
  down_sample_ratio=1.0
```

（2）运行 EB-Habitat 的完整 `base` 子集：

```bash
python -m embodiedbench.main \
  env=eb-hab \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct \
  exp_name=lmdeploy_qwen3vl_fp16 \
  eval_sets='[base]' \
  down_sample_ratio=1.0
```

`eval_sets` 可以按需选择一个或多个子集；命令中的 `'[base]'` 只运行 `base`。
例如，`eval_sets='[base,common_sense]'` 依次运行两个子集，`eval_sets='[]'` 运行全部子集。
请保留列表外层的引号。EB-ALFRED 支持 `base`、`common_sense`、
`complex_instruction`、`spatial`、`visual_appearance`、`long_horizon`；
EB-Habitat 支持 `base`、`common_sense`、`complex_instruction`、
`spatial_relationship`、`visual_appearance`、`long_horizon`。这一配置方式也适用于下文的
Transformers 测评命令。



## 7. 使用 Transformers 本地服务运行 Qwen3-VL 测评

### 7.1 BF16

#### 启动 BF16 模型服务

```bash
tmux new -s transformers-bf16-server
conda activate vlm-profile
cd /YOUR/PATH/vlm-planning
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /YOUR/MODELS/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --port 8001
```

等待模型加载完成后，在新终端检查服务：

```bash
conda activate vlm-profile
cd /YOUR/PATH/vlm-planning
curl -fsS http://127.0.0.1:8001/health
python scripts/smoke/vlm_image.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct
```

#### 运行 BF16 测评脚本

```bash
tmux new -s transformers-bf16-eval
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
export DISPLAY=:1
export remote_url=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY
```

```bash
# EB-ALFRED base
python -m embodiedbench.main \
  env=eb-alf \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct \
  exp_name=transformers_qwen3vl_bf16 \
  eval_sets='[base]' \
  down_sample_ratio=1.0

# EB-Habitat base
python -m embodiedbench.main \
  env=eb-hab \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct \
  exp_name=transformers_qwen3vl_bf16 \
  eval_sets='[base]' \
  down_sample_ratio=1.0
```

### 7.2 FP8

FP8 checkpoint 的权重使用 fine-grained FP8，服务使用 BF16 激活。

#### 启动 FP8 模型服务

```bash
tmux new -s transformers-fp8-server
conda activate vlm-profile
cd /YOUR/PATH/vlm-planning
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /YOUR/MODELS/Qwen3-VL-8B-Instruct-FP8 \
  --model-name Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --dtype bfloat16 \
  --port 8002
```

等待模型加载完成后，在新终端检查服务：

```bash
conda activate vlm-profile
cd /YOUR/PATH/vlm-planning
curl -fsS http://127.0.0.1:8002/health
python scripts/smoke/vlm_image.py \
  --base-url http://127.0.0.1:8002/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct-FP8
```

#### 运行 FP8 测评脚本

```bash
tmux new -s transformers-fp8-eval
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
export DISPLAY=:1
export remote_url=http://127.0.0.1:8002/v1
export OPENAI_API_KEY=EMPTY
```

```bash
# EB-ALFRED base
python -m embodiedbench.main \
  env=eb-alf \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct-FP8 \
  exp_name=transformers_qwen3vl_fp8 \
  eval_sets='[base]' \
  down_sample_ratio=1.0

# EB-Habitat base
python -m embodiedbench.main \
  env=eb-hab \
  model_type=remote \
  model_name=Qwen/Qwen3-VL-8B-Instruct-FP8 \
  exp_name=transformers_qwen3vl_fp8 \
  eval_sets='[base]' \
  down_sample_ratio=1.0
```

结果位于 `third_party/EmbodiedBench/running/eb_alfred/` 和
`third_party/EmbodiedBench/running/eb_habitat/`。每轮实验使用新的 `exp_name`。

tmux 中按 `Ctrl-b`，松开后按 `d` 可退出而不停止任务。开始前用 `nvidia-smi`
选择空闲 GPU，并避免占用其他用户的端口。其它 EmbodiedBench 环境(manipulation, navigation,这里没有使用)见
[third_party/EmbodiedBench/README.md](third_party/EmbodiedBench/README.md)。

## 8. `kv-reuse` 分支：跨调用 KV 复用与消融

本节给出运行方法；模块职责、P/A/B/C/S 的精确定义、缓存生命周期和完整调用链参见
[inference/README.md](inference/README.md)。

该实验使用三处相互隔离的状态：

| 对象 | 名称 | 内容 |
| --- | --- | --- |
| 顶层仓库分支 | `kv-reuse` | 服务、prompt 分段、测评包装器和统计 |
| Transformers 子模块记录版本 | `kv-reuse` 开发分支上的提交 | Qwen3-VL 分段 prefill 与相对 mRoPE 修正 |
| Conda 环境 | `vlm-profile` | 模型服务依赖，不保存源码 |

EmbodiedBench 子模块保持固定提交，不需要创建实验分支。服务支持四个互斥模式：

| 模式 | 首次调用后保存 | 后续完整计算 | 后续复用 |
| --- | --- | --- | --- |
| `off` | 无 | P/A/B/C/S | 无 |
| `a` | A 每层 post-RoPE K/V | P/B/C/S | A K/V |
| `ac` | A/C 每层 post-RoPE K/V | P/B/S | A K/V；C V 与位置修正后的 K |
| `a-layerwise` | A 每层 post-RoPE K/V | 第1层全部；后续层的未选 A、P/B/C/S | 后续层所选 A 子段 K/V |

`a` 是 A-only 消融，当前 C 会重新执行投影、Attention 和 MLP。`ac` 中 C 的 K 使用旧、新
三维 mRoPE 的相对旋转修正，C 的 V 和位置不变的 A K/V 直接复用。

这里的序列划分为 `P + A + B + C + S`：P 包含当前图片和 chat template 前缀；A 包含固定
规则、动作表、few-shot 和当前任务指令；B 是动态动作历史与环境反馈；C 是 JSON 输出格式说明；
S 是 assistant generation prompt。A 位于当前图片之后，所以复用 A 会引入首张图片对
A hidden state/KV 的影响导致精度下降。

### 8.1 启动 AC 复用服务

```bash
tmux new -s qwen3-kv-server
conda activate vlm-profile
cd /YOUR/PATH/vlm-planning

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /YOUR/MODELS/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --port 8001 \
  --kv-reuse-mode ac
```

调试边界或 cache mismatch 时可添加 `--kv-reuse-strict`；正式长测评不添加，校验失败的
请求会完整 prefill 并在 JSONL 中记录原因，但不会替换 episode 首次调用建立的 canonical
cache。A-only 消融把模式改为 `a`，baseline 改为 `off`。
各模式及不同 `a-layerwise` 子段配置必须分别重启服务。

分层 A 子段实验例如 A1+A5：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /YOUR/MODELS/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --port 8001 \
  --kv-reuse-mode a-layerwise \
  --kv-reuse-a-parts 1,5 \
  --kv-recompute-prefix-layers 1
```

### 8.2 运行带 episode 上下文的测评

所有 KV 实验必须使用包装器运行，而不是直接执行 `embodiedbench.main`，否则 HTTP 请求没有
episode ID，服务会安全回退。`off` 也应使用同一包装器，才能获得相同 prompt 规范化、episode
选择和性能日志。包装器不会修改 EmbodiedBench 源码；它会统一 EmbodiedBench 首次和后续
prompt 中 few-shot 标题及 action-history 前的空白差异，使实验定义的 A 段真正 token 一致。
`off`、`a` 和 `ac` 都执行相同规范化。

```bash
tmux new -s qwen3-kv-eval
conda activate embench
cd /YOUR/PATH/vlm-planning/third_party/EmbodiedBench
export DISPLAY=:1
export OPENAI_API_KEY=EMPTY

python /YOUR/PATH/vlm-planning/scripts/kv_reuse/run_embodiedbench.py \
  --benchmark eb-alf \
  --base-url http://127.0.0.1:8001/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --kv-mode ac \
  --eval-set base \
  --episodes 1 \
  --output-dir /YOUR/PATH/vlm-planning/inference/kv_reuse/runs/eb_alf_ac \
  --save-prompts
```

完整消融时分别启动 `off`、`a` 和 `ac` 服务，并让包装器的 `--kv-mode` 与服务一致。EB-ALFRED
可用 `--episode-indexes 0,1,2,3,4` 固定样本；EB-Habitat 改为 `--benchmark eb-hab`，并用
`--start-episode-index` 指定连续样本起点。`--output-dir` 必须位于仓库的
`inference/kv_reuse/runs/` 下。每个目录中并列保存 `manifest.json`、`calls.jsonl` 和
`server_metrics.jsonl`；使用 `--save-prompts` 时还会创建 `prompts/`。每次实验应使用新的
`--output-dir`，否则 JSONL 会追加到已有文件。EmbodiedBench 的 episode 结果仍写入
`third_party/EmbodiedBench/running/`，其绝对路径记录在 `manifest.json` 的
`evaluation_results` 字段中。

对应上面的 A1+A5 服务，包装器需增加相同配置：

```bash
python /YOUR/PATH/vlm-planning/scripts/kv_reuse/run_embodiedbench.py \
  --benchmark eb-alf --base-url http://127.0.0.1:8001/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --kv-mode a-layerwise --kv-reuse-a-parts 1,5 \
  --kv-recompute-prefix-layers 1 --eval-set base --episodes 50 \
  --output-dir /YOUR/PATH/vlm-planning/inference/kv_reuse/runs/eb_alf_base_a15_l1_epi50_GROUP \
  --save-prompts
```

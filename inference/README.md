# Transformers 本地推理服务

本目录提供基于原生 Transformers `model.generate()` 的本地多模态模型服务。模型推理与 EmbodiedBench 测评分属两个 Conda 环境，通过本机 OpenAI 兼容 HTTP 接口通信，不使用 LMDeploy。

## 分支、环境与代码关系

当前开发配置如下：

| 对象 | 名称 | 用途 |
| --- | --- | --- |
| `vlm-planning` 顶层 Git 分支 | `transformers-profile` | 管理服务脚本、smoke 脚本、实验配置和 Transformers 子模块指针 |
| `third_party/transformers` 子模块分支 | `vlm-profile` | 管理模型内部、生成流程、KV cache 和后续 token merge 修改 |
| 推理 Conda 环境 | `vlm-profile` | 运行 PyTorch、Transformers、FastAPI 和 Qwen3-VL |
| 测评 Conda 环境 | `embench` | 运行 EmbodiedBench、AI2-THOR/Habitat，不加载模型权重 |

顶层仓库和 Transformers 子模块是两个独立 Git 仓库，因此各自拥有独立分支。Conda 环境只保存依赖，不纳入 Git。模型权重保存在仓库外，例如：

```text
/data/home/x_long32/workspace/models/Qwen3-VL-8B-Instruct
```

调用关系如下：

```text
embench 环境中的 EmbodiedBench
        │  HTTP /v1/chat/completions
        ▼
vlm-profile 环境中的 inference/server.py
        │  model.generate()
        ▼
third_party/transformers 中的模型与生成代码
        │
        ▼
仓库外的本地模型权重
```

## 环境配置

当前服务器上的 `vlm-profile` 从已经跑通 Qwen3-VL 的 `qwen-vl` 环境克隆而来：

```bash
conda create -y -n vlm-profile --clone qwen-vl
conda activate vlm-profile

python -m pip install --no-deps --no-build-isolation -e \
  ./third_party/transformers

python -m pip install -r inference/requirements.txt
```

`-e` 表示 editable install。修改 `third_party/transformers/src/transformers/` 中的 Python 源码后，新启动的推理进程会直接使用这些修改，不需要进入 Conda 的 `site-packages` 编辑文件。

检查实际环境和导入路径：

```bash
conda activate vlm-profile

python - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Transformers source:", transformers.__file__)
PY

python -m pip check
```

`Transformers source` 应指向：

```text
vlm-planning/third_party/transformers/src/transformers/__init__.py
```

当前已验证的主要版本为 Python 3.11、PyTorch 2.11.0+cu128、Transformers 5.19.0.dev0、Accelerate 1.15.0、FastAPI 0.142.2 和 Uvicorn 0.54.0。Transformers 子模块基线提交为 `bea93d0faf337b048e83b91f5bb436ed135f53b3`。

官方 fine-grained FP8 checkpoint 还依赖 `kernels==0.17.0`。安装依赖后必须重启 FP8 服务；已经启动的 Python 进程不会重新探测该包。本项目已将 `kernels-community/finegrained-fp8` 的 v4 映射固定到验证过的 revision `0e5764aa5c27d0f005117a92645156481540950a`。首次需要联网下载该 revision 的完整 snapshot；缓存完成后，FP8 服务可以使用 `HF_HUB_OFFLINE=1` 完全离线启动。

## 启动模型服务

先检查 GPU 和端口，再在单独的 tmux 中运行。

### BF16 服务

下面以 GPU 0、端口 8001 为例：

```bash
tmux new -s vlm-profile
conda activate vlm-profile
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /data/home/x_long32/workspace/models/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --port 8001
```

健康检查：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8001/health
curl --noproxy '*' -sS http://127.0.0.1:8001/v1/models
```

### FP8 服务

确认 `kernels==0.17.0` 已安装，且上文固定 revision 的完整
`kernels-community/finegrained-fp8` snapshot 已缓存。下面在 GPU 1、端口
8002 上完全离线启动 FP8 服务：

```bash
tmux new -s vlm-fp8
conda activate vlm-profile
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /data/home/x_long32/workspace/models/Qwen3-VL-8B-Instruct-FP8 \
  --model-name Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --port 8002
```

健康检查：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8002/health
curl --noproxy '*' -sS http://127.0.0.1:8002/v1/models
```

服务只监听 `127.0.0.1`，默认使用 BF16 激活、SDPA 和单请求串行处理；FP8 服务的权重仍使用 checkpoint 中的 fine-grained FP8 量化格式。修改服务或 Transformers 源码后需要重启服务。

`transformers_source` 必须指向本项目的 Transformers 子模块。

## Smoke 测试

Smoke 脚本位于 `scripts/smoke/`，用于快速定位环境、模型服务或 EmbodiedBench 接口问题，不用于统计正式成功率。

### 图片推理测试

[vlm_image.py](../scripts/smoke/vlm_image.py) 会生成一张纯红色图片，通过 `127.0.0.1:8001/v1/chat/completions` 发送给模型，并打印完整响应。

保持模型服务运行，在项目根目录执行：

```bash
conda activate vlm-profile
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning
python scripts/smoke/vlm_image.py
```

`--base-url` 和 `--model-name` 可用于检查另一端口上的模型。例如 FP8 服务运行在 8002 时：

```bash
python scripts/smoke/vlm_image.py \
  --base-url http://127.0.0.1:8002/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct-FP8
```

预期模型识别图片主要颜色为红色，并返回 token 数、结束原因和服务端耗时。

### EB-ALFRED 单步连通测试

[eb_alfred_once.py](../scripts/smoke/eb_alfred_once.py) 会：

1. 加载 EB-ALFRED `base` 子集的第 0 个 episode；
2. 保存真实环境截图；
3. 通过 EmbodiedBench 的 `VLMPlanner` 请求模型；
4. 解析模型 JSON；
5. 执行模型给出的第一个动作；
6. 保存连通性报告并关闭该脚本创建的仿真实例。

确保 Xorg `:1` 和模型服务均已运行，然后执行：

```bash
conda activate embench
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning/third_party/EmbodiedBench

export DISPLAY=:1
export remote_url=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=EMPTY
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

python ../../scripts/smoke/eb_alfred_once.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct
```

检查 FP8 服务时把参数改为 `http://127.0.0.1:8002/v1` 和
`Qwen/Qwen3-VL-8B-Instruct-FP8`。脚本会按模型名称写入独立的
`*_transformers_connectivity` 目录，避免覆盖另一模型的报告。

成功时终端会出现 `CONNECTIVITY_SMOKE_PASS`。报告保存在：

```text
third_party/EmbodiedBench/running/eb_alfred/
Qwen3-VL-8B-Instruct_transformers_connectivity/base/connectivity_report.json
```

该脚本只验证一次“环境截图 → 模型推理 → JSON 解析 → 环境动作”的调用链。动作失败不等于服务连接失败，也不代表完整 episode 的成功率。

## 使用本地 Transformers 服务运行 EmbodiedBench 测评

保持上文的 Transformers 模型服务和 Xorg `:1` 运行，然后在另一个 tmux
窗口中启动测评。下面以 EB-Habitat 的五个子集为例：

```bash
tmux new -s qwen3_eval
conda activate embench
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning/third_party/EmbodiedBench

export DISPLAY=:1
export OPENAI_API_KEY=EMPTY
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
```

评测 BF16（或以 `--dtype float16` 启动的 FP16）服务时：

```bash
export remote_url=http://127.0.0.1:8001/v1
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
EXP_NAME=transformers_bf16

python -m embodiedbench.main \
  env=eb-hab \
  model_name="$MODEL_NAME" \
  model_type=remote \
  exp_name="$EXP_NAME" \
  eval_sets='[common_sense,complex_instruction,spatial_relationship,visual_appearance,long_horizon]'
```

评测 FP8 服务时使用同一条 EmbodiedBench 命令，只需将服务端口从
8001 切换到 8002，并使用 FP8 服务在 `/v1/models` 中声明的模型名。
`EXP_NAME` 仅用于分开两组结果，不会改变测评逻辑：

```bash
export remote_url=http://127.0.0.1:8002/v1
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
EXP_NAME=transformers_fp8

python -m embodiedbench.main \
  env=eb-hab \
  model_name="$MODEL_NAME" \
  model_type=remote \
  exp_name="$EXP_NAME" \
  eval_sets='[common_sense,complex_instruction,spatial_relationship,visual_appearance,long_horizon]'
```

FP8 与 FP16/BF16 使用相同的 EmbodiedBench evaluator、prompt、数据子集和生成参数；
区别只是请求被发送到不同端口的模型服务。`model_name` 必须与对应服务
`/v1/models` 返回的 ID 一致，否则本地服务会返回 HTTP 404。如果要测评其他
EmbodiedBench 环境，只需像常规测评一样修改 `env`、Conda 环境和相关选项，
不需要为 FP8 改动 evaluator。

## EmbodiedBench 推理设置与格式约束

- 服务读取 EmbodiedBench 发来的 `temperature=0` 和 `max_tokens=2048`；温度为 0 时执行 greedy decoding。
- 对 `response_format=json_schema`，服务不会额外追加一条 schema 用户消息，因此不会改变 EmbodiedBench 已构造好的对话和提示词。
- 服务使用 XGrammar 在每一步采样时约束合法 token；生成完成后，再用 `jsonschema` 按同一份 schema 校验。若 schema 本身不合法或结果未通过校验，接口明确返回错误，不把非法 JSON 交给 EmbodiedBench。
- `response_format=json_object` 同样约束顶层必须是 JSON object；普通 `text` 请求不启用格式约束。

输出未通过 JSON/schema 后校验时，服务仍返回 HTTP 500，并通过
`x-should-retry: false` 阻止 SDK 自动重试。VLMPlanner 识别这两种明确的
输出校验错误，记录一次 `output_json_error` 和 planner step，返回 `action=-1`。
EB-Habitat 会累计 invalid action，达到现有上限（10 次）后保存当前 episode
的失败结果并继续下一集。其他 HTTP 错误、网络故障和 CUDA 错误不走此兜底。
旧服务返回的同类错误也能被新客户端识别，但 SDK 可能先做有限次数的重试。
更新后需要重启测评进程；要启用禁止 SDK 重试的响应头，还需重启模型服务。

健康检查中的 `schema_mode` 应为：

```text
xgrammar+jsonschema-validation
```

## 当前实现边界

- 服务暂不支持 streaming、batching、beam search 和 tool calling，目前只适配 Qwen3-VL。
- `finish_reason=length` 表示达到输出 token 上限；遇到 EOS 时返回 `stop`。
- 当前计时覆盖输入处理和完整生成阶段，还没有加入逐层计时或 KV cache 大小监测。
- 进行正式成功率对比或 token merge 实验前，应先用这条 Transformers 推理路径建立独立基线，不应直接与 LMDeploy 基线混为同一设置。

## Git 提交规则

服务、smoke 脚本和分析代码在顶层仓库提交：

```bash
git add inference/ scripts/smoke/
git commit -m "Add Transformers inference service and smoke checks"
```

修改 Transformers 内部代码时，先在子模块提交并推送，再在顶层仓库记录新的子模块指针：

```bash
git -C third_party/transformers add <修改的文件>
git -C third_party/transformers commit -m "Add VLM profiling changes"
git -C third_party/transformers push

git add third_party/transformers
git commit -m "Update Transformers profiling revision"
```

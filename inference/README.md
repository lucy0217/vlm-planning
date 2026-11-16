# Transformers 本地推理服务

本目录提供基于仓库内 Transformers fork 的 Qwen3-VL 本地服务。`server.py` 使用原生
`model.generate()`，对外暴露 OpenAI 兼容的多模态接口，供独立 `embench` 环境中的
EmbodiedBench 调用。环境安装、模型下载、启动命令和测评命令参见
[根目录 README](../README.md)；本文件说明服务实现、KV 复用模块和当前限制。

## 目录职责

```text
inference/
├── server.py                 # HTTP 服务、结构化生成和模型加载入口
├── test_kv_precompute.py     # 项目侧 prepare、分段和 fatal-error 单元测试
└── kv_reuse/                 # 服务端 KV 复用库，不是独立命令行程序
    ├── __init__.py           # 导出 ACKVReuseEngine
    ├── engine.py             # off/a/ac/a-layerwise 调度、首次缓存和回退
    ├── precompute.py         # a-precompute 单次纯文本准备、跨 episode 缓存与位置修正调度
    ├── segments.py           # 定位 P/A1-A6/B/C/S，并生成子段 mask
    ├── cache_manager.py      # canonical cache 的 episode 隔离和生命周期
    ├── metrics.py            # CUDA 分阶段计时与 JSONL 日志路由
    └── runs/                 # 本地实验产物，已被 Git 忽略

scripts/kv_reuse/
|__ run_embodiedbench.py      # 测评入口：注入 episode 上下文并调用 EmbodiedBench


third_party/transformers/src/transformers/models/qwen3_vl/kv_reuse.py
                              # 模型侧 KV 捕获、校验、分段 prefill 和 mRoPE 修正
third_party/transformers/src/transformers/models/qwen3_vl/kv_precompute.py
                              # 独立文本预计算、紧凑 hidden state 和预计算 KV 注入
third_party/transformers/tests/models/qwen3_vl/test_kv_precompute.py
                              # 模型侧数值、计算跳过与位置修正单元测试
```

`inference/kv_reuse` 中的文件由 `server.py` 导入；用户不需要逐个运行。测评时应运行
`scripts/kv_reuse/run_embodiedbench.py`。

## 普通推理调用链

```text
EmbodiedBench（embench 环境）
  → OpenAI SDK：POST /v1/chat/completions
  → inference/server.py（vlm-profile 环境）
  → processor.apply_chat_template()
  → ACKVReuseEngine.generate(mode="off")
  → Qwen3VLForConditionalGeneration.generate()
  → third_party/transformers 中的 Qwen3-VL 实现
  → 仓库外的本地模型权重
```

模型服务和测评运行在不同 Conda 环境及进程中，通过 HTTP 交换图片、prompt 和结果；KV
张量只存在于模型服务进程内，不通过 HTTP 传输。顶层仓库和 `third_party/transformers`
是两个独立 Git 仓库，`vlm-profile` Conda 环境只保存依赖，不保存源码。

## 服务接口与生成约束

`server.py` 只加载本地模型目录，并监听 `127.0.0.1`：

- `GET /health`：模型路径、Transformers 来源、数据类型、KV 模式和结构化生成方式。
- `GET /v1/models`：返回 `--model-name` 指定的模型 ID。
- `POST /v1/chat/completions`：接收文本和内嵌 Base64 图片；只支持非流式、`n=1`。
- `POST /v1/kv-reuse/prepare`：仅供 `a-precompute` 使用；在首个正式推理前准备一次纯文本 KV。

服务依次加载 processor、模型权重和 XGrammar，完成后才启动端口。原始 Qwen3-VL checkpoint
可使用 BF16 或 FP16；官方 fine-grained FP8 checkpoint 使用 FP8 权重和 BF16 激活。默认设置为
SDPA、BF16、单请求串行、端口 8001 和 16000 token 上下文。可配置参数包括
`--model-path`、`--model-name`、`--dtype`、`--port` 和 `--max-context`。修改服务或
Transformers 源码后必须重启服务。

服务读取请求中的 `temperature`、`max_completion_tokens`（兼容 `max_tokens`）和
`response_format`：

- `json_schema`：XGrammar 在采样阶段约束 token，生成后再用 `jsonschema` 校验。
- `json_object`：约束顶层输出为 JSON object。
- `text`：不启用格式约束。

schema 无效返回 HTTP 400；约束生成结果仍未通过校验时返回 HTTP 500，并附带
`x-should-retry: false`，避免 OpenAI SDK 对同一请求自动重试。EmbodiedBench 仍可能把该次
规划视为无效动作，并在它自己的 planner 循环中发起下一次调用。`/health` 中的
`schema_mode` 应为 `xgrammar+jsonschema-validation`。

## KV 复用的 prompt 分段

KV 实验把 processor 最终生成的输入 token 划分为连续的：

```text
P + A + B + C + S
```

| 段 | 当前代码中的含义 | 每次调用是否变化 |
| --- | --- | --- |
| P | Chat template 前缀、当前图片 token，以及原始文本前的模型 token | 图片通常变化 |
| A | 用户文本开头至当前 human instruction 句号，包含规则、动作表、few-shot 和任务指令 | 各子段稳定性不同；A3 随场景、A6 随任务变化 |
| B | 首次调用说明，或后续动作历史和环境反馈 | 动态变化 |
| C | 从 `The output json format should be` 开始到用户文本末尾 | 内容固定，但绝对位置可随 B 改变 |
| S | 用户文本之后的 assistant generation prompt token | 固定但仍需在当前调用计算 |

`segments.py` 先用唯一文本 marker 定位字符区间，再借助 tokenizer 的 offset mapping 映射到
`processor.apply_chat_template()` 产生的实际 token 位置。当前实现要求 batch size 为 1、恰好
一个文本 part、marker 唯一，并要求所有视觉 token 位于 P；不满足时触发 cache miss。

需要特别注意：A 位于当前图片 P 之后。即使 A 的文本完全相同，它首次调用保存的 KV 也已经
受到首张图片和前文影响。因此 `a`/`ac` 是有意进行的近似复用实验，不是数值等价的 prefix
cache；它可能保留旧视觉语义并降低任务成功率。

## `off`、`a`、`ac`、`a-layerwise`、`a-precompute` 执行路径

`server.py` 通过 `--kv-reuse-mode` 选择普通完整计算、episode 内复用或服务生命周期内的
任务前预计算：

| 模式 | 首次请求 | 后续请求或命中路径 |
| --- | --- | --- |
| `off` | 标准完整 `generate()`，不保存 canonical cache | 每次完整计算 P/A/B/C/S |
| `a` | 完整 prefill，并保存每层 A 的 post-RoPE K/V | 计算 P，拼入旧 A KV，再完整计算 B/C/S |
| `ac` | 完整 prefill，并保存每层 A/C 的 post-RoPE K/V | 计算 P，拼入旧 A，计算 B，修正并拼入旧 C，计算 S |
| `a-layerwise` | 完整 prefill，并保存每层 A 的 post-RoPE K/V | 第1层计算全部 token；第2–36层只跳过指定 A 子段，其他 token（包括 C/S）重算 |
| `a-precompute` | 正式请求前独立 prepare A1/固定 A2/A4；第一条请求即可复用 | 第1层计算全部 P/A/B/C；第2–36层跳过所选文本的 hidden state、QKV、Attention query 和 MLP；S/decode 重算 |

AC 模式中，C 的 V 直接复用；C 的 K 使用旧、新三维 mRoPE position IDs 做相对旋转。canonical
cache 始终以该 episode 第一次调用为基准，命中后不会滚动更新。

`a-layerwise` 把 A 连续划分为：A1 角色与动作规则、A2 固定动作 ID 0–161、A3 动态动作
ID≥162、A4 10-shot examples、A5 Guidelines、A6 当前 human instruction。使用
`--kv-reuse-a-parts 1,5` 选择子段，使用 `--kv-recompute-prefix-layers 1` 指定完整重算的
前缀层数。所选 token 在后续层不进入 Q/K/V projection、Attention query 或 MLP；其 canonical
K/V 被拼入完整 cache，供未选 token 的当前 query 使用。该路径是近似实验，不保证与完整推理等价。

详细调用关系如下：

```text
server.complete()
  ├── convert_messages() / apply_chat_template()
  ├── ACKVReuseEngine.generate()
  │   ├── cache_manager.parse_context()       # 校验 run/episode/call 身份
  │   ├── segments.locate_prompt_segments()   # 得到 P/A/B/C/S 边界
  │   ├── EpisodeCacheManager.state_for()     # 读取唯一活动 episode 的缓存
  │   ├── first：标准 generate() 后 build_*_reuse_state()
  │   ├── hit：prefill_with_a_reuse()/prefill_with_ac_reuse()/
  │   │        prefill_with_layerwise_a_reuse()
  │   └── miss：非 strict 时完整 generate()
  ├── metrics.ModelStageTimer                 # vision/prefill/decode 计时
  └── JSONLMetricsWriter.write()              # 写 server_metrics.jsonl
```

模型侧的 `build_*_reuse_state()`、`prefill_with_*_reuse()` 和
`rotate_cached_key_to_positions()` 位于 Transformers 子模块的 `qwen3_vl/kv_reuse.py`。
`a-precompute` 的独立文本 forward、KV 打包和紧凑 prefill 位于同目录的
`kv_precompute.py`，项目侧服务生命周期和日志调度位于 `inference/kv_reuse/precompute.py`。
正常 Qwen3-VL `forward()`/`generate()` 默认行为没有被替换；只有本服务显式调用这些实验接口。

## 缓存身份、生命周期与回退

包装器随请求发送：

```json
{
  "kv_reuse_context": {
    "run_id": "...",
    "benchmark": "eb-alf",
    "eval_set": "base",
    "episode": 1,
    "call_id": 3,
    "env_step": 2,
    "planner_step": 3,
    "kv_mode": "ac",
    "a_parts": [],
    "recompute_prefix_layers": 1,
    "run_dir": "/.../inference/kv_reuse/runs/example"
  }
}
```

对于 `a`、`ac` 和 `a-layerwise`，缓存键为 `(run_id, benchmark, eval_set, episode)`。
当前 `EpisodeCacheManager` 只保留一个活动
episode 的首次 canonical cache；切换 episode 会释放旧缓存。服务使用全局锁串行处理请求，
不支持交错运行多个 episode 或多个测评任务共享同一服务实例。

`a-precompute` 不使用 episode canonical cache。它在服务内存中保存一次 prepare 得到的全局
文本 KV，通过唯一 `precompute_id` 跨 episode 使用。每个服务生命周期只允许一次 prepare
尝试；重复准备、配置不一致或 token 不一致都会返回不可重试的 HTTP 409。切换 run、策略或
准备文本前必须重启服务。

`a` 模式要求 A token、A 三维 position IDs 和区间一致；`ac` 额外要求 C token 和长度一致，
但允许 C 的 position IDs 改变。缺少 context、marker 不唯一、token 不匹配或模式不匹配都会
产生 miss：

`a-layerwise` 只校验所选 A 子段的 token/三维 position IDs，并要求请求中的 `a_parts` 和
`recompute_prefix_layers` 与服务启动配置完全一致。

- 默认行为：本次请求回退完整 prefill，记录 `cache_miss_reason`，但不替换首次 canonical cache。
- `--kv-reuse-strict`：直接报错，适合开发定位，不适合长时间正式测评。
- `--kv-reuse-log PATH`：旧式固定日志路径，优先于请求中的 `run_dir`；包装器测评通常不要设置。

## 测评包装器与产物

KV 实验必须通过 `scripts/kv_reuse/run_embodiedbench.py` 启动。它不修改 EmbodiedBench 子模块，
而是在当前进程中临时包装 `VLMPlanner`：

1. 规范化首次/后续 prompt 中 few-shot 标题和 action-history 前的空白差异；
2. 在 OpenAI 请求的 `extra_body` 中注入 `kv_reuse_context`；
3. 记录 planner 调用并可保存实际发送的文本 prompt；
4. `a-precompute` 在第一条正式请求前调用 prepare，并在随后请求注入 `precompute_id`；
5. 测评结束或异常退出时恢复原始方法并关闭环境。

`--output-dir` 必须是 `inference/kv_reuse/runs/` 的子目录。新 run 通常包含：

| 文件 | 写入方 | 内容 |
| --- | --- | --- |
| `manifest.json` | 包装器 | 参数、运行状态，以及 EmbodiedBench 实际结果目录 `evaluation_results` |
| `calls.jsonl` | 包装器 | 每次 `VLMPlanner.act()` 的 episode/env/planner 序号；不是模型性能日志 |
| `server_metrics.jsonl` | 模型服务 | 每个成功完成请求的缓存状态、token、耗时和显存指标 |
| `precompute_metrics.jsonl` | 模型服务（仅 `a-precompute`） | 一次 prepare 的分组 token、耗时、摘要和全部层 KV 字节数 |
| `prompts/*.txt` | 包装器（需 `--save-prompts`） | 每个实际 HTTP request 的文本部分，不含图片二进制 |

EmbodiedBench 自己的 episode JSON 和 `summary.json` 仍保存在
`third_party/EmbodiedBench/running/`；包装器不会把这些文件复制进 run 目录，而是在 manifest
中记录其绝对路径。

现有 `scripts/kv_reuse/summarize_metrics.py` 是早期 off/a/ac 日志的可选汇总工具；本轮分层
实验不运行也不修改它，直接读取 `server_metrics.jsonl` 和 EmbodiedBench `summary.json`。
分层日志额外记录 A1–A6 token 数、`reused_token_layers`、`computed_prefill_token_layers`、
`layer1_full_seconds`、`later_layers_mixed_seconds` 和 `kv_assembly_seconds`。

## Smoke 测试与当前限制

[vlm_image.py](../scripts/smoke/vlm_image.py) 会在内存中生成一张红色图片并请求服务，只打印
响应，不创建测评结果。启动和测评命令见根 README。

当前限制：

- 仅适配 Qwen3-VL，且只支持单 GPU、batch size 1、非流式和 `n=1`；
- 不支持 batching、beam search、tool calling 或多个并发 episode；
- `a-layerwise` 当前只支持所有后续层使用同一个 A 子段 mask；尚不支持逐层改变 mask；
- `a-precompute` 只支持 A1、A1+A2、A1+A4、A1+A2+A4，并固定完整重算第1层；
- `a-precompute` 的非连续 active query 当前使用 `[active_tokens, prefix_tokens]` 稠密因果
  mask，结果正确但会增加显存访问和 attention 开销；
- `finish_reason=length` 表示达到 completion token 上限，遇到 EOS 返回 `stop`；
- `server_metrics.jsonl` 只在生成和校验成功后写入，HTTP 处理早期失败可能没有对应记录。

## A1/A2/A4 单次预计算模式

本节对应顶层及 Transformers 子模块的 `kv-precompute` 分支。模型环境为
`vlm-profile`，测评环境为 `embench`。旧 `off/a/ac/a-layerwise` 模式继续可用。

启动服务（先确认所选 GPU 和端口可用）：

```bash
conda activate vlm-profile
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
python inference/server.py \
  --model-path /data/home/x_long32/workspace/models/Qwen3-VL-8B-Instruct \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 --port 8001 \
  --kv-reuse-mode a-precompute --kv-reuse-a-parts 1,4 \
  --kv-recompute-prefix-layers 1
```

服务 `/health` 显示 `status=ok` 后，在另一个终端运行：

```bash
conda activate embench
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning/third_party/EmbodiedBench
export DISPLAY=:1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
python ../../scripts/kv_reuse/run_embodiedbench.py \
  --benchmark eb-alf --base-url http://127.0.0.1:8001/v1 \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --kv-mode a-precompute --kv-reuse-a-parts 1,4 \
  --kv-recompute-prefix-layers 1 \
  --eval-set base --episode-indexes 0,1 \
  --output-dir /data/home/x_long32/workspace/tapeout_2027/vlm-planning/inference/kv_reuse/runs/precompute_a14_example \
  --save-prompts
```

仅复用 A1 时，服务和测评两侧都改为 `--kv-reuse-a-parts 1`，并使用新 run 目录。
每次新测评或策略切换都重启服务；不要复用已有结果目录。off 对照两侧分别使用
`--kv-reuse-mode off`、`--kv-mode off` 并去掉 `--kv-reuse-a-parts`。

调用关系：包装器截取第一条正式请求的真实文本 →
`POST /v1/kv-reuse/prepare` → `inference/kv_reuse/precompute.py` →
Transformers `models/qwen3_vl/kv_precompute.py` 独立计算每组文本全部层 KV →
正式 `/v1/chat/completions` 按当前三维位置修正 K → 第1层完整计算、后续层仅维护
未复用位置的 hidden states → 完整 cache 继续处理 S 并 decode。

prepare 只传纯文本、模型名、A 段配置和 run context；不传图片、不执行 decode。
token 从完整原始文本分词结果中切出，不额外加入 BOS 或 chat template。
A1、固定 A2、A4 分别从位置0独立计算；它们彼此没有交叉注意力。A4 的全部示例作为
一个连续组，内部保留因果注意力。固定 A2 只包含 `action id 0` 至 `action id 161`；
随场景变化的动作表标题（例如 `0 ~ 207`）和 A3 的 ID≥162 不进入预计算。当前支持
`--kv-reuse-a-parts 1`、`1,2`、`1,4` 和 `1,2,4`。
每个服务生命周期只允许一次准备尝试，失败后也需重启；KV 不落盘。
准备成功后 `precompute_id` 随后续请求传递，所有 episode（包括首调用）都应命中。
内容、配置或标识不一致会返回带 `kv_precompute_error` 的 HTTP 409，禁止 SDK 重试，
包装器用专门的退出异常绕过测评内部重试并保存失败 manifest。

run 目录包含 `manifest.json`、`calls.jsonl`、`prompts/`、`server_metrics.jsonl`
以及一次 prepare 的 `precompute_metrics.jsonl`。后者记录内容摘要、每组 token 数、
全部层 KV 字节数及准备耗时。在线指标中 `position_correction_seconds` 包含缓存
校验、K 位置修正和紧凑 KV 打包；`online_prefill_seconds` 为语言 prefill 加该开销，
不含视觉编码和 decode。`kv_assembly_seconds` 已包含在语言 prefill 内，不能重复相加。
预计算成本独立报告，并可摊入整个 run 的耗时；它不计入每条在线请求。

该模式不改变 EmbodiedBench 写好的 prompt、图片位置或 token 顺序，但属于近似计算：
预计算文本没有看到当前图片和其他文本组，第2层起直接复用的 KV 因而不等价于完整多模态
prefill。第1层完整重算只能让未复用 token 获取当前跨模态信息，不能更新后续层被复用文本
自身的 hidden state。精度实验必须与相同模型、生成参数和 episode 列表的 `off` 基线比较。

验证命令（项目根目录，vlm-profile 环境）：

```bash
PYTHONPATH=third_party/transformers/src:inference python -m unittest \
  inference/test_kv_reuse.py inference/test_kv_precompute.py \
  third_party/transformers/tests/models/qwen3_vl/test_kv_reuse.py \
  third_party/transformers/tests/models/qwen3_vl/test_kv_precompute.py -v
```

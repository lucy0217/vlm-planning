# Transformers 本地推理服务

本目录提供基于仓库内 Transformers fork 的 Qwen3-VL 本地服务。`server.py` 使用原生
`model.generate()`，对外暴露 OpenAI 兼容的多模态接口，供独立 `embench` 环境中的
EmbodiedBench 调用。环境安装、模型下载、启动命令和测评命令参见
[根目录 README](../README.md)；本文件说明服务实现、KV 复用模块和当前限制。

## 目录职责

```text
inference/
├── server.py                 # HTTP 服务、结构化生成和模型加载入口
└── kv_reuse/                 # 服务端 KV 复用库，不是独立命令行程序
    ├── __init__.py           # 导出 ACKVReuseEngine
    ├── engine.py             # off/a/ac 调度、首次缓存和 hit/miss 回退
    ├── segments.py           # 从 EmbodiedBench prompt 定位 P/A/B/C/S
    ├── cache_manager.py      # canonical cache 的 episode 隔离和生命周期
    ├── metrics.py            # CUDA 分阶段计时与 JSONL 日志路由
    └── runs/                 # 本地实验产物，已被 Git 忽略

scripts/kv_reuse/
|__ run_embodiedbench.py      # 测评入口：注入 episode 上下文并调用 EmbodiedBench


third_party/transformers/src/transformers/models/qwen3_vl/kv_reuse.py
                              # 模型侧 KV 捕获、校验、分段 prefill 和 mRoPE 修正
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
| A | 用户文本开头至当前 human instruction 句号，包含规则、动作表、few-shot 和任务指令 | 规范化后同一 episode 内要求 token/位置完全一致 |
| B | 首次调用说明，或后续动作历史和环境反馈 | 动态变化 |
| C | 从 `The output json format should be` 开始到用户文本末尾 | 内容固定，但绝对位置可随 B 改变 |
| S | 用户文本之后的 assistant generation prompt token | 固定但仍需在当前调用计算 |

`segments.py` 先用唯一文本 marker 定位字符区间，再借助 tokenizer 的 offset mapping 映射到
`processor.apply_chat_template()` 产生的实际 token 位置。当前实现要求 batch size 为 1、恰好
一个文本 part、marker 唯一，并要求所有视觉 token 位于 P；不满足时触发 cache miss。

需要特别注意：A 位于当前图片 P 之后。即使 A 的文本完全相同，它首次调用保存的 KV 也已经
受到首张图片和前文影响。因此 `a`/`ac` 是有意进行的近似复用实验，不是数值等价的 prefix
cache；它可能保留旧视觉语义并降低任务成功率。

## `off`、`a`、`ac` 执行路径

`server.py` 通过 `--kv-reuse-mode off|a|ac` 创建 `ACKVReuseEngine`：

| 模式 | 首次请求 | 同 episode 后续 cache hit |
| --- | --- | --- |
| `off` | 标准完整 `generate()`，不保存 canonical cache | 每次完整计算 P/A/B/C/S |
| `a` | 完整 prefill，并保存每层 A 的 post-RoPE K/V | 计算 P，拼入旧 A KV，再完整计算 B/C/S |
| `ac` | 完整 prefill，并保存每层 A/C 的 post-RoPE K/V | 计算 P，拼入旧 A，计算 B，修正并拼入旧 C，计算 S |

AC 模式中，C 的 V 直接复用；C 的 K 使用旧、新三维 mRoPE position IDs 做相对旋转。canonical
cache 始终以该 episode 第一次调用为基准，命中后不会滚动更新。

详细调用关系如下：

```text
server.complete()
  ├── convert_messages() / apply_chat_template()
  ├── ACKVReuseEngine.generate()
  │   ├── cache_manager.parse_context()       # 校验 run/episode/call 身份
  │   ├── segments.locate_prompt_segments()   # 得到 P/A/B/C/S 边界
  │   ├── EpisodeCacheManager.state_for()     # 读取唯一活动 episode 的缓存
  │   ├── first：标准 generate() 后 build_*_reuse_state()
  │   ├── hit：prefill_with_a_reuse() 或 prefill_with_ac_reuse()
  │   └── miss：非 strict 时完整 generate()
  ├── metrics.ModelStageTimer                 # vision/prefill/decode 计时
  └── JSONLMetricsWriter.write()              # 写 server_metrics.jsonl
```

模型侧的 `build_*_reuse_state()`、`prefill_with_*_reuse()` 和
`rotate_cached_key_to_positions()` 位于 Transformers 子模块的 `qwen3_vl/kv_reuse.py`。
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
    "run_dir": "/.../inference/kv_reuse/runs/example"
  }
}
```

缓存键为 `(run_id, benchmark, eval_set, episode)`。当前 `EpisodeCacheManager` 只保留一个活动
episode 的首次 canonical cache；切换 episode 会释放旧缓存。服务使用全局锁串行处理请求，
不支持交错运行多个 episode 或多个测评任务共享同一服务实例。

`a` 模式要求 A token、A 三维 position IDs 和区间一致；`ac` 额外要求 C token 和长度一致，
但允许 C 的 position IDs 改变。缺少 context、marker 不唯一、token 不匹配或模式不匹配都会
产生 miss：

- 默认行为：本次请求回退完整 prefill，记录 `cache_miss_reason`，但不替换首次 canonical cache。
- `--kv-reuse-strict`：直接报错，适合开发定位，不适合长时间正式测评。
- `--kv-reuse-log PATH`：旧式固定日志路径，优先于请求中的 `run_dir`；包装器测评通常不要设置。

## 测评包装器与产物

KV 实验必须通过 `scripts/kv_reuse/run_embodiedbench.py` 启动。它不修改 EmbodiedBench 子模块，
而是在当前进程中临时包装 `VLMPlanner`：

1. 规范化首次/后续 prompt 中 few-shot 标题和 action-history 前的空白差异；
2. 在 OpenAI 请求的 `extra_body` 中注入 `kv_reuse_context`；
3. 记录 planner 调用并可保存实际发送的文本 prompt；
4. 测评结束或异常退出时恢复原始方法并关闭环境。

`--output-dir` 必须是 `inference/kv_reuse/runs/` 的子目录。新 run 通常包含：

| 文件 | 写入方 | 内容 |
| --- | --- | --- |
| `manifest.json` | 包装器 | 参数、运行状态，以及 EmbodiedBench 实际结果目录 `evaluation_results` |
| `calls.jsonl` | 包装器 | 每次 `VLMPlanner.act()` 的 episode/env/planner 序号；不是模型性能日志 |
| `server_metrics.jsonl` | 模型服务 | 每个成功完成请求的缓存状态、token、耗时和显存指标 |
| `prompts/*.txt` | 包装器（需 `--save-prompts`） | 每个实际 HTTP request 的文本部分，不含图片二进制 |

EmbodiedBench 自己的 episode JSON 和 `summary.json` 仍保存在
`third_party/EmbodiedBench/running/`；包装器不会把这些文件复制进 run 目录，而是在 manifest
中记录其绝对路径。

`scripts/kv_reuse/summarize_metrics.py` 汇总请求数、hit/miss、复用 token 和平均耗时。提供
`--baseline-jsonl` 时，它按 benchmark、eval set、episode、planner step 配对 baseline 与 hit
请求，报告 prefill 和 generation speedup。off/a/ac 应使用相同 episode 列表分别运行。

## Smoke 测试与当前限制

[vlm_image.py](../scripts/smoke/vlm_image.py) 会在内存中生成一张红色图片并请求服务，只打印
响应，不创建测评结果。启动和测评命令见根 README。

当前限制：

- 仅适配 Qwen3-VL，且只支持单 GPU、batch size 1、非流式和 `n=1`；
- 不支持 batching、beam search、tool calling 或多个并发 episode；
- KV 模式目前按所有层整体复用 A/AC，不支持分层动态选择；
- `finish_reason=length` 表示达到 completion token 上限，遇到 EOS 返回 `stop`；
- `server_metrics.jsonl` 只在生成和校验成功后写入，HTTP 处理早期失败可能没有对应记录。

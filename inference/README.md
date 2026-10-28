# Transformers 本地推理服务

本目录的 `server.py` 使用仓库内的 Transformers fork 和原生
`model.generate()`，为 EmbodiedBench 提供 OpenAI 兼容的本地多模态接口。
环境安装、Qwen3-VL 模型下载、BF16/FP8 服务启动和测评命令统一参见
[根目录 README](../README.md)，本文件只说明服务自身的行为和限制。

## 调用关系

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
仓库外的本地 Qwen3-VL 权重
```

顶层仓库和 `third_party/transformers` 是两个独立 Git 仓库。当前 `master`
记录固定的 Transformers 子模块提交；子模块处于 detached HEAD 是正常现象。
`vlm-profile` Conda 环境只保存运行依赖，不纳入 Git。

## 服务行为

`server.py` 只读取本地模型目录，并监听 `127.0.0.1`。主要接口为：

- `GET /health`：返回模型、Transformers 来源、数据类型和结构化生成模式。
- `GET /v1/models`：返回启动时通过 `--model-name` 指定的模型 ID。
- `POST /v1/chat/completions`：接收 EmbodiedBench 的文本和内嵌 Base64 图片请求。

服务依次加载 processor、模型权重和 XGrammar，完成后才启动 HTTP 端口。因此模型加载期间
`/health` 尚不可访问。原始 Qwen3-VL checkpoint 可使用 BF16 或 FP16；官方 FP8
checkpoint 保持 fine-grained FP8 权重，并使用 BF16 激活。FP8 还依赖
`kernels==0.17.0` 和根 README 中固定 revision 的 kernel snapshot。

服务默认使用 SDPA、单请求串行处理和 16000 token 上下文。`--model-path`、
`--model-name`、`--dtype`、`--port` 和 `--max-context` 可通过命令行配置。
修改服务代码或 Transformers 源码后需要重启服务。

## 结构化输出

服务读取 EmbodiedBench 传入的 `temperature`、token 上限和 `response_format`：

- `json_schema` 使用 XGrammar 约束生成，并在生成后用 `jsonschema` 再次校验。
- `json_object` 约束顶层为 JSON object。
- 普通 `text` 请求不启用格式约束。

schema 无效或输出未通过校验时，服务返回 HTTP 500，并通过
`x-should-retry: false` 阻止 SDK 自动重试。EmbodiedBench 会把该次结果记为无效动作，
而不会无限等待同一请求。`/health` 中的 `schema_mode` 应为
`xgrammar+jsonschema-validation`。

## Smoke 测试与限制

[vlm_image.py](../scripts/smoke/vlm_image.py) 会在内存中生成一张红色图片并请求服务，
响应只打印到终端，不会创建结果文件。运行方法见根 README。

当前服务只适配 Qwen3-VL，不支持 streaming、batching、beam search 或 tool calling。
`finish_reason=length` 表示达到输出 token 上限，遇到 EOS 时返回 `stop`。

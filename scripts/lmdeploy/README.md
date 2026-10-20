# RTX 5090: Qwen2-VL with an independent LMDeploy service

This uses EmbodiedBench's method 2. The model runs in the `lmdeploy` Conda
environment; evaluation continues in `embench`. No changes to EmbodiedBench's
planner or custom `server.py` are required.

## Environment

Core versions are pinned in `requirements.txt`: Python 3.11, LMDeploy 0.13.0,
PyTorch 2.10.0, torchvision 0.25.0, Transformers 4.57.6. The Linux x86_64 PyPI
PyTorch 2.10.0 package includes CUDA 12.8 dependencies, supporting SM120.
This is a newer software stack than the benchmark's historical environment.
XGrammar is pinned to 0.1.27: 0.2.8 rejects tensor token IDs passed by this
LMDeploy release and crashes structured generation. Keep this pin even though
unconstrained image description also works with the newer XGrammar.

`requirements.lock.txt` records the full installed pip environment; use it instead
of `requirements.txt` when reproducing all transitive dependency versions.

For a fresh installation, from the `vlm-planning` directory:

```bash
conda create -n lmdeploy python=3.11 pip -y
conda activate lmdeploy
python -m pip install -r scripts/lmdeploy/requirements.txt
python -m pip check
```

On this machine the previously failed environment contained only Python and
packaging tools, so it was upgraded in place with
`conda install -n lmdeploy python=3.11 pip -y` instead of recreating it.
Neither `embench` nor `qwen-vl` is modified.

If shell proxy variables point to an unavailable local proxy, run installation
with those variables removed for that command only:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    python -m pip install -r scripts/lmdeploy/requirements.txt \
    --index-url https://pypi.org/simple --timeout 120
```

## Start the model service

In a tmux window, from `vlm-planning`:

```bash
conda activate lmdeploy
CUDA_VISIBLE_DEVICES=0 bash scripts/lmdeploy/serve_qwen2vl.sh
```

Choose an available GPU before starting. Defaults: PyTorch backend, FP16,
TP=1, session length 12000, batch size 1, KV cache fraction 0.4, and local-only
endpoint `http://127.0.0.1:23333/v1`. The script uses the already downloaded
Qwen2-VL-7B snapshot and enables Hugging Face offline mode. Override `MODEL_PATH`,
`LMDEPLOY_BIN`, `PORT`, or `CUDA_VISIBLE_DEVICES` as needed. API model name stays
`Qwen/Qwen2-VL-7B-Instruct` even when the weight path is local.

Keep the service running during evaluation. Ctrl-C in its tmux window stops it
and releases the model's GPU memory; no administrator is required.

## Verify image inference

In another terminal, from `vlm-planning`:

```bash
conda activate lmdeploy
python scripts/lmdeploy/smoke_test.py \
    --output scripts/lmdeploy/artifacts/smoke_test.json
```

To additionally verify the repository's actual client from its original environment:

```bash
conda activate embench
python scripts/lmdeploy/smoke_test.py --benchmark-client \
    --output scripts/lmdeploy/artifacts/embench_client_test.json
```

The test sends a repository kitchen image as base64 to the OpenAI-compatible
endpoint. It checks both plain image description and structured output with
EmbodiedBench's unmodified `vlm_generation_guide` JSON schema. This validates the
model service, not the full Habitat benchmark or its task success rate. Timings
include first-request warmup and are not steady-state profiling results.

Validated on 2026-10-19 with GPU 0 (RTX 5090, driver 595.58.03):

- PyTorch reports `2.10.0+cu128`, CUDA runtime 12.8, and `sm_120`; GPU FP16
  matrix multiplication passed.
- `pip check` passed.
- Real image request correctly described a kitchen, stove, sink, and coffee maker.
- The benchmark JSON schema request returned a valid object with all four required
  fields and an executable plan; generation finished normally.
- The unmodified `embodiedbench.planner.remote_model.RemoteModel` in `embench`
  successfully called this service and returned schema-valid JSON.
- The verification server was stopped after testing; start it again before evaluation.

The current Transformers version selects the fast image processor by default.
Together with the newer inference stack, this may change outputs relative to the
paper's historical dependencies. Full Habitat episodes and benchmark scores have
not been tested here.

## Run evaluation

In another terminal:

```bash
conda activate embench
cd /data/home/x_long32/workspace/tapeout_2027/vlm-planning/third_party/EmbodiedBench
export remote_url="http://127.0.0.1:23333/v1"
export OPENAI_API_KEY="EMPTY"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
python -m embodiedbench.main \
    env=eb-hab \
    model_name=Qwen/Qwen2-VL-7B-Instruct \
    model_type=remote \
    exp_name=qwen2vl_lmdeploy
```

`EMPTY` is a non-secret client placeholder for the local server without API-key
authentication. Use `remote_url` with `/v1`, not method 3's `server_url` and
`/process`. The model name must retain its capitalization to select the correct
branch of the repository's client. Any unrelated PyTorch GPU work in the old
evaluation environment still needs separate compatibility checks.

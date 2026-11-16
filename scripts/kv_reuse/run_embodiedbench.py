"""Run EB-ALFRED or EB-Habitat while attaching episode KV-reuse context.

Run this script from ``third_party/EmbodiedBench`` in the embench environment.
It intentionally avoids modifying the EmbodiedBench submodule.
"""

import argparse
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path


VALID_EVAL_SETS = {
    "eb-alf": ("base", "common_sense", "complex_instruction", "spatial", "visual_appearance", "long_horizon"),
    "eb-hab": (
        "base",
        "common_sense",
        "complex_instruction",
        "spatial_relationship",
        "visual_appearance",
        "long_horizon",
    ),
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = (PROJECT_ROOT / "inference" / "kv_reuse" / "runs").resolve()


class FatalPrecomputeError(BaseException):
    """Escape benchmark Exception handlers; main's finally still saves the manifest."""


def prepare_once(base_url, model_name, messages, parts, context):
    text = [part['text'] for message in messages for part in message.get('content', [])
            if isinstance(part, dict) and part.get('type') == 'text']
    if len(text) != 1:
        raise FatalPrecomputeError('prepare requires exactly one prompt text')
    payload = {'model': model_name, 'text': text[0], 'a_parts': list(parts), 'context': context}
    request = urllib.request.Request(
        base_url.rstrip('/') + '/kv-reuse/prepare', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=300) as response:
            return json.load(response)
    except Exception as error:
        raise FatalPrecomputeError(f'prepare failed: {error}') from error


def is_precompute_error(error):
    body = getattr(error, 'body', None)
    if not isinstance(body, dict):
        return False
    detail = body.get('detail', body)
    return isinstance(detail, dict) and detail.get('code') == 'kv_precompute_error'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("eb-alf", "eb-hab"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--kv-mode", choices=("off", "a", "ac", "a-layerwise", "a-precompute"), required=True)
    parser.add_argument("--kv-reuse-a-parts", default="",
                        help="A parts for a-layerwise; a-precompute supports 1, 1,2, 1,4, or 1,2,4")
    parser.add_argument("--kv-recompute-prefix-layers", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-set", default="base")
    parser.add_argument("--episode-indexes", help="Comma-separated zero-based indexes for EB-ALFRED")
    parser.add_argument("--start-episode-index", type=int, default=0, help="Zero-based first EB-Habitat episode")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-prompts", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.eval_set not in VALID_EVAL_SETS[args.benchmark]:
        parser.error(f"Invalid --eval-set; choose from {', '.join(VALID_EVAL_SETS[args.benchmark])}")
    if args.benchmark == "eb-hab" and args.episode_indexes:
        parser.error("--episode-indexes is only supported by EB-ALFRED")
    if args.benchmark == "eb-alf" and args.start_episode_index:
        parser.error("--start-episode-index is only supported by EB-Habitat")
    try:
        args.kv_reuse_a_parts = tuple(
            int(value) for value in args.kv_reuse_a_parts.split(",") if value.strip()
        )
    except ValueError:
        parser.error("--kv-reuse-a-parts must contain comma-separated integers")
    if args.kv_mode in ("a-layerwise", "a-precompute") and not args.kv_reuse_a_parts:
        parser.error("--kv-reuse-a-parts is required for this reuse mode")
    if args.kv_mode not in ("a-layerwise", "a-precompute") and args.kv_reuse_a_parts:
        parser.error("--kv-reuse-a-parts requires a-layerwise or a-precompute mode")
    if any(part not in range(1, 7) for part in args.kv_reuse_a_parts):
        parser.error("A parts must be integers from 1 through 6")
    args.kv_reuse_a_parts = tuple(sorted(set(args.kv_reuse_a_parts)))
    if args.kv_mode == 'a-precompute' and (
        args.kv_reuse_a_parts not in ((1,), (1, 2), (1, 4), (1, 2, 4))
        or args.kv_recompute_prefix_layers != 1
    ):
        parser.error('a-precompute requires 1, 1,2, 1,4, or 1,2,4 and exactly one recomputed layer')
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    try:
        output.relative_to(RUNS_ROOT)
    except ValueError:
        raise SystemExit(f"--output-dir must be inside {RUNS_ROOT}") from None
    if output == RUNS_ROOT:
        raise SystemExit(f"--output-dir must name a run directory below {RUNS_ROOT}")
    output.mkdir(parents=True, exist_ok=True)
    prompts = output / "prompts"
    if args.save_prompts:
        prompts.mkdir(exist_ok=True)
    os.environ["remote_url"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    for name in ('NO_PROXY', 'no_proxy'):
        os.environ[name] = ','.join(filter(None, [os.environ.get(name, ''), '127.0.0.1', 'localhost']))

    from omegaconf import OmegaConf
    from embodiedbench.planner.vlm_planner import VLMPlanner

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    strategy = args.kv_mode
    if args.kv_mode == "a-layerwise":
        strategy = "a" + "".join(str(part) for part in args.kv_reuse_a_parts) + f"_l{args.kv_recompute_prefix_layers}"
    elif args.kv_mode == 'a-precompute':
        strategy = 'precompute_a' + ''.join(map(str, args.kv_reuse_a_parts)) + '_l1'
    run_id = f"{args.benchmark}-{args.eval_set}-{strategy}-{timestamp}"
    common = {
        "model_name": args.model_name,
        "model_type": "remote",
        "exp_name": f"kv_{strategy}_{timestamp}",
        "eval_sets": [args.eval_set],
    }
    if args.benchmark == "eb-alf":
        from embodiedbench.evaluator.eb_alfred_evaluator import EB_AlfredEvaluator

        config = OmegaConf.load("embodiedbench/configs/eb-alf.yaml")
        try:
            selected = (
                list(range(args.episodes))
                if not args.episode_indexes
                else [int(value) for value in args.episode_indexes.split(",")]
            )
        except ValueError:
            raise SystemExit("--episode-indexes must contain comma-separated integers") from None
        if not selected or min(selected) < 0:
            raise SystemExit("--episode-indexes must contain nonnegative integers")
        config.update({**common, "selected_indexes": selected})
        evaluator = EB_AlfredEvaluator(config)
    else:
        from embodiedbench.evaluator.eb_habitat_evaluator import EB_HabitatEvaluator

        config = OmegaConf.load("embodiedbench/configs/eb-hab.yaml")
        config.update({**common, "down_sample_ratio": 1.0, "start_epi_index": args.start_episode_index})
        evaluator = EB_HabitatEvaluator(config)
        original_evaluate = evaluator.evaluate

        def evaluate_limited():
            start = evaluator.env._current_episode_num
            if args.episodes > int(evaluator.env.number_of_episodes) - start:
                raise ValueError("Requested more Habitat episodes than remain in the split")
            evaluator.env.number_of_episodes = start + args.episodes
            return original_evaluate()

        evaluator.evaluate = evaluate_limited

    original_act = VLMPlanner.act
    original_process_prompt = VLMPlanner.process_prompt
    call_number = 0
    prepared = None

    def normalized_process_prompt(planner, *prompt_args, **prompt_kwargs):
        # EmbodiedBench emits "Example {i}" on the first planner call but
        # "Example  {i}" on later calls. Normalize that whitespace-only typo so
        # the experiment's A segment is actually token-identical. Apply this in
        # both off/ac modes to keep the baseline comparison fair.
        prompt = original_process_prompt(planner, *prompt_args, **prompt_kwargs)
        return prompt.replace("## Task Execution Example  ", "## Task Execution Example ").replace(
            "\n\n The action history:", " The action history:"
        )

    def contextualized_act(planner, observation, instruction):
        nonlocal call_number
        call_number += 1
        local_episode = evaluator.env._current_episode_num
        if args.benchmark == "eb-alf" and evaluator.env.selected_indexes:
            dataset_episode = evaluator.env.selected_indexes[local_episode - 1] + 1
        else:
            dataset_episode = local_episode
        context = {
            "run_id": run_id,
            "benchmark": args.benchmark,
            "eval_set": evaluator.eval_set,
            "episode": dataset_episode,
            "call_id": call_number,
            "env_step": evaluator.env._current_step,
            "planner_step": planner.planner_steps,
            "kv_mode": args.kv_mode,
            "a_parts": list(args.kv_reuse_a_parts),
            "recompute_prefix_layers": args.kv_recompute_prefix_layers,
            "run_dir": str(output),
        }
        completions = planner.model.model.chat.completions
        original_create = completions.create
        request_number = 0

        def create(**kwargs):
            nonlocal request_number, prepared
            request_number += 1
            if args.kv_mode == 'a-precompute':
                if prepared is None:
                    prepared = prepare_once(args.base_url, args.model_name, kwargs.get('messages', []),
                                            args.kv_reuse_a_parts, context)
                    manifest['precompute'] = prepared
                    (output / 'manifest.json').write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
                context['precompute_id'] = prepared['precompute_id']
            if request_number == 1:
                with (output / "calls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(context, ensure_ascii=False) + "\n")
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "kv_reuse_context": context}
            if args.save_prompts:
                text = []
                for message in kwargs.get("messages", []):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text.append(content)
                    elif isinstance(content, list):
                        text.extend(part.get("text", "") for part in content if part.get("type") == "text")
                name = (
                    f"call_{call_number:03d}_episode_{dataset_episode}_envstep_{evaluator.env._current_step}"
                    f"_request_{request_number}.txt"
                )
                (prompts / name).write_text("\n\n".join(text), encoding="utf-8")
            try:
                return original_create(**kwargs)
            except Exception as error:
                if is_precompute_error(error):
                    raise FatalPrecomputeError(str(error)) from error
                raise

        completions.create = create
        try:
            return original_act(planner, observation, instruction)
        finally:
            completions.create = original_create

    VLMPlanner.process_prompt = normalized_process_prompt
    VLMPlanner.act = contextualized_act
    manifest = {
        "status": "running",
        "run_id": run_id,
        "benchmark": args.benchmark,
        "kv_mode": args.kv_mode,
        "strategy": strategy,
        "reused_a_parts": list(args.kv_reuse_a_parts),
        "recompute_prefix_layers": args.kv_recompute_prefix_layers,
        "base_url": args.base_url,
        "config": OmegaConf.to_container(config),
        "calls_file": str(output / "calls.jsonl"),
        "server_metrics_file": str(output / "server_metrics.jsonl"),
        "precompute_metrics_file": str(output / "precompute_metrics.jsonl") if args.kv_mode == 'a-precompute' else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        evaluator.evaluate_main()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=str(error))
        raise
    finally:
        VLMPlanner.act = original_act
        VLMPlanner.process_prompt = original_process_prompt
        if evaluator.env is not None:
            manifest["evaluation_results"] = str(Path(evaluator.env.log_path).resolve())
            evaluator.env.close()
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

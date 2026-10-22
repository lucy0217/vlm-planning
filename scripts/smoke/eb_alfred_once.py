"""使用真实 ALFRED 截图完成一次规划请求并执行第一个动作。

需要在 EmbodiedBench 仓库目录中使用 embench 环境运行。
这只是连通性测试，不是完整 episode 或成功率测评。
"""
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default=os.getenv('remote_url', 'http://127.0.0.1:8001/v1'))
    parser.add_argument('--model-name', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--exp-name', default=None)
    args = parser.parse_args()

    # RemoteModel reads remote_url when its module is imported, so set it first.
    os.environ['remote_url'] = args.base_url.rstrip('/')
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
    from embodiedbench.evaluator.eb_alfred_evaluator import example_path, system_prompt
    from embodiedbench.planner.vlm_planner import VLMPlanner

    config = OmegaConf.load('embodiedbench/configs/eb-alf.yaml')
    run_name = args.exp_name or f"{args.model_name.split('/')[-1]}_transformers_connectivity/base"
    env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[0],
                   exp_name=run_name, resolution=config.resolution,
                   detection_box=config.detection_box)
    try:
        obs = env.reset()
        image = env.save_image(obs)
        with open(example_path) as f:
            examples = json.load(f)
        planner = VLMPlanner(args.model_name, 'remote', env.language_skill_set,
                             system_prompt, examples, n_shot=config.n_shots,
                             obs_key='head_rgb', chat_history=config.chat_history,
                             language_only=config.language_only, use_feedback=config.env_feedback,
                             multistep=config.multistep, tp=1)
        planner.reset()
        planner.set_actions(env.language_skill_set)
        action, reasoning = planner.act(image, env.episode_language_instruction)
        report = dict(instruction=env.episode_language_instruction, image=str(image),
                      action=action, reasoning=reasoning, json_errors=planner.output_json_error)
        first = action[0] if isinstance(action, list) and action else action
        if not isinstance(first, int) or first < 0:
            raise RuntimeError(f'Planner did not return an executable action: {action}')
        _, reward, done, info = env.step(first, reasoning=reasoning)
        report.update(executed_action=first, reward=reward, done=done, info=info)
        output = Path(env.log_path) / 'connectivity_report.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print('CONNECTIVITY_SMOKE_PASS', output.resolve())
        print(json.dumps(report, ensure_ascii=False, default=str))
    finally:
        env.close()


if __name__ == '__main__':
    main()

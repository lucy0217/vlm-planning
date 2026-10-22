"""使用真实 ALFRED 截图完成一次规划请求并执行第一个动作。

需要在 EmbodiedBench 仓库目录中使用 embench 环境运行。
这只是连通性测试，不是完整 episode 或成功率测评。
"""
import json
from pathlib import Path

from omegaconf import OmegaConf
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
from embodiedbench.evaluator.eb_alfred_evaluator import example_path, system_prompt
from embodiedbench.planner.vlm_planner import VLMPlanner

config = OmegaConf.load('embodiedbench/configs/eb-alf.yaml')
env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[0],
               exp_name='Qwen3-VL-8B-Instruct_transformers_connectivity/base',
               resolution=config.resolution, detection_box=config.detection_box)
try:
    obs = env.reset()
    image = env.save_image(obs)
    with open(example_path) as f:
        examples = json.load(f)
    planner = VLMPlanner('Qwen/Qwen3-VL-8B-Instruct', 'remote', env.language_skill_set,
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

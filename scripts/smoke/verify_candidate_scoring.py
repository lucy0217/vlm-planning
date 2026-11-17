"""GPU numerical check against full forwards with exactly identical token IDs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from inference.candidate_scoring import score_candidates, token_scores


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tiny-fp32', action='store_true',
                        help='Use a small random Qwen3-VL in FP32 to check exact cache/position logic')
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    if args.tiny_fp32:
        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, rope_parameters=dict(rope_type='default',
                    rope_theta=500000, mrope_section=[2, 3, 3], mrope_interleaved=True)),
            vision_config=dict(depth=2, hidden_size=32, intermediate_size=64,
                num_heads=4, out_hidden_size=64, deepstack_visual_indexes=[0, 1]))
        model = Qwen3VLForConditionalGeneration(config).to('cuda').eval()
        tolerance = 1e-4
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa').eval()
        # BF16 forwards with different GEMM shapes can quantize logits differently;
        # use FP32 above to isolate indexing/mRoPE correctness from that effect.
        tolerance = 0.5
    counts = [0]
    hook = model.model.visual.register_forward_hook(
        lambda *unused: counts.__setitem__(0, counts[0] + 1))
    candidates = [' red', ' blue', ' green and yellow', ' dark red', ' bright blue']
    results = []
    for color in ('red', 'blue'):
        messages = [dict(role='user', content=[
            dict(type='image', image=Image.new('RGB', (300, 300), color)),
            dict(type='text', text='Describe the color of the image.')])]
        inputs = processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
        references = []
        for text in candidates:
            suffix = torch.tensor([processor.tokenizer.encode(text, add_special_tokens=False)],
                                  device=model.device)
            width = suffix.shape[1]
            full = dict(inputs)
            full['input_ids'] = torch.cat([inputs.input_ids, suffix], -1)
            full['attention_mask'] = torch.ones_like(full['input_ids'])
            if 'mm_token_type_ids' in full:
                full['mm_token_type_ids'] = torch.cat([
                    inputs.mm_token_type_ids, torch.zeros_like(suffix)], -1)
            model.model.rope_deltas = None
            output = model(**full, use_cache=False, logits_to_keep=width + 1)
            references.append(token_scores(output.logits[:, :-1], suffix,
                                            torch.ones_like(suffix), 'sum').item())
            del output
        for batch in (1, 8):
            before = counts[0]
            result = score_candidates(model, processor, messages, candidates,
                                      candidate_batch_size=batch)
            assert counts[0] - before == 1, 'Visual encoder must run exactly once per request'
            errors = [abs(row['score'] - ref) for row, ref in zip(result['scores'], references)]
            assert max(errors) < tolerance, (color, batch, errors)
            assert result['best_index'] == max(range(len(references)), key=references.__getitem__)
            results.append(dict(color=color, batch=batch, tolerance=tolerance,
                                tiny_fp32=args.tiny_fp32, max_absolute_error=max(errors),
                                reference=references, result=result))
        avg = score_candidates(model, processor, messages, candidates, score_function='avg')
        for row, summed in zip(avg['scores'], results[-1]['result']['scores']):
            assert abs(row['score'] * row['token_count'] - summed['score']) < 0.02
    hook.remove()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2))
    print(json.dumps([dict(color=r['color'], batch=r['batch'],
                           max_absolute_error=r['max_absolute_error']) for r in results], indent=2))


if __name__ == '__main__':
    main()

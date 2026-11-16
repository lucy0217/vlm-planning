"""One-shot, service-lifetime text memories; no episode-first-call capture."""

import hashlib
import json
import time
import uuid

import torch
from transformers.models.qwen3_vl.kv_precompute import (
    build_precomputed_reuse_plan,
    precompute_text_group,
    prefill_with_precomputed_reuse,
)
from transformers.models.qwen3_vl.kv_reuse import prepare_qwen3_vl_position_ids

from .cache_manager import parse_context
from .metrics import ModelStageTimer, synchronize
from .segments import locate_prompt_segments


class PrecomputeError(ValueError):
    """Fatal experiment configuration error, never an invalid planner action."""


def extract_prepare_tokens(tokenizer, text, parts):
    raw = tokenizer(text, add_special_tokens=False).input_ids
    # Locator-only sentinels provide P/S boundaries; never fed to the model.
    ids = torch.tensor([[-1, *raw, -2]])
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    segments = locate_prompt_segments(tokenizer, ids, messages)
    return {part: ids[:, segments.precompute_a_parts[part]].clone() for part in parts}


class PrecomputedKVEngine:
    def __init__(self, model, tokenizer, parts):
        if parts not in ((1,), (1, 2), (1, 4), (1, 2, 4)):
            raise ValueError("a-precompute supports A1, A1+A2, A1+A4, or A1+A2+A4")
        self.model, self.tokenizer, self.parts = model, tokenizer, parts
        self.prepare_attempted = False
        self.prepared = None
        self.groups = {}

    @torch.inference_mode()
    def prepare(self, text, parts):
        if self.prepare_attempted:
            raise PrecomputeError("prepare_already_attempted: restart the service")
        if tuple(parts) != self.parts or not isinstance(text, str) or not text:
            raise PrecomputeError("invalid_prepare_configuration")
        self.prepare_attempted = True
        synchronize()
        started = time.perf_counter()
        try:
            tokens = extract_prepare_tokens(self.tokenizer, text, self.parts)
            groups, records = {}, {}
            for part, ids in tokens.items():
                synchronize()
                begin = time.perf_counter()
                group = precompute_text_group(self.model.model.language_model, ids)
                synchronize()
                groups[part] = group
                records[f"A{part}"] = {
                    "tokens": ids.shape[1],
                    "token_sha256": hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
                    "seconds": time.perf_counter() - begin,
                    "kv_bytes": sum(t.numel() * t.element_size() for t in (*group.keys, *group.values)),
                    "layers": len(group.keys),
                    "source": "independent_text_forward",
                }
        except ValueError as error:
            raise PrecomputeError(str(error)) from error
        self.groups = groups
        self.prepared = {
            "precompute_id": uuid.uuid4().hex, "a_parts": list(self.parts), "groups": records,
            "prepare_seconds": time.perf_counter() - started,
            "dtype": str(next(self.model.parameters()).dtype),
            "text_config": self.model.config.text_config.to_dict(),
        }
        return self.prepared

    @torch.inference_mode()
    def generate(self, inputs, messages, context, generation_kwargs):
        started = time.perf_counter()
        try:
            parse_context(context)
            if self.prepared is None:
                raise ValueError("prepare_required")
            if context.get("precompute_id") != self.prepared["precompute_id"]:
                raise ValueError("precompute_id_mismatch")
            if (context.get("kv_mode") != "a-precompute"
                    or tuple(context.get("a_parts", ())) != self.parts
                    or context.get("recompute_prefix_layers") != 1):
                raise ValueError("precompute_configuration_mismatch")
            segments = locate_prompt_segments(self.tokenizer, inputs["input_ids"], messages)
            positions, deltas = prepare_qwen3_vl_position_ids(self.model, inputs)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            synchronize()
            rotation_start = time.perf_counter()
            plan = build_precomputed_reuse_plan(
                self.model.model.language_model, self.groups, segments.precompute_a_parts,
                inputs["input_ids"], positions, segments.model.c_end,
            )
            synchronize()
            rotation_seconds = time.perf_counter() - rotation_start
            with ModelStageTimer(self.model) as timer:
                prefix = prefill_with_precomputed_reuse(
                    self.model, inputs, positions, deltas, segments.model, plan, segments.token_counts,
                )
                output = self.model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    position_ids=positions, past_key_values=prefix.past_key_values,
                    return_dict_in_generate=True, **generation_kwargs,
                )
        except ValueError as error:
            raise PrecomputeError(str(error)) from error
        timing = timer.summary()
        assembly = sum(
            a.elapsed_time(b) / 1000 if isinstance(a, torch.cuda.Event) else b - a
            for a, b in plan.kv_assembly_timing
        )
        timing.update(
            position_correction_seconds=rotation_seconds, kv_assembly_seconds=assembly,
            online_prefill_seconds=timing["prefill_seconds"] + rotation_seconds,
            total_generation_seconds=time.perf_counter() - started,
        )
        reused = int(plan.reuse_indices.numel())
        layers = self.model.config.text_config.num_hidden_layers
        token_count = inputs["input_ids"].shape[1]
        metadata = {
            "reuse_mode": "a-precompute", "cache_status": "hit", "cache_miss_reason": None,
            "precompute_id": self.prepared["precompute_id"], "reused_a_parts": list(self.parts),
            "recompute_prefix_layers": 1, "segment_tokens": segments.token_counts,
            "computed_prefill_tokens": token_count, "reused_tokens": reused,
            "directly_reused_tokens": 0, "rope_corrected_reused_tokens": reused,
            "reused_token_layers": reused * (layers - 1),
            "computed_prefill_token_layers": token_count * layers - reused * (layers - 1),
            "positions": {
                f"A{part}": {
                    "token_start": segments.precompute_a_parts[part].start,
                    "token_end": segments.precompute_a_parts[part].stop,
                    "current_start": positions[1:, 0, segments.precompute_a_parts[part].start].tolist(),
                    "current_end": positions[1:, 0, segments.precompute_a_parts[part].stop - 1].tolist(),
                } for part in self.parts
            },
            "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "timing": timing,
        }
        return output.sequences, metadata

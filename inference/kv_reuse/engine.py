"""Generation orchestration for first-call capture and later segmented prefill."""

import time

import torch
from transformers.models.qwen3_vl.kv_reuse import (
    build_a_reuse_state,
    build_ac_reuse_state,
    prefill_with_a_reuse,
    prefill_with_ac_reuse,
    prepare_qwen3_vl_position_ids,
    validate_a_reuse,
    validate_ac_reuse,
)

from .cache_manager import EpisodeCacheManager, parse_context
from .metrics import ModelStageTimer, synchronize
from .segments import locate_prompt_segments


class ACKVReuseEngine:
    def __init__(self, model, tokenizer, mode: str = "off", strict: bool = False):
        if mode not in ("off", "a", "ac"):
            raise ValueError("mode must be 'off', 'a', or 'ac'")
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.strict = strict
        self.cache = EpisodeCacheManager()

    def _full_generate(self, inputs, position_ids, generation_kwargs):
        return self.model.generate(
            **inputs,
            position_ids=position_ids,
            return_dict_in_generate=True,
            **generation_kwargs,
        )

    @torch.inference_mode()
    def generate(self, inputs, messages, context, generation_kwargs):
        started = time.perf_counter()
        cache_status = "disabled"
        miss_reason = None
        prompt_segments = None
        position_ids = rope_deltas = None
        c_position_delta_values = None
        output = None
        peak_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak_before = torch.cuda.memory_allocated()

        with ModelStageTimer(self.model) as stage_timer:
            if self.mode == "off":
                output = self.model.generate(return_dict_in_generate=True, **inputs, **generation_kwargs)
            else:
                try:
                    key, call_id = parse_context(context)
                    if context.get("kv_mode") not in (None, self.mode):
                        raise ValueError("kv_mode_mismatch")
                    prompt_segments = locate_prompt_segments(self.tokenizer, inputs["input_ids"], messages)
                    position_ids, rope_deltas = prepare_qwen3_vl_position_ids(self.model, inputs)
                    state = self.cache.state_for(key)
                    if state is None:
                        cache_status = "first"
                        output = self._full_generate(inputs, position_ids, generation_kwargs)
                        build_state = build_a_reuse_state if self.mode == "a" else build_ac_reuse_state
                        state = build_state(
                            self.model,
                            output.past_key_values,
                            inputs["input_ids"],
                            position_ids,
                            prompt_segments.model,
                        )
                        self.cache.set_first(key, call_id, state)
                    else:
                        validate_state = validate_a_reuse if self.mode == "a" else validate_ac_reuse
                        miss_reason = validate_state(
                            self.model, state, inputs["input_ids"], position_ids, prompt_segments.model
                        )
                        if miss_reason is not None:
                            raise ValueError(miss_reason)
                        cache_status = "hit"
                        synchronize()
                        assembly_started = time.perf_counter()
                        segmented_prefill = prefill_with_a_reuse if self.mode == "a" else prefill_with_ac_reuse
                        prefix = segmented_prefill(
                            self.model,
                            inputs,
                            position_ids,
                            rope_deltas,
                            prompt_segments.model,
                            state,
                        )
                        synchronize()
                        assembly_total = time.perf_counter() - assembly_started
                        kv_assembly_seconds = sum(
                            (
                                pair[0].elapsed_time(pair[1]) / 1000.0
                                if isinstance(pair[0], torch.cuda.Event)
                                else pair[1] - pair[0]
                            )
                            for pair in prefix.kv_assembly_timing
                        )
                        if self.mode == "ac":
                            c_delta = position_ids[1:, :, prompt_segments.model.c] - state.c_position_ids
                            c_position_delta_values = torch.unique(c_delta).detach().cpu().tolist()
                        output = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            position_ids=position_ids,
                            past_key_values=prefix.past_key_values,
                            return_dict_in_generate=True,
                            **generation_kwargs,
                        )
                except ValueError as error:
                    if self.strict:
                        raise
                    cache_status = "miss"
                    miss_reason = str(error)
                    if position_ids is None:
                        try:
                            position_ids, rope_deltas = prepare_qwen3_vl_position_ids(self.model, inputs)
                        except ValueError:
                            position_ids = None
                    if position_ids is None:
                        output = self.model.generate(return_dict_in_generate=True, **inputs, **generation_kwargs)
                    else:
                        output = self._full_generate(inputs, position_ids, generation_kwargs)

        synchronize()
        timing = stage_timer.summary()
        total = time.perf_counter() - started
        if cache_status == "hit":
            timing["segmented_prefill_total_seconds"] = assembly_total
            timing["kv_assembly_seconds"] = kv_assembly_seconds
        timing["total_generation_seconds"] = total
        peak_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if cache_status == "hit":
            computed_prefill_tokens = (
                prompt_segments.token_counts["P"]
                + prompt_segments.token_counts["B"]
                + prompt_segments.token_counts["S"]
                + (prompt_segments.token_counts["C"] if self.mode == "a" else 0)
            )
            directly_reused_tokens = prompt_segments.token_counts["A"]
            rope_corrected_reused_tokens = prompt_segments.token_counts["C"] if self.mode == "ac" else 0
        else:
            computed_prefill_tokens = inputs["input_ids"].shape[-1]
            directly_reused_tokens = 0
            rope_corrected_reused_tokens = 0
        metadata = {
            "reuse_mode": self.mode,
            "cache_status": cache_status,
            "cache_miss_reason": miss_reason,
            "segment_tokens": prompt_segments.token_counts if prompt_segments is not None else None,
            "computed_prefill_tokens": computed_prefill_tokens,
            "reused_tokens": directly_reused_tokens + rope_corrected_reused_tokens,
            "directly_reused_tokens": directly_reused_tokens,
            "rope_corrected_reused_tokens": rope_corrected_reused_tokens,
            "c_absolute_start": prompt_segments.model.c_start if prompt_segments is not None else None,
            "c_absolute_end": prompt_segments.model.c_end if prompt_segments is not None else None,
            "c_position_delta_values": c_position_delta_values,
            "timing": timing,
            "peak_memory_bytes": max(peak_before, peak_after),
        }
        return output.sequences, metadata

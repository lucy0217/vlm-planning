"""Teacher-forced candidate scoring; all caches are local to one request."""
import math
import time

import torch
from transformers import DynamicCache


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def encode_candidates(tokenizer, candidates):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidates must be a nonempty list')
    if any(not isinstance(text, str) or not text.strip() for text in candidates):
        raise ValueError('Each candidate must be a nonempty string')
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in candidates]
    if any(not ids for ids in encoded):
        raise ValueError('Each candidate must contain at least one token')
    return encoded


def clone_prefix_cache(cache, batch_size):
    # Never append into the shared prefix or share mutable batch storage.
    return DynamicCache([
        (layer.keys.repeat_interleave(batch_size, dim=0),
         layer.values.repeat_interleave(batch_size, dim=0))
        for layer in cache.layers
    ])


def token_scores(logits, labels, mask, score_function):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    scores = values.masked_fill(~mask.bool(), 0).sum(-1)
    if score_function == 'avg':
        scores = scores / mask.sum(-1)
    return scores


@torch.inference_mode()
def score_candidates(model, processor, messages, candidates, score_function='sum',
                     candidate_batch_size=8, max_context=16000):
    if score_function not in ('sum', 'avg'):
        raise ValueError('score_function must be sum or avg')
    if (isinstance(candidate_batch_size, bool) or
            not isinstance(candidate_batch_size, int) or not 1 <= candidate_batch_size <= 64):
        raise ValueError('candidate_batch_size must be an integer in 1..64')
    encoded = encode_candidates(processor.tokenizer, candidates)
    device = model.device
    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt').to(device)
    prefix_length = inputs.input_ids.shape[-1]
    if prefix_length + max(map(len, encoded)) > max_context:
        raise ValueError('Prompt plus longest candidate exceeds configured context limit')
    synchronize(device)
    prepared = time.perf_counter()
    base = model.model
    previous_delta = base.rope_deltas
    base.rope_deltas = None
    try:
        prefix_output = model(**inputs, use_cache=True, logits_to_keep=1)
        prefix_cache = prefix_output.past_key_values
        first_logits = prefix_output.logits[:, -1:, :]
        delta = base.rope_deltas
        if delta is None:
            delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            delta = delta.clone()
        synchronize(device)
        prefilled = time.perf_counter()
        records = []
        pad = processor.tokenizer.pad_token_id
        if pad is None:
            pad = processor.tokenizer.eos_token_id
        if pad is None:
            pad = 0
        for start in range(0, len(encoded), candidate_batch_size):
            group = encoded[start:start + candidate_batch_size]
            batch, width = len(group), max(map(len, group))
            ids = torch.full((batch, width), pad, dtype=torch.long, device=device)
            suffix_mask = torch.zeros_like(ids)
            for row, tokens in enumerate(group):
                ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
                suffix_mask[row, :len(tokens)] = 1
            attention_mask = torch.cat([
                inputs.attention_mask.expand(batch, -1), suffix_mask], dim=-1)
            positions = (torch.arange(prefix_length, prefix_length + width, device=device)
                         .view(1, -1).expand(batch, -1) + delta.reshape(1, 1))
            positions = positions.unsqueeze(0).expand(3, -1, -1)
            cache = clone_prefix_cache(prefix_cache, batch)
            output = model(
                input_ids=ids, attention_mask=attention_mask, position_ids=positions,
                past_key_values=cache, use_cache=True)
            logits = torch.cat([first_logits.expand(batch, -1, -1),
                                output.logits[:, :-1, :]], dim=1)
            scores = token_scores(logits, ids, suffix_mask, score_function).tolist()
            for row, score in enumerate(scores):
                if not math.isfinite(score):
                    raise RuntimeError('Candidate scoring produced a non-finite score')
                index = start + row
                records.append(dict(index=index, candidate=candidates[index],
                                    score=score, token_count=len(group[row])))
            del cache, output, logits
        synchronize(device)
        finished = time.perf_counter()
        best = max(range(len(records)), key=lambda i: records[i]['score'])
        return dict(scores=records, best_index=best, score_function=score_function,
                    usage=dict(prompt_tokens=prefix_length,
                               candidate_tokens=sum(map(len, encoded))),
                    server_timing=dict(preprocessing_seconds=prepared-started,
                                       prefill_seconds=prefilled-prepared,
                                       scoring_seconds=finished-prefilled))
    finally:
        base.rope_deltas = previous_delta

"""Map EmbodiedBench's character-defined A/C prompt segments to model tokens."""

from dataclasses import dataclass

import torch
from transformers.models.qwen3_vl.kv_reuse import Qwen3VLACSegments


A_MARKER = "## Now the human instruction is:"
FIRST_B_MARKER = " You are supposed to output in json."
HISTORY_B_MARKER = "\n\n The action history:"
NORMALIZED_HISTORY_B_MARKER = " The action history:"
C_MARKER = "The output json format should be"


@dataclass(frozen=True)
class PromptSegments:
    model: Qwen3VLACSegments
    character_spans: dict[str, tuple[int, int]]

    @property
    def token_counts(self) -> dict[str, int]:
        value = self.model
        return {
            "P": value.a_start,
            "A": value.a_end - value.a_start,
            "B": value.c_start - value.a_end,
            "C": value.c_end - value.c_start,
            "S": value.sequence_length - value.c_end,
        }


def _unique_find(text: str, marker: str) -> int:
    positions = []
    start = 0
    while (position := text.find(marker, start)) >= 0:
        positions.append(position)
        start = position + 1
    if len(positions) != 1:
        raise ValueError(f"Expected one {marker!r} marker, found {len(positions)}")
    return positions[0]


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    matches = [
        start
        for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one raw prompt token span, found {len(matches)}")
    return matches[0]


def _prompt_text(messages: list[dict]) -> str:
    parts = [
        part["text"]
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "text"
    ]
    if len(parts) != 1:
        raise ValueError(f"A/C reuse requires exactly one text part, found {len(parts)}")
    return parts[0]


def locate_prompt_segments(tokenizer, input_ids: torch.Tensor, messages: list[dict]) -> PromptSegments:
    """Locate P/A/B/C/S in the exact processor-produced input IDs."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("A/C reuse requires batch size one")
    text = _prompt_text(messages)
    instruction_start = _unique_find(text, A_MARKER)
    c_start = _unique_find(text, C_MARKER)
    history_position = text.find(HISTORY_B_MARKER, instruction_start)
    if history_position < 0:
        history_position = text.find(NORMALIZED_HISTORY_B_MARKER, instruction_start)
    b_positions = [text.find(FIRST_B_MARKER, instruction_start), history_position]
    b_positions = [position for position in b_positions if position >= 0]
    if len(b_positions) != 1:
        raise ValueError(f"Expected exactly one first-call/history B marker, found {len(b_positions)}")
    a_end = b_positions[0]
    if not text[:a_end].rstrip().endswith("."):
        raise ValueError("A must end at the human-instruction period")
    if not 0 < a_end <= c_start < len(text):
        raise ValueError("Character-defined A/B/C segments are not ordered")

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    raw_ids = list(encoded.input_ids)
    absolute_start = _find_subsequence(input_ids[0].tolist(), raw_ids)
    character_spans = {"A": (0, a_end), "C": (c_start, len(text))}
    token_spans = {}
    for name, (begin, end) in character_spans.items():
        local = [
            index
            for index, (left, right) in enumerate(encoded.offset_mapping)
            if right > begin and left < end and right > left
        ]
        if not local:
            raise ValueError(f"{name} contains no tokens")
        covered_begin = encoded.offset_mapping[local[0]][0]
        covered_end = encoded.offset_mapping[local[-1]][1]
        if (text[covered_begin:begin] if covered_begin < begin else "").strip() or (
            text[end:covered_end] if covered_end > end else ""
        ).strip():
            raise ValueError(f"{name} boundary cuts through a tokenizer token")
        token_spans[name] = (absolute_start + local[0], absolute_start + local[-1] + 1)

    segments = Qwen3VLACSegments(
        a_start=token_spans["A"][0],
        a_end=token_spans["A"][1],
        c_start=token_spans["C"][0],
        c_end=token_spans["C"][1],
        sequence_length=input_ids.shape[-1],
    )
    segments.validate()
    return PromptSegments(model=segments, character_spans=character_spans)

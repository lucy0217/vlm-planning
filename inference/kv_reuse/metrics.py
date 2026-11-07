"""Timing and JSONL logging for the A/C reuse experiment."""

import json
import time
from pathlib import Path

import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ModelStageTimer:
    """Classify vision, multi-token prefill, and one-token decode calls."""

    def __init__(self, model):
        self.model = model
        self.records = []
        self.handles = []

    def __enter__(self):
        pending = {}

        def pre(name):
            def hook(module, args, kwargs):
                inputs = kwargs.get("inputs_embeds")
                sequence_length = inputs.shape[1] if inputs is not None and inputs.ndim == 3 else None
                if torch.cuda.is_available():
                    started = torch.cuda.Event(enable_timing=True)
                    started.record()
                else:
                    started = time.perf_counter()
                pending[id(module)] = (name, sequence_length, started)

            return hook

        def post(module, args, kwargs, output):
            name, sequence_length, started = pending.pop(id(module))
            if torch.cuda.is_available():
                finished = torch.cuda.Event(enable_timing=True)
                finished.record()
                self.records.append((name, sequence_length, started, finished))
            else:
                self.records.append((name, sequence_length, time.perf_counter() - started))

        self.handles.append(self.model.model.visual.register_forward_pre_hook(pre("vision"), with_kwargs=True))
        self.handles.append(self.model.model.visual.register_forward_hook(post, with_kwargs=True))
        language = self.model.model.language_model
        self.handles.append(language.register_forward_pre_hook(pre("language"), with_kwargs=True))
        self.handles.append(language.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()

    def summary(self) -> dict[str, float]:
        synchronize()
        records = []
        for record in self.records:
            if len(record) == 4:
                name, length, started, finished = record
                records.append((name, length, started.elapsed_time(finished) / 1000.0))
            else:
                records.append(record)
        return {
            "vision_seconds": sum(duration for name, _, duration in records if name == "vision"),
            "prefill_seconds": sum(
                duration for name, length, duration in records if name == "language" and length != 1
            ),
            "decode_seconds": sum(
                duration for name, length, duration in records if name == "language" and length == 1
            ),
        }


class JSONLMetricsWriter:
    """Write metrics to a fixed legacy path or the run directory in a request.

    Dynamic paths are restricted to ``runs_root`` so an HTTP client cannot make
    the local service write arbitrary files elsewhere on the server.
    """

    def __init__(self, path: str | None, runs_root: str | Path | None = None):
        self.path = Path(path).resolve() if path else None
        self.runs_root = Path(runs_root).resolve() if runs_root else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, context: dict | None = None) -> Path | None:
        if self.path is not None:
            return self.path
        run_dir = (context or {}).get("run_dir")
        if not run_dir:
            return None
        if self.runs_root is None:
            raise ValueError("Dynamic metrics logging is not configured")
        resolved = Path(run_dir).resolve()
        try:
            resolved.relative_to(self.runs_root)
        except ValueError as error:
            raise ValueError(f"run_dir must be inside {self.runs_root}") from error
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved / "server_metrics.jsonl"

    def write(self, record: dict, context: dict | None = None) -> None:
        path = self.resolve_path(context)
        if path is None:
            return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

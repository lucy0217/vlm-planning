"""Single-episode canonical cache ownership and request-context validation."""

from dataclasses import dataclass


CONTEXT_FIELDS = ("run_id", "benchmark", "eval_set", "episode", "call_id")


@dataclass(frozen=True)
class EpisodeKey:
    run_id: str
    benchmark: str
    eval_set: str
    episode: str


@dataclass
class CacheEntry:
    key: EpisodeKey
    state: object
    first_call_id: str


def parse_context(context) -> tuple[EpisodeKey, str]:
    if not isinstance(context, dict):
        raise ValueError("missing_context")
    missing = [field for field in CONTEXT_FIELDS if field not in context]
    if missing:
        raise ValueError("missing_context_fields:" + ",".join(missing))
    key = EpisodeKey(*(str(context[field]) for field in CONTEXT_FIELDS[:-1]))
    return key, str(context["call_id"])


class EpisodeCacheManager:
    """Keep only the first-call canonical cache for the active episode."""

    def __init__(self):
        self.entry: CacheEntry | None = None

    def state_for(self, key: EpisodeKey):
        if self.entry is None:
            return None
        if self.entry.key != key:
            self.clear()
            return None
        return self.entry.state

    def set_first(self, key: EpisodeKey, call_id: str, state) -> None:
        if self.entry is None:
            self.entry = CacheEntry(key=key, state=state, first_call_id=call_id)
        elif self.entry.key != key:
            self.clear()
            self.entry = CacheEntry(key=key, state=state, first_call_id=call_id)

    def clear(self) -> None:
        self.entry = None

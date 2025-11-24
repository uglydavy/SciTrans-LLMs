from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple


@dataclass
class MemoryEntry:
    source: str
    translation: str


class TranslationMemory:
    """Lightweight in-process translation memory.

    The memory keeps the most recent segments so prompts can reference
    prior choices (terminology, style, acronyms). It does not persist to
    disk; callers may serialize ``snapshot()`` if long-lived caching is
    desired.
    """

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._entries: Deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, source: str, translation: str) -> None:
        if not source or not translation:
            return
        self._entries.append(MemoryEntry(source=source, translation=translation))

    def snapshot(self) -> List[Tuple[str, str]]:
        return [(e.source, e.translation) for e in list(self._entries)]

    def contextual_prompt(self, limit: int = 6) -> str:
        if not self._entries:
            return ""
        recent = list(self._entries)[-limit:]
        joined = "\n".join(f"- {src} -> {tgt}" for src, tgt in [(e.source, e.translation) for e in recent])
        return (
            "Reuse these prior translations for consistency (do not retranslate them, just honor the terminology):\n"
            f"{joined}\n"
        )

    def glossary_candidates(self, min_len: int = 12) -> Iterable[Tuple[str, str]]:
        for entry in self._entries:
            if len(entry.source) >= min_len:
                yield (entry.source.lower(), entry.translation)

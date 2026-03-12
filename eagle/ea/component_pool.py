from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json
import random


@dataclass
class ComponentPool:
    """
    Pool of candidate prompt components.

    Example structure:
    {
        "game_rules": [...],
        "format_instructions": [...],
        "strategy": [...],
        ...
    }
    """

    pools: Dict[str, List[str]] = field(default_factory=dict)
    max_keep_per_type: int = 50

    @staticmethod
    def _stringify_candidate(value: Any) -> str:
        """
        Convert a nested JSON value into one prompt component string.
        """
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            parts = [ComponentPool._stringify_candidate(item) for item in value]
            return "\n".join(part for part in parts if part).strip()

        if isinstance(value, dict):
            parts = [ComponentPool._stringify_candidate(item) for item in value.values()]
            return "\n\n".join(part for part in parts if part).strip()

        return str(value).strip()

    @classmethod
    def _normalize_candidates(cls, value: Any) -> List[str]:
        """
        Normalize one JSON pool entry into a list of plain string candidates.
        """
        if isinstance(value, dict):
            grouped_candidates = {
                key: cls._normalize_candidates(subvalue)
                for key, subvalue in value.items()
            }
            if not grouped_candidates:
                return []

            candidate_count = max(len(candidates) for candidates in grouped_candidates.values())
            combined_candidates: List[str] = []

            for index in range(candidate_count):
                parts: List[str] = []
                for candidates in grouped_candidates.values():
                    if not candidates:
                        continue
                    parts.append(candidates[index % len(candidates)])

                text = "\n\n".join(part for part in parts if part).strip()
                if text:
                    combined_candidates.append(text)

            return combined_candidates

        if isinstance(value, list):
            candidates = [cls._stringify_candidate(item) for item in value]
            return [candidate for candidate in candidates if candidate]

        text = cls._stringify_candidate(value)
        return [text] if text else []

    @classmethod
    def from_json(cls, path: str | Path) -> "ComponentPool":
        """
        Load component pools from a JSON file.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        normalized = {
            key: cls._normalize_candidates(value)
            for key, value in data.items()
        }
        return cls(pools=normalized)

    def has_type(self, key: str) -> bool:
        """
        Return True when the component type exists and is non-empty.
        """
        return key in self.pools and len(self.pools[key]) > 0

    def sample(self, key: str) -> str:
        """
        Sample one candidate component from a given type.
        """
        candidates = self.pools.get(key, [])
        if not candidates:
            raise ValueError(f"No candidates found for component type: {key}")
        return random.choice(candidates)

    def add(self, key: str, text: str) -> None:
        """
        Add a new component candidate to the pool.

        This is useful when you later want to keep good mutated components
        as reusable building blocks.
        """
        self.pools.setdefault(key, []).append(text)

        # Keep the pool bounded to avoid uncontrolled growth.
        if len(self.pools[key]) > self.max_keep_per_type:
            self.pools[key] = random.sample(self.pools[key], self.max_keep_per_type)
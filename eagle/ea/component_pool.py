"""
Component pool for managing prompt components in the evolutionary algorithm.
This module defines the ComponentPool class, which loads and organizes prompt components from a JSON file. The ComponentPool provides access to different categories of components, such as combat policies, economy policies, production policies, worker behaviors, targeting policies, and movement policies. These components are used by the evolutionary algorithm to construct and evolve effective prompts for guiding agent behavior in MicroRTS.
"""

from __future__ import annotations

import json
from typing import Dict, List

class ComponentPool:
    """
        A pool of prompt components loaded from a JSON file.

        The JSON file should have the following structure:
        {
            "role": [ [ ... ] ],
            "critical_rules": [ [ ... ] ],
            "actions": [ [ ... ] ],
            "json_schema": [ [ ... ] ],
            "field_requirements": [ [ ... ] ],
            "examples": [ [ ... ] ],
            "strategy": {
                "global_strategy": [ [ ... ] ],
                "combat_policy": [ [ ... ] ],
                "economy_policy": [ [ ... ] ],
                "production_policy": [ [ ... ] ],
                "worker_policy": [ [ ... ] ],
                "targeting_policy": [ [ ... ] ],
                "movement_policy": [ [ ... ] ]
            }
        }
    """
    
    def __init__(self, components: Dict[str, List[List[str]]]):
        self.components = components
        # components keys
        self.component_keys = list(self.components.keys())
        self.strategy_keys = list(self.components["strategy"].keys())
        

    @classmethod
    def from_json(cls, filepath: str) -> ComponentPool:
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(data)

    def has_category(self, category: str) -> bool:
        return category in self.components and bool(self.components[category])
    
    def get_component(self, category: str, index: int) -> List[str]:
        if category not in self.components:
            raise KeyError(f"Component category not found: {category}")
        candidates = self.components[category]
        if not candidates:
            raise ValueError(f"No candidates found for component category: {category}")
        if index < 0 or index >= len(candidates):
            raise IndexError(
                f"Component index out of range for '{category}': {index} (valid: 0..{len(candidates)-1})"
            )
        return candidates[index]
    
    def get_strategy_component(self, strategy: str, index: int) -> List[str]:
        candidates = self.components["strategy"][strategy]
        if not candidates:
            raise ValueError(f"No candidates found for strategy category: {strategy}")
        if index < 0 or index >= len(candidates):
            raise IndexError(
                f"Strategy index out of range for '{strategy}': {index} (valid: 0..{len(candidates)-1})"
            )
        return candidates[index]

    def get_random_strategy_component_index(self, strategy: str) -> int:
        import random
        candidates = self.components["strategy"][strategy]
        if not candidates:
            raise ValueError(f"No candidates found for strategy category: {strategy}")
        return random.randint(0, len(candidates) - 1)
    
    def get_random_component_index(self, category: str) -> int:
        import random
        if category not in self.components:
            raise KeyError(f"Component category not found: {category}")
        candidates = self.components[category]
        if not candidates:
            raise ValueError(f"No candidates found for component category: {category}")
        return random.randint(0, len(candidates) - 1)
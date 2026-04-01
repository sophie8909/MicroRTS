"""
Individual class for representing a candidate solution in the genetic algorithm.
"""

from __future__ import annotations
import ast
import itertools
from dataclasses import dataclass
from typing import Any
from .component_pool import ComponentPool
from .fitness_utils import DEFAULT_FITNESS


@dataclass(frozen=True)
class ComponentEntry:
    name: str
    value: Any


class Individual:
    _id_counter = itertools.count()

    def __init__(self, 
                 critical_rules: int = 0, 
                 actions: int = 0, 
                 json_schema: int = 0, 
                 field_requirements: int = 0, 
                 examples: int = 0, 
                 strategy: dict[str, int] | None = None,
                 role: int = 0):
        self.id = f"ind-{next(self._id_counter)}"
        # use indices to reference components in the component pool
        # stable components (only one option, not evolved)
        self.stable_components = [actions, json_schema, field_requirements, examples]
        self.actions = actions
        self.json_schema = json_schema
        self.field_requirements = field_requirements
        self.examples = examples
        # evolving components
        self.evolving_components = [critical_rules, role]
        self.critical_rules = critical_rules
        self.role = role
        self.strategy = self._normalize_strategy(strategy)

        # fitness = [win_score, number_of_turns_score, game_round_score]
        self.fitness = DEFAULT_FITNESS.copy()  # [win_score, turn_score, game_round_score]
        # self.fitness = 0.0  # Initialize fitness with a single value for simplicity
    
    @property
    def components(self) -> list[ComponentEntry]:
        # Backward-compatible view for legacy code expecting ind.components.
        strategy_items = tuple(sorted((self.strategy or {}).items()))
        return [
            ComponentEntry("role", self.role),
            ComponentEntry("critical_rules", self.critical_rules),
            ComponentEntry("actions", self.actions),
            ComponentEntry("json_schema", self.json_schema),
            ComponentEntry("field_requirements", self.field_requirements),
            ComponentEntry("examples", self.examples),
            ComponentEntry("strategy", strategy_items),
        ]

    def __repr__(self):
        return f"Individual(role={self.role}, critical_rules={self.critical_rules}, actions={self.actions}, json_schema={self.json_schema}, field_requirements={self.field_requirements}, examples={self.examples}, strategy={self.strategy})"

    @staticmethod
    def _normalize_strategy(strategy: dict[str, int] | str | None) -> dict[str, int]:
        if strategy is None:
            return {}
        if isinstance(strategy, dict):
            return strategy.copy()
        if isinstance(strategy, str):
            try:
                parsed = ast.literal_eval(strategy)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid strategy string: {strategy!r}") from exc
            if isinstance(parsed, dict):
                return parsed.copy()
        raise TypeError(f"strategy must be a dict, stringified dict, or None; got {type(strategy).__name__}")
    
    def initialize_randomly(self, component_pool: ComponentPool):
        # Initialize the individual's components randomly from the component pool
        def _safe_index(category: str, default: int = 0) -> int:
            if component_pool.has_category(category):
                return component_pool.get_random_component_index(category)
            return default

        self.role = _safe_index('role')
        self.examples = _safe_index('examples')
        self.strategy = {
            strategy_key: component_pool.get_random_strategy_component_index(strategy_key)
            for strategy_key in component_pool.strategy_keys
        }

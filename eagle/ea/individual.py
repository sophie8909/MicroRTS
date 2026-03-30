"""
Individual class for representing a candidate solution in the genetic algorithm.
"""

from __future__ import annotations
from .component_pool import ComponentPool

class Individual:
    def __init__(self, 
                 critical_rules: int = 0, 
                 actions: int = 0, 
                 json_schema: int = 0, 
                 field_requirements: int = 0, 
                 examples: int = 0, 
                 strategy: list[int] = None, 
                 role: int = 0):
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
        if strategy is None:
            strategy = [0] * 8  # Assuming 8 strategy components
        self.strategy = strategy

        # fitness = [win_score, number_of_turns_score, game_round_score]
        # self.fitness = [0.0, 0.0, 0.0]  # Initialize fitness with default values
        self.fitness = 0.0  # Initialize fitness with a single value for simplicity
    
    def __repr__(self):
        return f"Individual(role={self.role}, critical_rules={self.critical_rules}, actions={self.actions}, json_schema={self.json_schema}, field_requirements={self.field_requirements}, examples={self.examples}, strategy={self.strategy})"
    
    def initialize_randomly(self, component_pool: ComponentPool):
        # Initialize the individual's components randomly from the component pool
        def _safe_index(category: str, default: int = 0) -> int:
            if component_pool.has_category(category):
                return component_pool.get_random_component_index(category)
            return default

        self.role = _safe_index('role')
        self.examples = _safe_index('examples')
        self.strategy = [
            component_pool.get_random_strategy_component_index(strategy_key)
            for strategy_key in component_pool.strategy_keys
        ]

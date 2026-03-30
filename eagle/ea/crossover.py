"""Crossover methods for the genetic algorithm."""

from __future__ import annotations

from .component_pool import ComponentPool
from .individual import Individual

class Crossover:
    @staticmethod
    def uniform_crossover(component_pool: ComponentPool, parent1: Individual, parent2: Individual) -> Individual:
        import random
        child = Individual()
        p1_strategy = parent1.strategy or {}
        p2_strategy = parent2.strategy or {}

        child.role = random.choice([parent1.role, parent2.role])
        child.critical_rules = random.choice([parent1.critical_rules, parent2.critical_rules])
        child.actions = random.choice([parent1.actions, parent2.actions])
        child.json_schema = random.choice([parent1.json_schema, parent2.json_schema])
        child.field_requirements = random.choice([parent1.field_requirements, parent2.field_requirements])
        child.examples = random.choice([parent1.examples, parent2.examples])

        child.strategy = {}
        for strategy_key in component_pool.strategy_keys:
            if strategy_key in p1_strategy and strategy_key in p2_strategy:
                child.strategy[strategy_key] = random.choice(
                    [p1_strategy[strategy_key], p2_strategy[strategy_key]]
                )
            elif strategy_key in p1_strategy:
                child.strategy[strategy_key] = p1_strategy[strategy_key]
            elif strategy_key in p2_strategy:
                child.strategy[strategy_key] = p2_strategy[strategy_key]
            else:
                child.strategy[strategy_key] = component_pool.get_random_strategy_component_index(strategy_key)
        return child
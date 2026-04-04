"""Crossover methods for the genetic algorithm."""

from __future__ import annotations

from .component_pool import ComponentPool
from .individual import Individual
from .llm import LLM
class Crossover:
    @staticmethod
    def uniform_crossover(component_pool: ComponentPool, parent1: Individual, parent2: Individual) -> Individual:
        import random
        child = Individual()
        p1_strategy = parent1.strategy or {}
        p2_strategy = parent2.strategy or {}
        child.game_rule = parent1.game_rule

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
    
    def llm_crossover(component_pool: ComponentPool, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual()
        instruction = "Combine the following components from two parent individuals to create a child individual. " \
                        "Ensure the child maintains coherence and incorporates key elements from both parents."
        child.game_rule = parent1.game_rule

        child.strategy = {}
        for strategy_key in component_pool.strategy_keys:
            p1_component = parent1.strategy.get(strategy_key, "")
            p2_component = parent2.strategy.get(strategy_key, "")
            child.strategy[strategy_key] = LLM.ollama_combine_components(p1_component, p2_component, instruction)
       
        return child

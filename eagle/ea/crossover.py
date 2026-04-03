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
    
    def llm_crossover(component_pool: ComponentPool, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual()
        instruction = "Combine the following components from two parent individuals to create a child individual. " \
                        "Ensure the child maintains coherence and incorporates key elements from both parents."
        child.role = LLM.ollama_combine_components(parent1.role, parent2.role, instruction)
        child.critical_rules = LLM.ollama_combine_components(parent1.critical_rules, parent2.critical_rules, instruction)
        child.actions = LLM.ollama_combine_components(parent1.actions, parent2.actions, instruction)
        child.json_schema = LLM.ollama_combine_components(parent1.json_schema, parent2.json_schema, instruction)
        child.field_requirements = LLM.ollama_combine_components(parent1.field_requirements, parent2.field_requirements, instruction)
        child.examples = LLM.ollama_combine_components(parent1.examples, parent2.examples, instruction)

        child.strategy = {}
        for strategy_key in component_pool.strategy_keys:
            p1_component = parent1.strategy.get(strategy_key, "")
            p2_component = parent2.strategy.get(strategy_key, "")
            child.strategy[strategy_key] = LLM.ollama_combine_components(p1_component, p2_component, instruction)
       
        return child
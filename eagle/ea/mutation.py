"""Mutation methods for the genetic algorithm.
"""

from __future__ import annotations
from .individual import Individual
from .component_pool import ComponentPool

class Mutation:
    @staticmethod
    def mutate_component(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
        import random
        mutated_individual = Individual(
            role=individual.role,
            critical_rules=individual.critical_rules,
            actions=individual.actions,
            json_schema=individual.json_schema,
            field_requirements=individual.field_requirements,
            examples=individual.examples,
            strategy=individual.strategy.copy()
        )
        
        if random.random() < mutation_rate:
            mutated_individual.role = component_pool.get_random_component_index('role')
        
        if random.random() < mutation_rate:
            mutated_individual.critical_rules = component_pool.get_random_component_index('critical_rules')
        
        if component_pool.has_category('actions') and random.random() < mutation_rate:
            mutated_individual.actions = component_pool.get_random_component_index('actions')
        
        if component_pool.has_category('json_schema') and random.random() < mutation_rate:
            mutated_individual.json_schema = component_pool.get_random_component_index('json_schema')
        
        if component_pool.has_category('field_requirements') and random.random() < mutation_rate:
            mutated_individual.field_requirements = component_pool.get_random_component_index('field_requirements')
        
        if component_pool.has_category('examples') and random.random() < mutation_rate:
            mutated_individual.examples = component_pool.get_random_component_index('examples')
        
        for i in range(len(mutated_individual.strategy)):
            if random.random() < mutation_rate:
                strategy_key = component_pool.strategy_keys[i]
                mutated_individual.strategy[i] = component_pool.get_random_strategy_component_index(strategy_key)
        
        return mutated_individual
"""Mutation methods for the genetic algorithm.
"""

from __future__ import annotations
import time
from .individual import Individual
from .component_pool import ComponentPool
from .llm import LLM

class Mutation:
    
    @staticmethod
    def mutate_component_from_pool(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
        import random
        base_strategy = dict(individual.strategy or {})
        mutated_individual = Individual(
            role=individual.role,
            critical_rules=individual.critical_rules,
            actions=individual.actions,
            json_schema=individual.json_schema,
            field_requirements=individual.field_requirements,
            examples=individual.examples,
            strategy=base_strategy.copy(),
        )
        
        # mutate evolving components
        if random.random() < mutation_rate:
            mutated_individual.role = component_pool.get_random_component_index('role')
        if random.random() < mutation_rate:
            mutated_individual.critical_rules = component_pool.get_random_component_index('critical_rules')

        for strategy_key in component_pool.strategy_keys:
            if random.random() < mutation_rate:
                # add or replace strategy component
                if random.random() < 0.5:
                    mutated_individual.strategy[strategy_key] = component_pool.get_random_strategy_component_index(strategy_key)
                # remove strategy component
                else:
                    if strategy_key in mutated_individual.strategy:
                        del mutated_individual.strategy[strategy_key]

        return mutated_individual
    
    @staticmethod
    def rewrite_component_with_LLM(component: str, rewrite_instruction: str) -> tuple[str, float]:
        start = time.perf_counter()
        rewritten_role_component = LLM.ollama_rewrite_component(
            original_text=component,
            instruction=rewrite_instruction,
            model="llama3.1:8b",
        )
        elapsed = time.perf_counter() - start
        return rewritten_role_component, elapsed



    @staticmethod
    def mutate_component_LLM(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
        # using LLM rewrite

        import random
        base_strategy = dict(individual.strategy or {})
        mutated_individual = Individual(
            role=individual.role,
            critical_rules=individual.critical_rules,
            actions=individual.actions,
            json_schema=individual.json_schema,
            field_requirements=individual.field_requirements,
            examples=individual.examples,
            strategy=base_strategy.copy(),
        )
        mutated_individual.ea_llm_call_time = 0.0

        # LLM rewrite for role
        if random.random() < mutation_rate:
            # LLM rewrite for role
            rewrite_instruction = (
                "Make this role description clearer and slightly more directive "
                "for a MicroRTS agent, while preserving its original function."
            )
            original_role_component_str = component_pool.get_component_str('role', individual.role)
            rewritten_role_component_str, elapsed = Mutation.rewrite_component_with_LLM(original_role_component_str, rewrite_instruction)
            mutated_individual.ea_llm_call_time += elapsed
            rewritten_role_component = component_pool.parse_component_str(rewritten_role_component_str)

            new_role_index = component_pool.add_component('role', rewritten_role_component)
            mutated_individual.role = new_role_index

        # LLM rewrite for critical_rules
        if random.random() < mutation_rate:
            # LLM rewrite for critical_rules
            rewrite_instruction = (
                "Enhance this critical rule for a MicroRTS agent, making it more robust and effective "
                "in handling complex game scenarios while maintaining its core functionality."
            )
            original_critical_rules_component_str = component_pool.get_component_str('critical_rules', individual.critical_rules)
            rewritten_critical_rules_component_str, elapsed = Mutation.rewrite_component_with_LLM(original_critical_rules_component_str, rewrite_instruction)
            mutated_individual.ea_llm_call_time += elapsed
            rewritten_critical_rules_component = component_pool.parse_component_str(rewritten_critical_rules_component_str)

            new_critical_rules_index = component_pool.add_component('critical_rules', rewritten_critical_rules_component)
            mutated_individual.critical_rules = new_critical_rules_index

        rewrite_stragey_list = [
            "Rewrite this strategy",
            "Make this strategy more aggressive",
            "Make this strategy more defensive",
        ]

        # Strategy components mutation with LLM rewrite
        for i, strategy_key in enumerate(component_pool.strategy_keys):
            if random.random() < mutation_rate:
                rewrite_instruction = random.choice(rewrite_stragey_list)
                if strategy_key in base_strategy:
                    original_strategy_component_str = "\n".join(
                        component_pool.get_strategy_component(strategy_key, base_strategy[strategy_key])
                    )
                    rewritten_strategy_component_str, elapsed = Mutation.rewrite_component_with_LLM(original_strategy_component_str, rewrite_instruction)
                    mutated_individual.ea_llm_call_time += elapsed
                    rewritten_strategy_component = component_pool.parse_component_str(rewritten_strategy_component_str)
                    new_strategy_index = component_pool.add_strategy_component(strategy_key, rewritten_strategy_component)
                    mutated_individual.strategy[strategy_key] = new_strategy_index

        return mutated_individual

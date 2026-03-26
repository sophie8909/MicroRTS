"""Mutation methods for the genetic algorithm.
"""

from __future__ import annotations
from .individual import Individual
from .component_pool import ComponentPool
from .llm import LLM

class Mutation:
    
    @staticmethod
    def mutate_component_from_pool(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
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
        
        # mutate evolving components
        if random.random() < mutation_rate:
            mutated_individual.role = component_pool.get_random_component_index('role')
        if random.random() < mutation_rate:
            mutated_individual.critical_rules = component_pool.get_random_component_index('critical_rules')

        for i in range(len(mutated_individual.strategy)):
            if random.random() < mutation_rate:
                strategy_key = component_pool.strategy_keys[i]
                mutated_individual.strategy[i] = component_pool.get_random_strategy_component_index(strategy_key)
        
        return mutated_individual
    
    @staticmethod
    def rewrite_component_with_LLM(component: str, rewrite_instruction: str) -> str:
        
        rewritten_role_component = LLM.ollama_rewrite_component(
            original_text=component,
            instruction=rewrite_instruction,
            model="llama3.1:8b",
        )
        return rewritten_role_component 



    @staticmethod
    def mutate_component_LLM(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
        # using LLM rewrite

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

        # LLM rewrite for role
        if random.random() < mutation_rate:
            # LLM rewrite for role
            rewrite_instruction = (
                "Make this role description clearer and slightly more directive "
                "for a MicroRTS agent, while preserving its original function."
            )
            original_role_component_str = component_pool.get_component_str('role', individual.role)
            rewritten_role_component_str = Mutation.rewrite_component_with_LLM(original_role_component_str, rewrite_instruction)
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
            rewritten_critical_rules_component_str = Mutation.rewrite_component_with_LLM(original_critical_rules_component_str, rewrite_instruction)
            rewritten_critical_rules_component = component_pool.parse_component_str(rewritten_critical_rules_component_str)

            new_critical_rules_index = component_pool.add_component('critical_rules', rewritten_critical_rules_component)
            mutated_individual.critical_rules = new_critical_rules_index

        rewrite_stragey_list = [
            "Rewrite this strategy",
            "Make this strategy more aggressive",
            "Make this strategy more defensive",
        ]

        # Strategy components mutation with LLM rewrite
        rewrite_instruction = random.choice(rewrite_stragey_list)
        for i in range(len(mutated_individual.strategy)):
            if random.random() < mutation_rate:
                strategy_key = component_pool.strategy_keys[i]
                original_strategy_component_str = "\n".join(
                    component_pool.get_strategy_component(strategy_key, individual.strategy[i])
                )
                rewritten_strategy_component_str = Mutation.rewrite_component_with_LLM(original_strategy_component_str, rewrite_instruction)
                rewritten_strategy_component = component_pool.parse_component_str(rewritten_strategy_component_str)
                new_strategy_index = component_pool.add_strategy_component(strategy_key, rewritten_strategy_component)
                mutated_individual.strategy[i] = new_strategy_index

        return mutated_individual

    def modify_strategy_component(individual: Individual, component_pool: ComponentPool, mutation_rate: float) -> Individual:
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

        # remove the strategy component
        if random.random() < 0.5:
            random_strategy_index = random.randint(0, len(mutated_individual.strategy)-1)
            # remove
            mutated_individual.strategy.remove(random_strategy_index)
        else:
            # add a new strategy component
            random_strategy_key = random.choice(component_pool.strategy_keys)
            new_strategy_index = component_pool.get_random_strategy_component_index(random_strategy_key)
            mutated_individual.strategy[random_strategy_key].append(new_strategy_index)
        return mutated_individual
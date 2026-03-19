"""
Genetic Algorithm implementation for evolving prompt components.
This module defines the GA class, which implements a genetic algorithm to optimize prompt components for guiding agent behavior in MicroRTS. The GA class initializes a population of candidate solutions, evaluates their fitness based on performance in MicroRTS, and applies selection, crossover, and mutation operations to evolve better solutions over multiple generations. The GA can be configured with various parameters such as population size, number of generations, mutation rate, and selection method.
"""

from __future__ import annotations

import random
from typing import List

from .evaluate import Evaluator
from .component_pool import ComponentPool
from .individual import Individual
from .config import EAConfig
from .parent_selection import ParentSelection
from .crossover import Crossover
from .mutation import Mutation
from .environment_selection import EnvironmentSelection

class GA:
    def __init__(self, config: EAConfig, component_pool: ComponentPool):
        self.config = config
        self.component_pool = component_pool
        self.population = self.initialize_population()
    
    def initialize_population(self) -> List[Individual]:
        # Initialize a population of random solutions based on the component pool
        individuals = []
        for _ in range(self.config.population_size):
            individual = Individual() 
            individual.initialize_randomly(self.component_pool)
            individuals.append(individual)
        return individuals

    def evaluate_fitness(self, individual: Individual) -> float:
        # Evaluate the fitness of a solution by running it in MicroRTS and measuring performance
        evaluator = Evaluator(self.component_pool)
        fitness = evaluator.evaluate(individual)
        individual.fitness = fitness  # Store the fitness in the individual for later use
        return fitness
    
    def select_parents(self) -> List[Individual]:
        # Select parents from the population using the configured selection method (e.g., binary tournament)
        if self.config.selection_method == "random":
            idx1 = ParentSelection.random_selection(self.population)
            idx2 = ParentSelection.random_selection(self.population)
            return self.population[idx1], self.population[idx2]

        if self.config.selection_method == "tournament":
            fitnesses = [ind.fitness for ind in self.population]
            idx1 = ParentSelection.tournament_selection(
                self.population,
                fitnesses,
                min(self.config.tournament_size, len(self.population)),
            )
            idx2 = ParentSelection.tournament_selection(
                self.population,
                fitnesses,
                min(self.config.tournament_size, len(self.population)),
            )
            return self.population[idx1], self.population[idx2]

        raise ValueError(f"Unsupported selection_method: {self.config.selection_method}")
    

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        # Perform crossover between two parents to produce an offspring solution (e.g., uniform crossover)
        if self.config.crossover_method == "uniform":
            offspring = Crossover.uniform_crossover(parent1, parent2)
            return offspring
        raise ValueError(f"Unsupported crossover_method: {self.config.crossover_method}")
    
    def mutate(self, individual: Individual) -> Individual:
        # Apply mutation to a solution with the configured mutation rate (e.g., mutate_solution function)
        if self.config.mutation_rate > 0:
            if random.random() < 0.5:
                mutated_individual = Mutation.mutate_component_from_pool(individual, self.component_pool, self.config.mutation_rate)
            else:
                mutated_individual = Mutation.mutate_component_LLM(individual, self.component_pool, self.config.mutation_rate)
            return mutated_individual
        return individual

    def environment_selection(self, current_population: List[Individual], new_population: List[Individual]) -> List[Individual]:
        # Select the next generation population from the current and new populations (e.g., elitism)
        if self.config.environment_selection_method == "elitism":
            selected_population = EnvironmentSelection.elitism_selection(current_population, new_population, self.config.population_size)
            return selected_population
        raise ValueError(
            f"Unsupported environment_selection_method: {self.config.environment_selection_method}"
        )
    
    def run(self):
        
        # log  logs/YYMMDD_HHMMSS/generation_X.txt
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{timestamp}"
        import os
        os.makedirs(log_dir, exist_ok=True)
        for individual in self.population:
            self.evaluate_fitness(individual)

        # # Test the evaluation of a random individual
        # test_individual = Individual()
        # test_individual.initialize_randomly(self.component_pool)
        # fitness = self.evaluate_fitness(test_individual)
        # print(f"Test Individual: {test_individual}")
        # print(f"Fitness: {fitness}")

        for generation in range(self.config.num_generations):
            new_population = []
            for _ in range(self.config.population_size):
                parent1, parent2 = self.select_parents()
                offspring = self.crossover(parent1, parent2)
                mutated_offspring = self.mutate(offspring)
                self.evaluate_fitness(mutated_offspring)
                new_population.append(mutated_offspring)
            self.population = self.environment_selection(self.population, new_population)

            # Save the best solution of the current generation            
            best_individual = max(self.population, key=lambda ind: ind.fitness)

            
            log_file = f"{log_dir}/generation_{generation+1}.txt"
            with open(log_file, "w") as f:
                f.write(f"Generation {generation+1}\n")
                f.write(f"Best Individual: {best_individual}\n")
                f.write(f"Prompt:\n{Evaluator(self.component_pool).construct_prompt(best_individual)}\n")
                f.write(f"Fitness: {best_individual.fitness}\n")
                f.write("\nPopulation:\n")
                for ind in self.population:
                    f.write(f"{ind} - Fitness: {ind.fitness}\n")

        # Store the components_pool in a file for later analysis
        import json
        components_file = f"{log_dir}/component_pool.json"
        with open(components_file, "w") as f:
            json.dump(self.component_pool.components, f, indent=4)

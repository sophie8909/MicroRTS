"""
NSGA-II implementation for multi-objective optimization of prompt components.
"""

from __future__ import annotations

import random
from typing import List

from .basic_ea import EA

from .evaluate import Evaluator
from .component_pool import ComponentPool
from .individual import Individual
from .config import EAConfig
from .parent_selection import ParentSelection
from .crossover import Crossover
from .mutation import Mutation
from .environment_selection import EnvironmentSelection

# inherit EA
class NSGA2(EA):
    def __init__(self, config: EAConfig, component_pool: ComponentPool, opponent_list: List[str]):
        super().__init__(config, component_pool, opponent_list)
    
    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
    

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        # Implement the fast non-dominated sorting algorithm to sort the population into Pareto fronts
        fronts = []
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]

        # Calculate domination relationships
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if population[i].dominates(population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Create first front with non-dominated individuals
        current_front = [i for i in range(len(population)) if domination_count[i] == 0]
        fronts.append([population[i] for i in current_front])

        # Generate remaining fronts
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append([population[i] for i in next_front])
            current_front = next_front

        return fronts


    def calculate_crowding_distance(self, front: List[Individual]) -> List[float]:
        # Calculate the crowding distance for individuals in a given Pareto front
        pass

    def select_next_generation(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        # Combine the current population and offspring, perform non-dominated sorting, and select the next generation based on Pareto fronts and crowding distance
        pass


    def run(self) -> list:
        # Main loop for evolving the population using NSGA-II
        
        log_dir = self.log_folder()

        for individual in self.population:
            self.real_evaluation(individual, random.choice(self.opponent_list))
            
        last_5_pareto_fronts = []
        
        for generation in range(self.config.generations):
        # 1. Generate offspring using selection, crossover, and mutation
            offspring = []
            for _ in range(self.config.population_size):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
        # 2. Combine current population and offspring
            combined_population = self.population + offspring
        # 3. Perform non-dominated sorting to identify Pareto fronts
            pareto_fronts = self.fast_non_dominated_sort(combined_population)
        # 4. Calculate crowding distance for individuals in each front
            for front in pareto_fronts:
                self.calculate_crowding_distance(front)
        # 5. Select the next generation based on Pareto fronts and crowding distance
            self.population = self.select_next_generation(combined_population, offspring)
        # 6. Log the Pareto fronts and their fitnesses for the current generation
            self.log_mo_generation(log_dir, generation, pareto_fronts)
        # 7. Check for convergence (e.g., if the best fitness has not improved for a certain number of generations)
            last_5_pareto_fronts.append(pareto_fronts[0]) 
            if len(last_5_pareto_fronts) > 5:
                last_5_pareto_fronts.pop(0)
            # last 5 generations have the same Pareto front, terminate the evolution
            if all(front == last_5_pareto_fronts[0] for front in last_5_pareto_fronts):   
                break

        self.save_components(log_dir)
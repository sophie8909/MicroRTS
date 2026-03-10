from __future__ import annotations

import random
from typing import List

from ea.base import EvolutionaryAlgorithm, Individual, NSGA2Config, Problem
from ea.operators import (
    binary_tournament_nsga2,
    compute_crowding_distance,
    non_dominated_sort,
)


class NSGA2(EvolutionaryAlgorithm):
    def __init__(self, problem: Problem, config: NSGA2Config):
        if not problem.is_multi_objective():
            raise ValueError("NSGA2 requires a multi-objective problem.")
        super().__init__(problem, config)
        self.config: NSGA2Config = config

    def assign_rank_and_crowding(self, population: List[Individual]) -> List[List[Individual]]:
        fronts = non_dominated_sort(population)
        for front in fronts:
            compute_crowding_distance(front)
        return fronts

    def make_offspring(self) -> List[Individual]:
        offspring: List[Individual] = []

        while len(offspring) < self.config.population_size:
            p1 = binary_tournament_nsga2(self.population)
            p2 = binary_tournament_nsga2(self.population)

            if random.random() < self.config.crossover_rate:
                c1, c2 = self.problem.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if random.random() < self.config.mutation_rate:
                c1 = self.problem.mutate(c1)
            if random.random() < self.config.mutation_rate:
                c2 = self.problem.mutate(c2)

            offspring.extend([c1, c2])

        return offspring[: self.config.population_size]

    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        self.evaluate_population(combined)
        fronts = self.assign_rank_and_crowding(combined)

        new_population: List[Individual] = []
        for front in fronts:
            if len(new_population) + len(front) <= self.config.population_size:
                new_population.extend(front)
            else:
                sorted_front = sorted(
                    front,
                    key=lambda ind: ind.crowding_distance,
                    reverse=True,
                )
                remaining = self.config.population_size - len(new_population)
                new_population.extend(sorted_front[:remaining])
                break

        return new_population

    def run(self) -> List[Individual]:
        self.initialize_population()
        self.evaluate_population(self.population)
        self.assign_rank_and_crowding(self.population)
        self.log_generation(0)

        for gen in range(1, self.config.generations + 1):
            offspring = self.make_offspring()
            combined = self.population + offspring
            self.population = self.environmental_selection(combined)
            self.assign_rank_and_crowding(self.population)
            self.log_generation(gen)

        fronts = self.assign_rank_and_crowding(self.population)
        return fronts[0]
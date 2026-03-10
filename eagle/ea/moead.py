from __future__ import annotations

import random
from typing import List

from ea.base import EvolutionaryAlgorithm, Individual, MOEADConfig, Problem
from ea.operators import (
    get_neighbors,
    tchebycheff_scalarization,
    uniform_weights_2d,
    weighted_sum_scalarization,
)


class MOEAD(EvolutionaryAlgorithm):
    """
    Simplified MOEA/D:
    - currently practical for 2-objective problems
    - one individual per weight vector
    - neighborhood-based mating and replacement
    """

    def __init__(self, problem: Problem, config: MOEADConfig):
        if not problem.is_multi_objective():
            raise ValueError("MOEAD requires a multi-objective problem.")
        if problem.num_objectives() != 2:
            raise ValueError("This simplified MOEA/D currently supports exactly 2 objectives.")
        super().__init__(problem, config)
        self.config: MOEADConfig = config

        self.weight_vectors = uniform_weights_2d(self.config.population_size)
        self.neighbors = get_neighbors(self.weight_vectors, self.config.n_neighbors)
        self.ideal_point: List[float] = [float("-inf")] * problem.num_objectives()

    def initialize_population(self) -> None:
        super().initialize_population()
        self.evaluate_population(self.population)
        self.update_ideal_point(self.population)

    def update_ideal_point(self, population: List[Individual]) -> None:
        for ind in population:
            for i, val in enumerate(ind.objectives):
                if val > self.ideal_point[i]:
                    self.ideal_point[i] = val

    def scalarize(self, objectives: List[float], weight: List[float]) -> float:
        if self.config.decomposition == "weighted_sum":
            return weighted_sum_scalarization(objectives, weight)
        return tchebycheff_scalarization(objectives, weight, self.ideal_point)

    def mating_selection(self, i: int) -> tuple[Individual, Individual]:
        nb = self.neighbors[i]
        p1_idx, p2_idx = random.sample(nb, 2)
        return self.population[p1_idx], self.population[p2_idx]

    def reproduce(self, p1: Individual, p2: Individual) -> Individual:
        if random.random() < self.config.crossover_rate:
            c1, _ = self.problem.crossover(p1, p2)
        else:
            c1 = p1.copy()

        if random.random() < self.config.mutation_rate:
            c1 = self.problem.mutate(c1)

        self.problem.evaluate(c1)
        c1.evaluated = True
        return c1

    def replace(self, i: int, child: Individual) -> None:
        target_indices = (
            range(self.config.population_size)
            if self.config.use_global_replacement
            else self.neighbors[i]
        )

        for j in target_indices:
            current = self.population[j]
            child_score = self.scalarize(child.objectives, self.weight_vectors[j])
            curr_score = self.scalarize(current.objectives, self.weight_vectors[j])

            if child_score > curr_score:
                self.population[j] = child.copy()

    def run(self) -> List[Individual]:
        self.initialize_population()
        self.log_generation(0)

        for gen in range(1, self.config.generations + 1):
            for i in range(self.config.population_size):
                p1, p2 = self.mating_selection(i)
                child = self.reproduce(p1, p2)
                self.update_ideal_point([child])
                self.replace(i, child)

            self.log_generation(gen)

        return self.population
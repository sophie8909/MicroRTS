from __future__ import annotations

import random
from typing import List
from .config import EAConfig
from .individual import Individual
from .operators import (
    neighborhood_indices,
    tchebycheff_scalarization,
    uniform_weights_2d,
    weighted_sum_scalarization,
)
from .selection import random_neighbor_indices


class MOEAD:
    """
    Simplified MOEA/D implementation.

    Important limitation:
    - This implementation supports exactly 2 objectives.
    - It is intended as a clean baseline, not a fully general MOEA/D framework.
    """

    def __init__(self, problem, cfg: EAConfig):
        self.problem = problem
        self.cfg = cfg
        self.population: List[Individual] = []
        self.history: List[dict] = []

        if self.problem.num_objectives() != 2:
            raise ValueError(
                "Simplified MOEA/D in this file supports exactly 2 objectives. "
                "Please set cfg.objective_names to length 2."
            )

        random.seed(self.cfg.seed)

        self.weight_vectors = uniform_weights_2d(self.cfg.population_size)
        self.neighbors = neighborhood_indices(
            self.weight_vectors,
            n_neighbors=min(self.cfg.moead_neighbors, self.cfg.population_size),
        )
        self.ideal_point = [float("-inf")] * 2

    def initialize(self) -> None:
        """
        Create and evaluate the initial population.
        """
        self.population = [
            self.problem.initialize_individual()
            for _ in range(self.cfg.population_size)
        ]
        self.evaluate_population(self.population)
        self.update_ideal_point(self.population)

    def evaluate_population(self, population: List[Individual]) -> None:
        """
        Evaluate all unevaluated individuals.
        """
        for ind in population:
            if not ind.evaluated:
                self.problem.evaluate(ind)

    def update_ideal_point(self, individuals: List[Individual]) -> None:
        """
        Update the ideal point using objective-wise maxima.
        """
        for ind in individuals:
            for i, val in enumerate(ind.objectives):
                if val > self.ideal_point[i]:
                    self.ideal_point[i] = val

    def scalarize(self, objectives: List[float], weights: List[float]) -> float:
        """
        Convert multi-objective values into a scalar value for one subproblem.
        """
        if self.cfg.moead_decomposition == "weighted_sum":
            return weighted_sum_scalarization(objectives, weights)

        return tchebycheff_scalarization(
            objectives=objectives,
            weights=weights,
            ideal_point=self.ideal_point,
        )

    def reproduce(self, index: int) -> Individual:
        """
        Reproduce one child for subproblem 'index' using neighborhood mating.
        """
        parent_indices = random_neighbor_indices(self.neighbors[index], n=2)
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]]

        # Apply crossover.
        if random.random() < self.cfg.crossover_rate:
            child, _ = self.problem.crossover(parent1, parent2)
        else:
            child = parent1.copy()

        # Apply mutation.
        if random.random() < self.cfg.mutation_rate:
            child = self.problem.mutate(child)

        self.problem.evaluate(child)
        self.update_ideal_point([child])
        return child

    def replace_neighbors(self, index: int, child: Individual) -> None:
        """
        Replace neighboring subproblems if the child improves their scalarized value.
        """
        if self.cfg.moead_global_replacement:
            target_indices = list(range(self.cfg.population_size))
        else:
            target_indices = self.neighbors[index]

        for j in target_indices:
            current = self.population[j]
            child_score = self.scalarize(child.objectives, self.weight_vectors[j])
            current_score = self.scalarize(current.objectives, self.weight_vectors[j])

            if child_score > current_score:
                self.population[j] = child.copy()

    def periodic_real_validation(self, generation: int) -> None:
        """
        Optionally run real game evaluation on top-k current candidates.

        We choose candidates by the sum of objectives as a simple proxy.
        """
        if not self.cfg.enable_real_eval:
            return

        if generation <= 0:
            return

        if generation % self.cfg.real_eval_every != 0:
            return

        sorted_population = sorted(
            self.population,
            key=lambda ind: sum(ind.objectives) if ind.objectives is not None else float("-inf"),
            reverse=True,
        )

        top_k = sorted_population[: self.cfg.real_eval_top_k]
        for ind in top_k:
            self.problem.validate_real(ind)

    def log_generation(self, generation: int) -> None:
        """
        Log simple population statistics for MOEA/D.
        """
        best = max(
            self.population,
            key=lambda ind: sum(ind.objectives) if ind.objectives is not None else float("-inf"),
        )

        record = {
            "generation": generation,
            "best_objectives": best.objectives,
        }
        self.history.append(record)

        if self.cfg.verbose:
            print(f"[MOEA/D] Gen {generation:03d} | best_objectives={best.objectives}")
            print(f"  prompt={best.metadata.get('prompt_text', '')[:200]}")

    def final_real_validation(self) -> List[Individual]:
        """
        Run real validation on the final top-k MOEA/D solutions.
        """
        if not self.cfg.enable_real_eval:
            return self.population

        sorted_population = sorted(
            self.population,
            key=lambda ind: sum(ind.objectives) if ind.objectives is not None else float("-inf"),
            reverse=True,
        )
        top_k = sorted_population[: self.cfg.real_eval_top_k]

        for ind in top_k:
            self.problem.validate_real(ind)

        return top_k

    def run(self) -> List[Individual]:
        """
        Run MOEA/D and return the final validated top-k solutions.
        """
        self.initialize()
        self.log_generation(0)

        for generation in range(1, self.cfg.generations + 1):
            for i in range(self.cfg.population_size):
                child = self.reproduce(i)
                self.replace_neighbors(i, child)

            self.periodic_real_validation(generation)
            self.log_generation(generation)

        return self.final_real_validation()
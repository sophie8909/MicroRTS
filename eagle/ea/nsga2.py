from __future__ import annotations

import random
from typing import List
from .config import EAConfig
from .individual import Individual
from .operators import compute_crowding_distance, non_dominated_sort
from .selection import binary_tournament_nsga2


class NSGA2:
    """
    Standard NSGA-II implementation for static prompt evolution.
    """

    def __init__(self, problem, cfg: EAConfig):
        self.problem = problem
        self.cfg = cfg
        self.population: List[Individual] = []
        self.history: List[dict] = []

        random.seed(self.cfg.seed)

    def initialize(self) -> None:
        """
        Create and evaluate the initial population.
        """
        self.population = [
            self.problem.initialize_individual()
            for _ in range(self.cfg.population_size)
        ]
        self.evaluate_population(self.population)
        self.assign_rank_and_crowding(self.population)

    def evaluate_population(self, population: List[Individual]) -> None:
        """
        Evaluate all unevaluated individuals.
        """
        for ind in population:
            if not ind.evaluated:
                self.problem.evaluate(ind)

    def assign_rank_and_crowding(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Assign non-domination rank and crowding distance to a population.
        """
        fronts = non_dominated_sort(population)
        for front in fronts:
            compute_crowding_distance(front)
        return fronts

    def make_offspring(self) -> List[Individual]:
        """
        Generate offspring using binary tournament selection, crossover, and mutation.
        """
        offspring: List[Individual] = []

        while len(offspring) < self.cfg.population_size:
            parent1 = binary_tournament_nsga2(self.population)
            parent2 = binary_tournament_nsga2(self.population)

            # Apply crossover probabilistically.
            if random.random() < self.cfg.crossover_rate:
                child1, child2 = self.problem.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation probabilistically.
            if random.random() < self.cfg.mutation_rate:
                child1 = self.problem.mutate(child1)
            if random.random() < self.cfg.mutation_rate:
                child2 = self.problem.mutate(child2)

            offspring.extend([child1, child2])

        return offspring[: self.cfg.population_size]

    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """
        Select the next population from parents + offspring using NSGA-II rules.
        """
        self.evaluate_population(combined)
        fronts = self.assign_rank_and_crowding(combined)

        next_population: List[Individual] = []
        for front in fronts:
            if len(next_population) + len(front) <= self.cfg.population_size:
                next_population.extend(front)
            else:
                # Sort by crowding distance descending and fill the remaining slots.
                remaining = self.cfg.population_size - len(next_population)
                sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
                next_population.extend(sorted_front[:remaining])
                break

        return next_population

    def periodic_real_validation(self, generation: int) -> None:
        """
        Optionally run real game evaluation on top-k current candidates.
        """
        if not self.cfg.enable_real_eval:
            return

        if generation <= 0:
            return

        if generation % self.cfg.real_eval_every != 0:
            return

        # Prefer low-rank and diverse solutions.
        sorted_population = sorted(
            self.population,
            key=lambda ind: (
                ind.rank if ind.rank is not None else 10**9,
                -ind.crowding_distance,
            ),
        )

        top_k = sorted_population[: self.cfg.real_eval_top_k]
        for ind in top_k:
            self.problem.validate_real(ind)

    def log_generation(self, generation: int) -> None:
        """
        Save and print generation-level logs.
        """
        fronts = self.assign_rank_and_crowding(self.population)
        pareto_front = fronts[0] if fronts else []

        record = {
            "generation": generation,
            "pareto_size": len(pareto_front),
            "front0_objectives": [ind.objectives for ind in pareto_front],
        }
        self.history.append(record)

        if self.cfg.verbose:
            print(f"[NSGA-II] Gen {generation:03d} | pareto_size={len(pareto_front)}")

            # Print one representative solution for debugging.
            if pareto_front:
                ref = pareto_front[0]
                print(f"  objectives={ref.objectives}")
                print(f"  prompt={ref.metadata.get('prompt_text', '')[:200]}")

    def final_real_validation(self) -> List[Individual]:
        """
        Run real game evaluation on final Pareto-front solutions.
        """
        if not self.cfg.enable_real_eval:
            return self.population

        fronts = self.assign_rank_and_crowding(self.population)
        pareto_front = fronts[0] if fronts else []

        for ind in pareto_front:
            self.problem.validate_real(ind)

        return pareto_front

    def run(self) -> List[Individual]:
        """
        Run NSGA-II and return the final Pareto front.
        """
        self.initialize()
        self.log_generation(0)

        for generation in range(1, self.cfg.generations + 1):
            offspring = self.make_offspring()
            combined = self.population + offspring
            self.population = self.environmental_selection(combined)
            self.assign_rank_and_crowding(self.population)

            self.periodic_real_validation(generation)
            self.log_generation(generation)

        return self.final_real_validation()
from __future__ import annotations

import random
from typing import List
from .config import EAConfig
from .individual import Individual


class GA:
    """
    Simple single-objective Genetic Algorithm.

    Design goals:
    - Keep the interface similar to NSGA-II / MOEA/D
    - Use one scalar fitness only
    - Make the full search pipeline easy to understand
    """

    def __init__(self, problem, cfg: EAConfig):
        self.problem = problem
        self.cfg = cfg
        self.population: List[Individual] = []
        self.history: List[dict] = []

        random.seed(self.cfg.seed)

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def initialize(self) -> None:
        """
        Create the initial population and evaluate all individuals.
        """
        self.population = [
            self.problem.initialize_individual()
            for _ in range(self.cfg.population_size)
        ]
        self.evaluate_population(self.population)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def evaluate_population(self, population: List[Individual]) -> None:
        """
        Evaluate all unevaluated individuals.
        """
        for ind in population:
            if not ind.evaluated:
                self.problem.evaluate(ind)

            if ind.fitness is None:
                raise ValueError(
                    "GA requires each individual to have a scalar fitness value. "
                    "Make sure problem.evaluate() sets individual.fitness."
                )

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------
    def tournament_selection(self) -> Individual:
        """
        Standard tournament selection.

        A few individuals are sampled randomly, and the one with the
        highest fitness wins.
        """
        competitors = random.sample(self.population, self.cfg.tournament_size)
        return max(competitors, key=lambda ind: ind.fitness)

    # --------------------------------------------------
    # Variation
    # --------------------------------------------------
    def make_offspring(self) -> List[Individual]:
        """
        Generate offspring using:
        - tournament selection
        - crossover
        - mutation
        """
        offspring: List[Individual] = []

        while len(offspring) < self.cfg.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Apply crossover with probability cfg.crossover_rate
            if random.random() < self.cfg.crossover_rate:
                child1, child2 = self.problem.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation with probability cfg.mutation_rate
            if random.random() < self.cfg.mutation_rate:
                child1 = self.problem.mutate(child1)

            if random.random() < self.cfg.mutation_rate:
                child2 = self.problem.mutate(child2)

            offspring.extend([child1, child2])

        return offspring[: self.cfg.population_size]

    # --------------------------------------------------
    # Survivor selection
    # --------------------------------------------------
    def environmental_selection(self, offspring: List[Individual]) -> None:
        """
        Select the next generation.

        Strategy:
        - Keep top elitism_size individuals from the current population
        - Fill the rest with the best individuals from offspring
        """
        self.evaluate_population(offspring)

        elites = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=True,
        )[: self.cfg.elitism_size]

        remaining_slots = self.cfg.population_size - len(elites)

        best_offspring = sorted(
            offspring,
            key=lambda ind: ind.fitness,
            reverse=True,
        )[:remaining_slots]

        self.population = elites + best_offspring

    # --------------------------------------------------
    # Optional real validation
    # --------------------------------------------------
    def periodic_real_validation(self, generation: int) -> None:
        """
        Optionally run real game evaluation on top-k individuals.

        This does not change fitness by default.
        It only adds extra metadata for analysis.
        """
        if not getattr(self.cfg, "enable_real_eval", False):
            return

        if generation <= 0:
            return

        if generation % self.cfg.real_eval_every != 0:
            return

        sorted_population = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=True,
        )

        top_k = sorted_population[: self.cfg.real_eval_top_k]

        for ind in top_k:
            self.problem.validate_real(ind)

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    def log_generation(self, generation: int) -> None:
        """
        Save and print generation statistics.
        """
        best = max(self.population, key=lambda ind: ind.fitness)
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)

        record = {
            "generation": generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg_fitness,
            "best_prompt": best.metadata.get("prompt_text", ""),
        }
        self.history.append(record)

        if self.cfg.verbose:
            print(
                f"[GA] Gen {generation:03d} | "
                f"best_fitness={best.fitness:.6f} | "
                f"avg_fitness={avg_fitness:.6f}"
            )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------
    def final_real_validation(self) -> List[Individual]:
        """
        Optionally validate final top-k individuals with real games.
        """
        if not getattr(self.cfg, "enable_real_eval", False):
            return sorted(self.population, key=lambda ind: ind.fitness, reverse=True)

        sorted_population = sorted(
            self.population,
            key=lambda ind: ind.fitness,
            reverse=True,
        )

        top_k = sorted_population[: self.cfg.real_eval_top_k]

        for ind in top_k:
            self.problem.validate_real(ind)

        return top_k

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    def run(self) -> List[Individual]:
        """
        Run the full GA loop and return the final best solutions.
        """
        self.initialize()
        self.log_generation(0)

        for generation in range(1, self.cfg.generations + 1):
            offspring = self.make_offspring()
            self.environmental_selection(offspring)
            self.periodic_real_validation(generation)
            self.log_generation(generation)

        return self.final_real_validation()
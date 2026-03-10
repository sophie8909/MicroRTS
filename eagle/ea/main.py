from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Individual:
    genome: Any
    fitness: Optional[float] = None
    objectives: Optional[List[float]] = None
    evaluated: bool = False

    # For multi-objective methods
    rank: Optional[int] = None
    crowding_distance: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Individual":
        return Individual(
            genome=copy.deepcopy(self.genome),
            fitness=self.fitness,
            objectives=copy.deepcopy(self.objectives),
            evaluated=self.evaluated,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass
class EAConfig:
    population_size: int = 20
    generations: int = 30
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 2
    elitism_size: int = 0
    seed: int = 42
    verbose: bool = True


@dataclass
class NSGA2Config(EAConfig):
    pass


@dataclass
class MOEADConfig(EAConfig):
    n_neighbors: int = 5
    decomposition: str = "tchebycheff"  # or "weighted_sum"
    use_global_replacement: bool = False


class Problem:
    """
    Base problem interface.
    Assume maximization by default.
    For multi-objective problems:
        objectives[i] larger is better
    """

    def initialize_individual(self) -> Individual:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> Individual:
        """
        Must set at least one of:
        - individual.fitness   (single-objective)
        - individual.objectives (multi-objective)
        """
        raise NotImplementedError

    def mutate(self, individual: Individual) -> Individual:
        raise NotImplementedError

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        raise NotImplementedError

    def is_multi_objective(self) -> bool:
        return False

    def num_objectives(self) -> int:
        return 1


class EvolutionaryAlgorithm:
    def __init__(self, problem: Problem, config: EAConfig):
        self.problem = problem
        self.config = config
        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
        random.seed(self.config.seed)

    def initialize_population(self) -> None:
        self.population = [
            self.problem.initialize_individual()
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(self, population: Sequence[Individual]) -> None:
        for ind in population:
            if not ind.evaluated:
                self.problem.evaluate(ind)
                ind.evaluated = True

    def log_generation(self, generation: int) -> None:
        record: Dict[str, Any] = {"generation": generation}

        if self.problem.is_multi_objective():
            fronts = [ind for ind in self.population if ind.rank == 0]
            record["pareto_size"] = len(fronts)
            record["front0_objectives"] = [ind.objectives for ind in fronts]

            if self.config.verbose:
                print(f"[Gen {generation:03d}] pareto_size={len(fronts)}")
        else:
            best = max(self.population, key=lambda x: x.fitness)
            avg = sum(ind.fitness for ind in self.population if ind.fitness is not None) / len(self.population)
            record["best_fitness"] = best.fitness
            record["best_genome"] = copy.deepcopy(best.genome)
            record["avg_fitness"] = avg

            if self.config.verbose:
                print(
                    f"[Gen {generation:03d}] "
                    f"best={best.fitness:.6f}, avg={avg:.6f}"
                )

        self.history.append(record)

    def run(self):
        raise NotImplementedError
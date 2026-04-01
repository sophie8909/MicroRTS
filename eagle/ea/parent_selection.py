"""Parent selection methods for the genetic algorithm."""

from __future__ import annotations

from .fitness_utils import fitness_key

class ParentSelection:
    @staticmethod
    def tournament_selection(population: list, fitnesses: list, tournament_size: int) -> int:
        import random
        tournament_indices = random.sample(range(len(population)), tournament_size)
        return max(tournament_indices, key=lambda idx: fitness_key(fitnesses[idx]))
    
    @staticmethod
    def random_selection(population: list) -> int:
        import random
        return random.randint(0, len(population) - 1)

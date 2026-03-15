"""Parent selection methods for the genetic algorithm."""

from __future__ import annotations

class ParentSelection:
    @staticmethod
    def tournament_selection(population: list, fitnesses: list, tournament_size: int) -> int:
        import random
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        best_index_in_tournament = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return best_index_in_tournament
    
    @staticmethod
    def random_selection(population: list) -> int:
        import random
        return random.randint(0, len(population) - 1)
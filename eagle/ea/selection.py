from __future__ import annotations

import random
from typing import List, Sequence
from .individual import Individual


_WORST_RANK = 10**9



def _effective_rank(individual: Individual) -> int:
    """
    Return the rank used for NSGA-II comparison.

    Individuals without an assigned rank are treated as worst rank.
    """
    return individual.rank if individual.rank is not None else _WORST_RANK


def binary_tournament_nsga2(population: List[Individual]) -> Individual:
    """
    Binary tournament selection for NSGA-II.

    Priority:
    1. Lower rank is better.
    2. Higher crowding distance is better.
    3. Random tie-break.
    """
    if len(population) < 2:
        raise ValueError("binary_tournament_nsga2 requires at least 2 individuals.")

    a, b = random.sample(population, 2)

    rank_a = _effective_rank(a)
    rank_b = _effective_rank(b)

    if rank_a < rank_b:
        return a
    if rank_b < rank_a:
        return b

    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b

    return random.choice((a, b))


def random_neighbor_indices(neighbor_list: Sequence[int], n: int = 2) -> List[int]:
    """
    Sample indices from a neighborhood list.

    This helper is used by MOEA/D for neighborhood mating.
    """
    if n <= 0:
        return []

    if not neighbor_list:
        raise ValueError("random_neighbor_indices requires a non-empty neighbor list.")

    neighbors = list(neighbor_list)
    if len(neighbors) < n:
        return random.choices(neighbors, k=n)
    return random.sample(neighbors, n)
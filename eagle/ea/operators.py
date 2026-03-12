from __future__ import annotations

import math
from typing import List, Sequence
from .individual import Individual


def dominates(a: Individual, b: Individual) -> bool:
    """
    Return True when individual a dominates individual b.

    Assumption:
    - Multi-objective maximization
    - Larger objective values are better

    Definition:
    - a is no worse than b in all objectives
    - a is strictly better than b in at least one objective
    """
    if a.objectives is None or b.objectives is None:
        raise ValueError("Both individuals must have objectives before dominance comparison.")

    no_worse = all(x >= y for x, y in zip(a.objectives, b.objectives))
    strictly_better = any(x > y for x, y in zip(a.objectives, b.objectives))
    return no_worse and strictly_better


def non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Perform standard non-dominated sorting.

    Returns:
    - A list of fronts
    - Front 0 is the Pareto front
    """
    if not population:
        return []

    n = len(population)
    dominates_list: List[List[int]] = [[] for _ in range(n)]
    dominated_count: List[int] = [0 for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if dominates(population[i], population[j]):
                dominates_list[i].append(j)
            elif dominates(population[j], population[i]):
                dominated_count[i] += 1

        if dominated_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front: List[int] = []

        for i in fronts[current_front]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)

        current_front += 1

    return [[population[i] for i in front] for front in fronts if front]


def compute_crowding_distance(front: List[Individual]) -> None:
    """
    Compute crowding distance for one front.

    Boundary points receive +inf to preserve extreme solutions.
    """
    if not front:
        return

    num_individuals = len(front)
    num_objectives = len(front[0].objectives or [])

    for ind in front:
        ind.crowding_distance = 0.0

    if num_individuals <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    for obj_idx in range(num_objectives):
        sorted_front = sorted(front, key=lambda ind: ind.objectives[obj_idx])

        sorted_front[0].crowding_distance = float("inf")
        sorted_front[-1].crowding_distance = float("inf")

        min_val = sorted_front[0].objectives[obj_idx]
        max_val = sorted_front[-1].objectives[obj_idx]

        if math.isclose(max_val, min_val):
            # When all values are identical, this objective contributes nothing.
            continue

        for i in range(1, num_individuals - 1):
            prev_val = sorted_front[i - 1].objectives[obj_idx]
            next_val = sorted_front[i + 1].objectives[obj_idx]
            distance = (next_val - prev_val) / (max_val - min_val)
            sorted_front[i].crowding_distance += distance


def uniform_weights_2d(n: int) -> List[List[float]]:
    """
    Generate evenly distributed 2D weight vectors for simplified MOEA/D.

    Example for n=5:
    [
        [0.0, 1.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.75, 0.25],
        [1.0, 0.0]
    ]
    """
    if n <= 1:
        return [[0.5, 0.5]]

    return [[i / (n - 1), 1.0 - (i / (n - 1))] for i in range(n)]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute Euclidean distance between two vectors.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def neighborhood_indices(weight_vectors: List[List[float]], n_neighbors: int) -> List[List[int]]:
    """
    Build neighborhood lists for MOEA/D based on weight vector distances.
    """
    all_neighbors: List[List[int]] = []

    for i, wi in enumerate(weight_vectors):
        distances = []
        for j, wj in enumerate(weight_vectors):
            distances.append((euclidean_distance(wi, wj), j))

        distances.sort(key=lambda x: x[0])
        all_neighbors.append([idx for _, idx in distances[:n_neighbors]])

    return all_neighbors


def weighted_sum_scalarization(objectives: Sequence[float], weights: Sequence[float]) -> float:
    """
    Weighted-sum scalarization for multi-objective maximization.
    """
    return sum(w * obj for w, obj in zip(weights, objectives))


def tchebycheff_scalarization(
    objectives: Sequence[float],
    weights: Sequence[float],
    ideal_point: Sequence[float],
) -> float:
    """
    Tchebycheff scalarization for multi-objective maximization.

    We return the negative distance to the ideal point so that:
    - larger returned values are better
    """
    values = []
    for obj, weight, ideal in zip(objectives, weights, ideal_point):
        safe_weight = max(weight, 1e-8)
        values.append(safe_weight * abs(ideal - obj))


    return -max(values)
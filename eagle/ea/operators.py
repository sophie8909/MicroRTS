from __future__ import annotations

import math
import random
from typing import List, Sequence

from ea.base import Individual


def tournament_selection(population: Sequence[Individual], tournament_size: int = 2) -> Individual:
    competitors = random.sample(list(population), tournament_size)
    return max(competitors, key=lambda ind: ind.fitness)


def binary_tournament_nsga2(population: Sequence[Individual]) -> Individual:
    a, b = random.sample(list(population), 2)

    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b

    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b

    return random.choice([a, b])


def dominates(a: Individual, b: Individual) -> bool:
    """
    Maximization version:
    a dominates b if:
    - a is no worse in all objectives
    - a is strictly better in at least one
    """
    assert a.objectives is not None and b.objectives is not None

    no_worse_all = all(x >= y for x, y in zip(a.objectives, b.objectives))
    better_any = any(x > y for x, y in zip(a.objectives, b.objectives))
    return no_worse_all and better_any


def non_dominated_sort(population: Sequence[Individual]) -> List[List[Individual]]:
    pop = list(population)
    domination_count = {}
    dominates_set = {}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        domination_count[prefix_id(p)] = 0
        dominates_set[prefix_id(p)] = []

        for q in pop:
            if p is q:
                continue

            if dominates(p, q):
                dominates_set[prefix_id(p)].append(q)
            elif dominates(q, p):
                domination_count[prefix_id(p)] += 1

        if domination_count[prefix_id(p)] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[Individual] = []

        for p in fronts[i]:
            for q in dominates_set[prefix_id(p)]:
                domination_count[prefix_id(q)] -= 1
                if domination_count[prefix_id(q)] == 0:
                    q.rank = i + 1
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def compute_crowding_distance(front: Sequence[Individual]) -> None:
    if not front:
        return

    n = len(front)
    for ind in front:
        ind.crowding_distance = 0.0

    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    m = len(front[0].objectives)

    for obj_idx in range(m):
        sorted_front = sorted(front, key=lambda ind: ind.objectives[obj_idx])
        sorted_front[0].crowding_distance = float("inf")
        sorted_front[-1].crowding_distance = float("inf")

        min_val = sorted_front[0].objectives[obj_idx]
        max_val = sorted_front[-1].objectives[obj_idx]

        if math.isclose(max_val, min_val):
            continue

        for i in range(1, n - 1):
            prev_val = sorted_front[i - 1].objectives[obj_idx]
            next_val = sorted_front[i + 1].objectives[obj_idx]
            distance = (next_val - prev_val) / (max_val - min_val)
            sorted_front[i].crowding_distance += distance


def prefix_id(ind: Individual) -> int:
    return id(ind)


def uniform_weights_2d(n: int) -> List[List[float]]:
    if n == 1:
        return [[0.5, 0.5]]
    return [[i / (n - 1), 1.0 - i / (n - 1)] for i in range(n)]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def get_neighbors(weight_vectors: List[List[float]], k: int) -> List[List[int]]:
    neighbors = []
    for i, w in enumerate(weight_vectors):
        dists = []
        for j, w2 in enumerate(weight_vectors):
            dist = euclidean_distance(w, w2)
            dists.append((dist, j))
        dists.sort(key=lambda x: x[0])
        neighbors.append([idx for _, idx in dists[:k]])
    return neighbors


def tchebycheff_scalarization(
    objectives: Sequence[float],
    weight: Sequence[float],
    ideal_point: Sequence[float],
) -> float:
    """
    Maximization version:
    convert to minimization-like scalar by negating relative gap to ideal
    larger is better, so closer to ideal is better.
    we return negative max gap so higher is better.
    """
    vals = []
    for obj, w, z in zip(objectives, weight, ideal_point):
        w = max(w, 1e-8)
        vals.append(w * abs(z - obj))
    return -max(vals)


def weighted_sum_scalarization(
    objectives: Sequence[float],
    weight: Sequence[float],
) -> float:
    return sum(w * obj for w, obj in zip(weight, objectives))
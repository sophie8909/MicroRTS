from __future__ import annotations

import random
from typing import Callable, List
from .component_pool import ComponentPool
from .individual import Individual


def _mutable_component_keys(
    individual: Individual,
    fixed_keys: List[str] | None = None,
    allow_strategy_mutation: bool = True,
) -> List[str]:
    """
    Return component keys that are allowed to be modified.
    """
    fixed_keys = fixed_keys or []
    keys = [k for k in individual.component_order if k not in fixed_keys]

    if not allow_strategy_mutation:
        keys = [k for k in keys if k != "strategy"]

    return keys


def component_mutation(
    individual: Individual,
    pool: ComponentPool,
    fixed_keys: List[str] | None = None,
    allow_strategy_mutation: bool = True,
) -> Individual:
    """
    Apply one component-level mutation.

    Supported actions:
    - replace: replace one component with another sample from the same pool
    - insert:  add/overwrite one component from the pool
    - delete:  remove one component from the prompt
    """
    child = individual.copy()
    mutable_keys = _mutable_component_keys(
        child,
        fixed_keys=fixed_keys,
        allow_strategy_mutation=allow_strategy_mutation,
    )

    if not mutable_keys:
        return child

    op = random.choice(["replace", "insert", "delete"])

    if op in ("replace", "insert"):
        key = random.choice(mutable_keys)
        if pool.has_type(key):
            child.components[key] = pool.sample(key)

    elif op == "delete":
        existing_keys = [k for k in mutable_keys if k in child.components]
        if existing_keys:
            key = random.choice(existing_keys)
            del child.components[key]

    # Invalidate cached evaluation.
    child.prompt_text = None
    child.objectives = None
    child.evaluated = False
    child.rank = None
    child.crowding_distance = 0.0
    return child


def token_mutation(
    individual: Individual,
    fill_mask_fn: Callable[[str], str],
    fixed_keys: List[str] | None = None,
    allow_strategy_mutation: bool = True,
    n_masks_min: int = 1,
    n_masks_max: int = 4,
) -> Individual:
    """
    Apply token-level mutation by masking and then refilling text.

    The mutation does not require real LLM support at the interface level.
    It only requires a function:
        fill_mask_fn(masked_text) -> new_text

    Example actions:
    - replace existing tokens with [MASK]
    - insert [MASK] before some positions
    """
    child = individual.copy()

    mutable_keys = [
        k for k in _mutable_component_keys(
            child,
            fixed_keys=fixed_keys,
            allow_strategy_mutation=allow_strategy_mutation,
        )
        if k in child.components
    ]

    if not mutable_keys:
        return child

    target_key = random.choice(mutable_keys)
    words = child.components[target_key].split()

    if not words:
        return child

    n_masks = random.randint(n_masks_min, n_masks_max)
    n_masks = min(n_masks, max(1, len(words)))

    # Use reverse-sorted positions so list insertion does not shift future indices.
    positions = sorted(random.sample(range(len(words)), k=n_masks), reverse=True)

    for pos in positions:
        if random.random() < 0.5:
            # Replace an existing token.
            words[pos] = "[MASK]"
        else:
            # Insert a new mask before the token.
            words.insert(pos, "[MASK]")

    masked_text = " ".join(words)
    refilled_text = fill_mask_fn(masked_text)
    child.components[target_key] = refilled_text

    # Invalidate cached evaluation.
    child.prompt_text = None
    child.objectives = None
    child.evaluated = False
    child.rank = None
    child.crowding_distance = 0.0
    return child


def component_crossover(
    parent1: Individual,
    parent2: Individual,
    fixed_keys: List[str] | None = None,
    allow_strategy_mutation: bool = True,
) -> tuple[Individual, Individual]:
    """
    Perform component-wise crossover between two parents.

    Shared mutable component keys are swapped between the two children.
    """
    fixed_keys = fixed_keys or []

    child1 = parent1.copy()
    child2 = parent2.copy()

    common_keys = set(child1.components.keys()) & set(child2.components.keys())
    common_keys = [k for k in common_keys if k not in fixed_keys]

    if not allow_strategy_mutation:
        common_keys = [k for k in common_keys if k != "strategy"]

    if not common_keys:
        return child1, child2

    # Swap roughly half of the available keys.
    n_swap = max(1, len(common_keys) // 2)
    swap_keys = random.sample(common_keys, k=n_swap)

    for key in swap_keys:
        child1.components[key], child2.components[key] = (
            child2.components[key],
            child1.components[key],
        )

    for child in (child1, child2):
        child.prompt_text = None
        child.objectives = None
        child.evaluated = False
        child.rank = None
        child.crowding_distance = 0.0

    return child1, child2
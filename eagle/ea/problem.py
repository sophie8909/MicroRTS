from __future__ import annotations

import random
from typing import Dict, List
from .component_pool import ComponentPool
from .config import EAConfig
from .evaluator import PromptEvaluator
from .individual import Individual
from .variation import component_crossover, component_mutation, token_mutation


class MicroRTSPromptProblem:
    """
    Problem definition for static prompt evolution in MicroRTS.

    Core idea:
    - Search target: static prompt only
    - Dynamic state: injected only during evaluation
    - Evolution objectives: selected from surrogate metrics and prompt fluency
    - Real game evaluation: optional validation, not required every time
    """

    def __init__(
        self,
        cfg: EAConfig,
        component_pool: ComponentPool,
        evaluator: PromptEvaluator,
        fill_mask_fn,
    ):
        self.cfg = cfg
        self.component_pool = component_pool
        self.evaluator = evaluator
        self.fill_mask_fn = fill_mask_fn

    def num_objectives(self) -> int:
        """
        Return the number of active evolutionary objectives.
        """
        return len(self.cfg.objective_names)

    def initialize_individual(self) -> Individual:
        """
        Build one individual by sampling one candidate from each component type.
        """
        components: Dict[str, str] = {}

        for key in self.cfg.component_order:
            if self.component_pool.has_type(key):
                components[key] = self.component_pool.sample(key)

        individual = Individual(
            components=components,
            component_order=self.cfg.component_order[:],
        )
        individual.render_prompt()
        return individual

    def evaluate(self, individual: Individual) -> Individual:
        """
        Evaluate one individual using the surrogate pipeline.

        The actual objective vector is built from cfg.objective_names.
        """
        prompt_text = individual.render_prompt()
        metrics = self.evaluator.evaluate_surrogate(
            prompt_text=prompt_text,
            n_states=self.cfg.surrogate_states,
        )

        objectives: List[float] = []
        for name in self.cfg.objective_names:
            if name not in metrics:
                raise KeyError(
                    f"Objective '{name}' not found in surrogate metrics. "
                    f"Available keys: {list(metrics.keys())}"
                )
            objectives.append(float(metrics[name]))

        individual.objectives = objectives
        individual.evaluated = True
        individual.metadata.update(metrics)
        individual.metadata["prompt_text"] = prompt_text
        return individual

    def validate_real(self, individual: Individual) -> Individual:
        """
        Run real Java evaluation for an already-rendered candidate prompt.

        This method does not alter the evolutionary objectives by default.
        It only appends metadata for analysis/final selection.
        """
        prompt_text = individual.render_prompt()
        real_metrics = self.evaluator.evaluate_real(
            prompt_text=prompt_text,
            n_matches=self.cfg.real_matches,
        )
        individual.metadata.update(real_metrics)
        return individual

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply mutation to one individual.

        Both component-level and token-level mutations may happen.
        """
        child = individual.copy()

        # Component-level mutation.
        if random.random() < self.cfg.component_mutation_prob:
            child = component_mutation(
                child,
                pool=self.component_pool,
                fixed_keys=self.cfg.fixed_components,
                allow_strategy_mutation=self.cfg.allow_strategy_mutation,
            )

        # Token-level mutation.
        if random.random() < self.cfg.token_mutation_prob:
            child = token_mutation(
                child,
                fill_mask_fn=self.fill_mask_fn,
                fixed_keys=self.cfg.fixed_components,
                allow_strategy_mutation=self.cfg.allow_strategy_mutation,
                n_masks_min=self.cfg.n_masks_min,
                n_masks_max=self.cfg.n_masks_max,
            )

        return child

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Perform component-wise crossover between two parents.
        """
        return component_crossover(
            parent1,
            parent2,
            fixed_keys=self.cfg.fixed_components,
            allow_strategy_mutation=self.cfg.allow_strategy_mutation,
        )
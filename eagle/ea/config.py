from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EAConfig:
    """
    Central configuration for prompt evolution.

    Notes:
    - The search target is always the static prompt.
    - Real game evaluation is optional and is used for validation/final ranking.
    - Surrogate evaluation is the default cheap evaluator during evolution.
    """

    # -------------------------
    # General experiment setup
    # -------------------------
    seed: int = 42
    algorithm: str = "ga"  # "nsga2" or "moead"
    generations: int = 5
    population_size: int = 3
    verbose: bool = True

    # -------------------------
    # Evolutionary parameters
    # -------------------------
    crossover_rate: float = 0.9
    mutation_rate: float = 0.9

    # Component-level and token-level mutation probabilities.
    # Both can happen in the same mutation step.
    component_mutation_prob: float = 0.7
    token_mutation_prob: float = 0.7

    # Token masking range for token-level mutation.
    n_masks_min: int = 1
    n_masks_max: int = 4

    # -------------------------
    # Prompt structure
    # -------------------------
    component_order: List[str] = field(default_factory=lambda: [
        "role",
        "critical_rules",
        "actions",
        "json_schema",
        "field_requirements",
        "examples",
        "strategy",
    ])

    # Components listed here will not be changed by mutation/crossover.
    fixed_components: List[str] = field(default_factory=list)

    # When False, the "strategy" component is frozen.
    allow_strategy_mutation: bool = True

    # -------------------------
    # Objectives used by evolution
    # -------------------------
    # These are the metrics that become the evolutionary objectives.
    # They are all assumed to be "larger is better".
    objective_names: List[str] = field(default_factory=lambda: [
        "fluency",
        "format_score",
        "strategy_score",
        "turn_score",
    ])

    # -------------------------
    # Surrogate evaluation
    # -------------------------
    surrogate_states: int = 8

    # -------------------------
    # Real game evaluation
    # -------------------------
    enable_real_eval: bool = True
    real_matches: int = 3
    real_eval_every: int = 5
    real_eval_top_k: int = 2

    # -------------------------
    # MOEA/D specific settings
    # -------------------------
    moead_neighbors: int = 5
    moead_decomposition: str = "tchebycheff"  # "tchebycheff" or "weighted_sum"
    moead_global_replacement: bool = False

    # -------------------------
    # Paths
    # -------------------------
    components_json_path: str = "eagle/prompts/components.json"

    # Optional surrogate state dataset path.
    # Supported formats in main.py:
    # - .json  -> list[dict]
    # - .jsonl -> one dict per line
    surrogate_states_path: str = ""

    # -------------------------
    # Java runner settings
    # -------------------------
    java_cmd: str = "java"
    java_classpath: str = "lib/*:bin"
    java_main_class: str = "rts.MicroRTS"
    java_timeout_sec: int = 180
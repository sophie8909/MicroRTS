from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import random


@dataclass
class SurrogateConfig:
    """
    Configuration for surrogate evaluation.
    """
    n_states: int = 8


class SurrogateEvaluator:
    """
    Cheap evaluator for prompt search.

    Expected dependencies:
    - executor: generates one-turn outputs from static prompt + state
    - judge: scores those outputs

    Required executor interface:
        generate_turn(static_prompt: str, state: dict) -> Any

    Required judge interface:
        evaluate_turn(static_prompt: str, state: dict, llm_output: Any) -> dict

    Expected judge return keys:
        {
            "format_score": float,
            "rule_score": float,
            "strategy_score": float,
            "turn_score": float
        }
    """

    def __init__(self, state_dataset: List[Dict[str, Any]], executor, judge):
        self.state_dataset = state_dataset
        self.executor = executor
        self.judge = judge

    def sample_states(self, n_states: int) -> List[Dict[str, Any]]:
        """
        Randomly sample game states for surrogate evaluation.
        """
        if not self.state_dataset:
            raise ValueError("state_dataset is empty. Surrogate evaluation cannot proceed.")

        if n_states >= len(self.state_dataset):
            return self.state_dataset[:]

        return random.sample(self.state_dataset, n_states)

    def evaluate_prompt(self, static_prompt: str, n_states: int = 8) -> Dict[str, float]:
        """
        Evaluate one static prompt using a sampled subset of states.

        The returned metrics are averaged across sampled states.
        """
        states = self.sample_states(n_states)

        format_scores: List[float] = []
        rule_scores: List[float] = []
        strategy_scores: List[float] = []
        turn_scores: List[float] = []

        for state in states:
            llm_output = self.executor.generate_turn(static_prompt=static_prompt, state=state)
            judge_result = self.judge.evaluate_turn(
                static_prompt=static_prompt,
                state=state,
                llm_output=llm_output,
            )

            format_scores.append(float(judge_result.get("format_score", 0.0)))
            rule_scores.append(float(judge_result.get("rule_score", 0.0)))
            strategy_scores.append(float(judge_result.get("strategy_score", 0.0)))
            turn_scores.append(float(judge_result.get("turn_score", 0.0)))

        return {
            "format_score": sum(format_scores) / len(format_scores),
            "rule_score": sum(rule_scores) / len(rule_scores),
            "strategy_score": sum(strategy_scores) / len(strategy_scores),
            "turn_score": sum(turn_scores) / len(turn_scores),
        }
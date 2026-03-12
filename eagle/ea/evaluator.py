from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
from .prompt_io import write_prompt_txt


@dataclass
class EvaluationConfig:
    """
    Configuration for the combined evaluator.
    """
    surrogate_states: int = 8
    real_matches: int = 3


class PromptEvaluator:
    """
    Combined evaluation interface.

    Responsibilities:
    - surrogate evaluation for cheap search-time scoring
    - real game evaluation for validation/final testing
    - fluency/readability scoring for the static prompt itself
    """

    def __init__(self, surrogate, java_runner, fluency_fn: Callable[[str], float]):
        self.surrogate = surrogate
        self.java_runner = java_runner
        self.fluency_fn = fluency_fn

    def evaluate_surrogate(self, prompt_text: str, n_states: int) -> Dict[str, float]:
        """
        Evaluate a prompt using the surrogate pipeline.
        """
        result = self.surrogate.evaluate_prompt(prompt_text, n_states=n_states)
        result["fluency"] = float(self.fluency_fn(prompt_text))
        return result

    def evaluate_real(self, prompt_text: str, n_matches: int) -> Dict[str, float]:
        """
        Evaluate a prompt by writing prompt.txt and then running real Java matches.

        Assumption:
        - On the Java side, our evolving agent is "player 0".
        """
        write_prompt_txt(prompt_text)

        wins = 0
        enemy_kills = []
        game_lengths = []
        successes = 0

        for _ in range(n_matches):
            match_result = self.java_runner.run_match()

            if match_result.get("success", False):
                successes += 1

            if match_result.get("winner") == 0:
                wins += 1

            if match_result.get("enemy_kills") is not None:
                enemy_kills.append(float(match_result["enemy_kills"]))

            if match_result.get("game_length") is not None:
                game_lengths.append(float(match_result["game_length"]))

        return {
            "win_rate": wins / n_matches if n_matches > 0 else 0.0,
            "avg_enemy_kills": sum(enemy_kills) / len(enemy_kills) if enemy_kills else 0.0,
            "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0.0,
            "successful_match_ratio": successes / n_matches if n_matches > 0 else 0.0,
        }
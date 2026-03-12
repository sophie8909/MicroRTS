from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from .component_pool import ComponentPool
from .config import EAConfig
from .evaluator import PromptEvaluator
from .java_runner import JavaEvalConfig, JavaGameRunner
from .moead import MOEAD
from .nsga2 import NSGA2
from .problem import MicroRTSPromptProblem
from .surrogate import SurrogateEvaluator


# =========================================================
# Utility functions
# =========================================================

def get_repo_root() -> Path:
    """
    Return the MICRORTS repository root.
    """
    return Path(__file__).resolve().parents[2]


def load_state_dataset(path_str: str) -> List[Dict[str, Any]]:
    """
    Load surrogate states from:
    - .json  : a list of dict objects
    - .jsonl : one dict per line

    If the path is empty or missing, a small fallback dataset is returned.
    """
    if not path_str:
        return default_state_dataset()

    path = get_repo_root() / path_str
    if not path.exists():
        print(f"[WARN] State dataset not found: {path}")
        print("[WARN] Falling back to the built-in demo state dataset.")
        return default_state_dataset()

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON state dataset must be a list of dicts.")
        return data

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    raise ValueError(f"Unsupported state dataset format: {path.suffix}")


def default_component_pool() -> ComponentPool:
    """
    Return a built-in fallback component pool.

    This lets the framework run even before the external JSON file is ready.
    """
    return ComponentPool(
        pools={
            "role": [
                "You are an AI playing a real-time strategy game. You control ALLY units only.",
                "You are a MicroRTS agent and must issue actions only for ALLY-controlled units.",
            ],
            "critical_rules": [
                "CRITICAL RULES:\nYou can ONLY command units marked as \"Ally\".\nNEVER command \"Enemy\" or \"Neutral\" units.\nEach move must be valid and legal for the current game state.",
                "CRITICAL RULES:\nOnly issue commands for Ally units.\nEvery move must include all required fields and match the unit position exactly.",
            ],
            "actions": [
                "ACTIONS:\n- move((x, y))\n- harvest((resource_x, resource_y), (base_x, base_y))\n- train(unit_type)\n- build((x, y), building_type)\n- attack((enemy_x, enemy_y))",
                "AVAILABLE ACTIONS:\nUse exact MicroRTS action syntax for move, harvest, train, build, and attack.",
            ],
            "json_schema": [
                "Return only JSON with keys: thinking and moves.",
                "Output valid JSON only. Each move object must contain raw_move, unit_position, unit_type, and action_type.",
            ],
            "field_requirements": [
                "Every move object must include all four required fields and unit_position must match an Ally unit in the state.",
                "Do not include markdown or extra commentary outside the JSON object.",
            ],
            "examples": [
                "Example move: {\"raw_move\": \"(2, 1): base train(worker)\", \"unit_position\": [2, 1], \"unit_type\": \"base\", \"action_type\": \"train\"}",
                "Example move: {\"raw_move\": \"(1, 1): worker harvest((0, 0), (2, 1))\", \"unit_position\": [1, 1], \"unit_type\": \"worker\", \"action_type\": \"harvest\"}",
            ],
            "strategy": [
                "Prioritize economy first, then transition into military production while keeping workers productive.",
                "Respond to nearby threats first, then pressure the enemy with efficient unit production.",
            ],
        }
    )


def default_state_dataset() -> List[Dict[str, Any]]:
    """
    Return a tiny built-in surrogate dataset for debugging.
    """
    return [
        {"state": "worker near resource, base idle, enemy far away"},
        {"state": "barracks ready, one ranged unit near center, enemy approaching"},
        {"state": "two workers idle, enough resources for unit production"},
        {"state": "enemy pressure on the front line, base remains intact"},
    ]


def simple_fluency_score(prompt_text: str) -> float:
    """
    A simple prompt fluency/readability proxy.

    This is intentionally lightweight.
    You can later replace it with:
    - perplexity
    - judge LLM score
    - hand-crafted readability metrics

    Larger is better.
    """
    num_words = max(1, len(prompt_text.split()))

    # Prefer a moderate prompt length instead of extremely short or extremely long prompts.
    target = 120
    score = 1.0 / (1.0 + abs(num_words - target) / target)
    return float(score)


# =========================================================
# Dummy executor and judge
# Replace these with your real model integrations later.
# =========================================================

class DummyExecutor:
    """
    Minimal executor for debugging.

    Real implementation suggestion:
    - Build full model input using:
        static prompt + current game state
    - Call Gemini / Ollama / local LLM / API
    - Return the one-turn response for surrogate evaluation
    """

    def generate_turn(self, static_prompt: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a fake one-turn response.
        """
        return {
            "thinking": "Gather resources, maintain economy, and avoid illegal moves.",
            "moves": [],
            "raw_state": state,
            "prompt_preview": static_prompt[:100],
        }


class DummyJudge:
    """
    Minimal judge for debugging.

    Real implementation suggestion:
    - Check JSON/schema validity
    - Check legality or consistency with current state
    - Check whether the output matches intended strategy
    - Score turn quality
    """

    def evaluate_turn(self, static_prompt: str, state: Dict[str, Any], llm_output: Any) -> Dict[str, float]:
        """
        Return fake surrogate scores in [0, 1].
        """
        # This is a simple deterministic-ish scoring trick for debugging.
        state_text = str(state).lower()
        base_score = 0.6

        if "resource" in state_text:
            base_score += 0.1
        if "enemy" in state_text:
            base_score += 0.1

        return {
            "format_score": 1.0,
            "rule_score": 0.9,
            "strategy_score": min(1.0, base_score),
            "turn_score": min(1.0, base_score - 0.05),
        }


def dummy_fill_mask(masked_text: str) -> str:
    """
    Minimal filler for token-level mutation.

    Real implementation suggestion:
    - Use masked language modeling
    - Use an LLM to rewrite/fill the text
    """
    return masked_text.replace("[MASK]", "carefully")


# =========================================================
# Framework setup
# =========================================================

def build_component_pool(cfg: EAConfig) -> ComponentPool:
    """
    Load component pools from JSON if available.
    Otherwise use the built-in fallback.
    """
    path = get_repo_root() / cfg.components_json_path
    if path.exists():
        return ComponentPool.from_json(path)

    print(f"[WARN] Component JSON not found: {path}")
    print("[WARN] Falling back to the built-in demo component pool.")
    return default_component_pool()


def build_surrogate(cfg: EAConfig) -> SurrogateEvaluator:
    """
    Build the surrogate evaluator.
    """
    state_dataset = load_state_dataset(cfg.surrogate_states_path)
    executor = DummyExecutor()
    judge = DummyJudge()
    return SurrogateEvaluator(
        state_dataset=state_dataset,
        executor=executor,
        judge=judge,
    )


def build_java_runner(cfg: EAConfig) -> JavaGameRunner:
    """
    Build the Java game runner.

    Important:
    - This assumes the Java code already knows how to read MICRORTS/prompt.txt.
    """
    return JavaGameRunner(
        JavaEvalConfig(
            java_cmd=cfg.java_cmd,
            classpath=cfg.java_classpath,
            main_class=cfg.java_main_class,
            timeout_sec=cfg.java_timeout_sec,
            workdir=str(get_repo_root()),
        )
    )


def build_problem(cfg: EAConfig) -> MicroRTSPromptProblem:
    """
    Build the full problem object used by the EA algorithm.
    """
    component_pool = build_component_pool(cfg)
    surrogate = build_surrogate(cfg)
    java_runner = build_java_runner(cfg)

    evaluator = PromptEvaluator(
        surrogate=surrogate,
        java_runner=java_runner,
        fluency_fn=simple_fluency_score,
    )

    return MicroRTSPromptProblem(
        cfg=cfg,
        component_pool=component_pool,
        evaluator=evaluator,
        fill_mask_fn=dummy_fill_mask,
    )


# =========================================================
# Entry point
# =========================================================

def main() -> None:
    """
    Main entry point.

    Usage idea:
    - Start with the built-in dummy executor/judge
    - Confirm the evolutionary loop works
    - Replace DummyExecutor and DummyJudge with your real surrogate pipeline
    - Ensure Java side reads MICRORTS/prompt.txt
    """
    cfg = EAConfig()

    # Example:
    # For simplified MOEA/D, use exactly two objectives.
    # Uncomment the following two lines if needed.
    #
    # cfg.algorithm = "moead"
    # cfg.objective_names = ["strategy_score", "turn_score"]

    random.seed(cfg.seed)

    problem = build_problem(cfg)

    if cfg.algorithm.lower() == "nsga2":
        algo = NSGA2(problem=problem, cfg=cfg)
        final_solutions = algo.run()

    elif cfg.algorithm.lower() == "moead":
        algo = MOEAD(problem=problem, cfg=cfg)
        final_solutions = algo.run()

    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

    print("\n=== Final Solutions ===")
    for idx, ind in enumerate(final_solutions):
        print(f"\n--- Solution {idx} ---")
        print("Objectives:", ind.objectives)
        print("Prompt:")
        print(ind.render_prompt())

        # Print any real game metrics if they exist.
        real_keys = ["win_rate", "avg_enemy_kills", "avg_game_length", "successful_match_ratio"]
        available_real = {k: ind.metadata[k] for k in real_keys if k in ind.metadata}
        if available_real:
            print("Real Evaluation:", available_real)


if __name__ == "__main__":
    main()
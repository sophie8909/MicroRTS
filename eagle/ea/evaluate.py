"""
This module defines the evaluation framework for the evolutionary algorithm. It includes the Evaluator class, which evaluates the fitness of candidate prompts by simulating games in MicroRTS and measuring performance against a baseline strategy. The Evaluator uses the ComponentPool to construct prompts based on selected components and runs multiple simulations to obtain an average fitness score. This evaluation process guides the evolution of prompts towards more effective strategies in MicroRTS.z
"""

from __future__ import annotations
from pathlib import Path

from .llm import LLM
from .component_pool import ComponentPool
from .individual import Individual
from .log_parse import parse_log


class Evaluator:
    def __init__(self, component_pool: ComponentPool):
        self.component_pool = component_pool
        self.repo_root = Path(__file__).resolve().parents[2]
    
    def evaluate(self, individual: Individual, real_eva: bool, opponent: str):
        # Construct the prompt based on the individual's components
        prompt = self.construct_prompt(individual)

        # put prompt in a temporary file
        prompt_path = self.repo_root / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        # Simulate games in MicroRTS using the constructed prompt and measure performance
        # if real_eva:
        fitness = self.simulate_games(opponent)
        # else:
        #     fitness = self.surrogate_evaluation(prompt)

        individual.fitness = fitness
        

    def construct_prompt(self, individual: Individual) -> str:
        # Use the individual's component indices to retrieve the corresponding components from the component pool and construct a prompt string
        prompt_lines: list[str] = []

        if self.component_pool.has_category('role'):
            prompt_lines.extend(self.component_pool.get_component('role', individual.role))
        if self.component_pool.has_category('critical_rules'):
            prompt_lines.extend(self.component_pool.get_component('critical_rules', individual.critical_rules))
        if self.component_pool.has_category('actions'):
            prompt_lines.extend(self.component_pool.get_component('actions', individual.actions))
        if self.component_pool.has_category('json_schema'):
            prompt_lines.extend(self.component_pool.get_component('json_schema', individual.json_schema))
        if self.component_pool.has_category('field_requirements'):
            prompt_lines.extend(self.component_pool.get_component('field_requirements', individual.field_requirements))
        if self.component_pool.has_category('examples'):
            prompt_lines.extend(self.component_pool.get_component('examples', individual.examples))

        strategy_components = [
            line
            for strategy in self.component_pool.strategy_keys
            if strategy in individual.strategy
            for line in self.component_pool.get_strategy_component(strategy, individual.strategy[strategy])
        ]
        # Combine the components into a single prompt string (this is a simplified example, the actual construction may be more complex)
        prompt = "\n".join(prompt_lines + strategy_components)
        return prompt
    
    def game_round_available_evaluation(self, log_content: str) -> float:
        # An alternative evaluation method that analyzes the log content from a MicroRTS game to compute a fitness score based on the game rounds and available actions. This can provide a more granular assessment of the agent's performance throughout the game, rather than just the final outcome.
        
        # Parse the log content to extract move results and compute fitness based on the number of successful moves, available actions, and game rounds.
        parsed_log = parse_log(log_content)
        # print(f"Parsed log: {parsed_log}")
        summary = parsed_log["summary"]
        # print(f"Parsed log summary: {summary}")
        llm_moves = summary["llm_move_count"]
        direct_failure_count = summary["direct_failure_count"]
        duplicate_skipped_count = summary["duplicate_skipped_count"]
        applied_failure_count = summary["applied_failure_count"]
        applied_success_count = summary["applied_success_count"]

        # fitness for game_round_available_evaluation
        # fitness: [0, 1]
        if llm_moves == 0:
            return 0.0
        fitness = (applied_success_count + 0.5 * applied_failure_count - 0.1 * duplicate_skipped_count - 0.3 * direct_failure_count) / llm_moves

        return fitness

    def win_loss_evaluation(self, log_content: str) -> float:
        # win = 1, loss = 0, draw = 0.5
        winning_score = 0.5  # Default to draw if no winner is found
        # find "WINNER: " in the log content
        for line in log_content.splitlines():
            if "WINNER: " in line:
                winner = line.split("WINNER: ")[1].strip()
                if winner == "0":  # Assuming Player1 is our agent
                    winning_score = 1.0  # Win
                else:
                    winning_score = 0.0  # Loss
        return winning_score
    
    def number_of_turns_evaluation(self, log_content: str) -> int:
        # parse the log content to get the number of turns in the game
        number_of_turns = 0    
        for line in log_content.splitlines():
            if "current time" in line:
                parts = line.split()
                try:
                    number_of_turns = int(parts[2])  # Assuming the format is consistent
                except ValueError:
                    pass  # If parsing fails, keep number_of_turns as 0
        
        score = number_of_turns / 1000.0  # Normalize the score (assuming 1000 turns is a reasonable upper bound)
        return score

    def calculate_fitness_score(self, log_content: str) -> float:
        
        winning_score = self.win_loss_evaluation(log_content)
        number_of_turns_score = self.number_of_turns_evaluation(log_content)
        game_round_score = self.game_round_available_evaluation(log_content)  # This can be used as an additional metric if desired

        print(f"Parsed fitness: winning_score={winning_score}, number_of_turns={number_of_turns_score}, game_round_fitness={game_round_score}")

        # fitness
        # v1: winning_score
        # v2: winning_score + number_of_turns (the more turns, the better when tie)
        # v3: winning_score + number_of_turns + game_round_fitness (consider both final outcome and in-game performance)
        # fitness = winning_score * 0.6 + game_round_score * 0.4

        return [winning_score, number_of_turns_score, game_round_score]

    def set_opponent(self, opponent: str):
        # Set the opponent strategy for the next simulation runs (this can be used to evaluate the evolved prompts against different baseline strategies in MicroRTS)
        # This function can modify a configuration file or set an environment variable that the MicroRTS simulation reads to determine the opponent strategy.
        config_path = self.repo_root / "resources" / "config.properties"
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        with open(config_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("AI2="):
                    f.write(f"AI2={opponent}\n")
                else:
                    f.write(line)

    def run_simulation(self) -> float:
        # call MicroRTS/RunLoop.sh to run
        import subprocess
        run_loop = self.repo_root / "RunLoop.sh"
        process = subprocess.Popen(
            [str(run_loop)],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            if stderr:
                print(stderr)
            return 0.0
        
    def get_latest_log_file(self) -> Path:
        # when the game end, read the result in MicroRTS/logs/run_2026-MM-DD_HH-MM-SS.log (the latest log file) to get the fitness score
        import glob
        log_files = glob.glob(str(self.repo_root / "logs" / "run_*.log"))
        if not log_files:
            return None
        latest_log_file = sorted(log_files)[-1]
        return Path(latest_log_file)

    def simulate_games(self, opponent: str) -> float:
        # Simulate multiple games in MicroRTS using the provided prompt and return an average fitness score based on performance against a baseline strategy

        self.set_opponent(opponent)
        self.run_simulation()
        

        # when the game end, read the result in MicroRTS/logs/run_2026-MM-DD_HH-MM-SS.log (the latest log file) to get the fitness score
        # 
        latest_log_file = self.get_latest_log_file()
        print(f"Testing parse_fitness with log file: {latest_log_file}")
        with open(latest_log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
        # parse the log content to get the fitness score
        fitness = self.calculate_fitness_score(log_content)
        return fitness
    
    def surrogate_evaluation(self, prompt: str) -> float:
        return LLM.ollama_evaluate_fitness(prompt)
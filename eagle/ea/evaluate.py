"""
This module defines the evaluation framework for the evolutionary algorithm. It includes the Evaluator class, which evaluates the fitness of candidate prompts by simulating games in MicroRTS and measuring performance against a baseline strategy. The Evaluator uses the ComponentPool to construct prompts based on selected components and runs multiple simulations to obtain an average fitness score. This evaluation process guides the evolution of prompts towards more effective strategies in MicroRTS.z
"""

from __future__ import annotations
from pathlib import Path
from .component_pool import ComponentPool
from .individual import Individual


class Evaluator:
    def __init__(self, component_pool: ComponentPool):
        self.component_pool = component_pool
        self.repo_root = Path(__file__).resolve().parents[2]
    
    def evaluate(self, individual: Individual) -> float:
        # Construct the prompt based on the individual's components
        prompt = self.construct_prompt(individual)

        # put prompt in a temporary file
        prompt_path = self.repo_root / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        # Simulate games in MicroRTS using the constructed prompt and measure performance
        fitness = self.simulate_games(prompt)
        return fitness
    
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
            for i, strategy in enumerate(self.component_pool.strategy_keys)
            for line in self.component_pool.get_strategy_component(strategy, individual.strategy[i])
        ]
        # Combine the components into a single prompt string (this is a simplified example, the actual construction may be more complex)
        prompt = "\n".join(prompt_lines + strategy_components)
        return prompt
    
    def parse_fitness(self, log_content: str) -> float:
        # Parse the log content from MicroRTS to extract the fitness score 

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


        number_of_turns = 0    
        # find current time 2 p0 player 0(5) p1 player 1(5) in the log content to get the number of turns
        for line in log_content.splitlines():
            if "current time" in line:
                parts = line.split()
                # print(f"Debug: parts of current time line: {parts}")
                try:
                    number_of_turns = int(parts[2])  # Assuming the format is consistent
                except ValueError:
                    pass  # If parsing fails, keep number_of_turns as 0
        print(f"Parsed fitness: winning_score={winning_score}, number_of_turns={number_of_turns}")
        
        # fitness
        # v1: winning_score
        # v2: winning_score + number_of_turns (the more turns, the better when tie)
        fitness = winning_score
        if winning_score == 0.5:  # Draw
            # 10 when game time is 60 sec
            fitness = winning_score + number_of_turns / 10  # Add a bonus for more turns in a draw

        return fitness


    def simulate_games(self, prompt: str) -> float:
        # Simulate multiple games in MicroRTS using the provided prompt and return an average fitness score based on performance against a baseline strategy
        
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

        # when the game end, read the result in MicroRTS/logs/run_2026-MM-DD_HH-MM-SS.log (the latest log file) to get the fitness score
        # 
        import glob
        import os
        log_files = glob.glob(str(self.repo_root / "logs" / "run_*.log"))
        if not log_files:
            return 0.0
        latest_log_file = sorted(log_files)[-1]
        print(f"Testing parse_fitness with log file: {latest_log_file}")
        with open(latest_log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
        # parse the log content to get the fitness score
        fitness = self.parse_fitness(log_content)
        return fitness
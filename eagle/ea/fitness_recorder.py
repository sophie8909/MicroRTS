"""
Record fitness evaluation results and related metadata for each evaluated individual in a structured format (e.g., JSONL) for later analysis and visualization.
"""

from pathlib import Path
from typing import Any

class FitnessRecorder:
    def __init__(self, log_folder: Path):
        self.log_path = log_folder / "fitness_records.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []

    def record_fitness(self, record: dict[str, Any]):
        # Append the fitness evaluation record to a JSONL file for later analysis
        # Each record should include the individual's component configuration, fitness score, and any relevant metadata (
        # e.g., opponent faced, generation number, evaluation time)
        """Example record structure:
        {
            "individual_id": "ind-0",
            "generation": 1,
            "fitness_score": [0.8, 0.5, 0.3],
            "opponent": "SimpleBot",
            "evaluation_time": 12.5,
            "components": {
                "critical_rules": 2,
                "actions": 1,
                "json_schema": 0,
                "field_requirements": 3,
                "examples": 1,
                "role": 0,
                "strategy": {
                    "resource_gathering": 1,
                    "unit_production": 0,
                    "combat_strategy": 2
                    ...
                }
            }
        }
        """
        self.records.append(record)
        
        self.records = self.records[-50:]  # Keep only the last 50 records
        
        with self.log_path.open("w", encoding="utf-8") as f:
            f.write(f"{self.records}\n")


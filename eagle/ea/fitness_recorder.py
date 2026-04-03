"""
Record fitness evaluation results and related metadata for each evaluated individual in a structured format (e.g., JSONL) for later analysis and visualization.
"""

import json
from pathlib import Path
from typing import Any

class FitnessRecorder:
    def __init__(self, log_folder: Path):
        self.log_path = log_folder / "fitness_records.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.history_records_path =  "fitness_history.jsonl"
        self.history = []  
    
    def init_from_history(self):
        if Path(self.history_records_path).exists():
            with open(self.history_records_path, "r", encoding="utf-8") as f:
                self.history = [json.loads(line) for line in f]
        else:
            self.history = []
    
    def add_history_record(self, record: dict[str, Any]):
        self.history.append(record)
        with open(self.history_records_path, "w", encoding="utf-8") as f:
            f.write("\n".join([json.dumps(r) for r in self.history]))

    def turn_record_to_history(self, record: dict[str, Any]):
        # only keep prompt hash and fitness score for history record to save space.
        prompt_hash = hash(json.dumps(record["prompt"], sort_keys=True))
        history_record = {
            "prompt_hash": prompt_hash,
            "fitness_score": record["fitness_score"]
        }
        return history_record


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
            f.write("\n".join([json.dumps(r) for r in self.records]))

        self.add_history_record(self.turn_record_to_history(record))


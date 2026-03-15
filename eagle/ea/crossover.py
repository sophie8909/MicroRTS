"""Crossover methods for the genetic algorithm."""

from __future__ import annotations

from .individual import Individual

class Crossover:
    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual) -> Individual:
        import random
        child = Individual()
        child.role = random.choice([parent1.role, parent2.role])
        child.critical_rules = random.choice([parent1.critical_rules, parent2.critical_rules])
        child.actions = random.choice([parent1.actions, parent2.actions])
        child.json_schema = random.choice([parent1.json_schema, parent2.json_schema])
        child.field_requirements = random.choice([parent1.field_requirements, parent2.field_requirements])
        child.examples = random.choice([parent1.examples, parent2.examples])
        child.strategy = [
            random.choice([parent1.strategy[i], parent2.strategy[i]])
            for i in range(len(parent1.strategy))
        ]
        return child
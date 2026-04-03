import random
from typing import List

import requests
import re
import ast

from .fitness_utils import normalize_fitness

class LLM:

    @staticmethod
    def ollama_rewrite_component(
            original_text: str,
            instruction: str,
            model: str = "llama3.1:8b",
            temperature: float = 0.7,
        ) -> str:
        prompt = f"""
        You are rewriting one component of a prompt for an RTS game-playing agent.

        Requirements:
        - Preserve the original semantic intent unless the instruction explicitly changes it.
        - Return ONLY the rewritten component text.
        - Do not add explanations, bullets, or quotation marks.

        Rewrite instruction:
        {instruction}

        Original component:
        {original_text}
        """.strip()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
    
    def ollama_combine_components(
            component1: str,
            component2: str,
            instruction: str,
            model: str = "llama3.1:8b",
            temperature: float = 0.7,
        ) -> str:
        prompt = f"""
        You are combining two components of a prompt for an RTS game-playing agent.

        Requirements:
        - Integrate the key elements of both components while following the instruction.
        - Ensure the combined component is coherent and maintains the original intent.
        - Return ONLY the combined component text without explanations or formatting.

        Combine instruction:
        {instruction}

        Component 1:
        {component1}

        Component 2:
        {component2}
        """.strip()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()


    @staticmethod
    def ollama_evaluate_fitness(
        prompt: str,
        example=None,
        model: str = "llama3.1:8b",
    ) -> list[float]:
        if example is None:
            example = []

        example_str = "\n\n".join(
            [f"Input:\n{inp}\nOutput:\n{out}" for inp, out in example]
        )

        evaluation_prompt = f"""
        You are evaluating a prompt used to instruct an LLM-based MicroRTS agent.

        Your task is NOT to assume the agent will win just because the prompt sounds strategic.
        You must be conservative.

        Evaluate the prompt on the following four dimensions:

        1. estimated_power:
        - Estimate how likely this prompt is to produce useful in-game decisions in MicroRTS.
        - Consider whether it gives actionable, executable, and game-relevant guidance.
        - Do NOT assume high power unless the prompt contains concrete and operational strategy guidance.
        - Score between 0 and 1.

        2. uncertainty:
        - Estimate how uncertain you are about the power prediction based on prompt text alone.
        - High uncertainty means the prompt might sound good but there is insufficient evidence that it will perform well in practice.
        - Score between 0 and 1, where 1 means highly uncertain.

        3. simplicity:
        - Is the prompt concise and free of unnecessary wording?
        - Score between 0 and 1.

        4. clarity:
        - Is the prompt precise, unambiguous, and easy for an LLM to follow?
        - Score between 0 and 1.

        Scoring rules:
        - Be conservative.
        - Do not give estimated_power > 0.8 unless the prompt contains concrete operational instructions that are directly useful for MicroRTS decision-making.
        - If the prompt is vague, generic, verbose, or only superficially strategic, reduce estimated_power and increase uncertainty.
        - Use the full range of scores, but avoid extreme values unless strongly justified.
        - Output only a Python-style list:
        [estimated_power, uncertainty, simplicity, clarity]

        {example_str}

        Input:
        {prompt}
        Output:
        """.strip()

        fallback_score = [0.0, 1.0, 0.0, 0.0]

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        def normalize_fitness(values) -> list[float]:
            """
            Normalize the surrogate output to:
            [estimated_power, uncertainty, simplicity, clarity]
            """
            if not isinstance(values, (list, tuple)):
                return fallback_score

            values = list(values)

            if len(values) < 4:
                values = values + [0.0] * (4 - len(values))
            elif len(values) > 4:
                values = values[:4]

            normalized = [clamp01(v) for v in values]

            # Keep uncertainty conservative if parsing gave something weird.
            # If all zeros, treat it as unreliable and restore fallback.
            if normalized == [0.0, 0.0, 0.0, 0.0]:
                return fallback_score

            return normalized

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": evaluation_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                    },
                },
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()
            raw_output = data.get("response", "").strip()

            # Step 1: Parse a Python-style list directly.
            try:
                parsed = ast.literal_eval(raw_output)
                return normalize_fitness(parsed)
            except (ValueError, SyntaxError):
                pass

            # Step 2: Regex fallback for bracketed or noisy output.
            matches = re.findall(r"-?\d*\.\d+|-?\d+", raw_output)
            if matches:
                parsed = [float(x) for x in matches[:4]]
                return normalize_fitness(parsed)

            # Step 3: Final fallback.
            return fallback_score

        except requests.exceptions.RequestException:
            return fallback_score

        except Exception:
            return fallback_score

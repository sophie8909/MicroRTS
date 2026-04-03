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
    def ollama_evaluate_fitness(prompt: str, example=None, model: str = "llama3.1:8b",):
        if example is None:
            example = []
        example_str = "\n".join([f"Input:\n {inp}\nOutput:\n {out}" for inp, out in example])
        
        # print(f"Evaluating prompt with LLM:Example:\n{example_str}")
        evaluation_prompt = f"""
        You are evaluating the quality of a prompt for an RTS game-playing agent.
        The prompt is designed to instruct an LLM to generate strategies for playing MicroRTS, a real-time strategy game.
        The evaluation should consider:
        1. Power: Win (1.0) or not (0.0)
        2. Simplicity: Is the prompt concise and straightforward, without unnecessary complexity or verbosity?
        3. Clarity: Is the prompt clear and unambiguous for an LLM to understand and follow?

        only return a list with 3 elements: [power_score, simplicity_score, clarity_score], each is a float between 0 and 1, where higher is better.
        
        {example_str}

        Input:
        {prompt}
        Output:
        """.strip()

        fallback_score = [0.0, 0.0, 0.0]
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": evaluation_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                    },
                },
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()
            raw_output = data.get("response", "").strip()
            # print(f"LLM evaluation raw output: {raw_output}")

            # Step 1: parse a Python-style list directly.
            try:
                parsed = ast.literal_eval(raw_output)
                return normalize_fitness(parsed)
            except (ValueError, SyntaxError):
                pass

            # Step 2: backward-compatible single float parse.
            try:
                score = float(raw_output)
                return normalize_fitness(score)
            except ValueError:
                pass

            # Step 3: regex fallback (extract up to 3 numeric values).
            matches = re.findall(r"-?\d*\.\d+|-?\d+", raw_output)
            if matches:
                return normalize_fitness(matches[:3])

            # Step 4: final fallback.
            return fallback_score

        except requests.exceptions.RequestException:
            # network / timeout / connection error
            return fallback_score

        except Exception:
            # anything unexpected
            return fallback_score

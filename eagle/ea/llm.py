import random
from typing import List

import requests
import re

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
    
 

    @staticmethod
    def ollama_evaluate_fitness(prompt: str, example=None):
        if example is None:
            example = []
        example_str = "\n".join([f"Input: {inp}\nOutput: {out}" for inp, out in example])
        
        evaluation_prompt = f"""
        You are evaluating the quality of a prompt for an RTS game-playing agent.
        The prompt is designed to instruct an LLM to generate strategies for playing MicroRTS, a real-time strategy game.
        The evaluation should consider:
        1. Power: Win (1.0) or not (0.0)
        2. Simplicity: Is the prompt concise and straightforward, without unnecessary complexity or verbosity?
        3. Clarity: Is the prompt clear and unambiguous for an LLM to understand and follow?

        only return a list with 3 elements: [power_score, simplicity_score, clarity_score], each is a float between 0 and 1, where higher is better.
        Example:
        {example_str}
        Prompt to evaluate:
        {prompt}
        """.strip()

        fallback_score = 0.0  # Default fallback score if parsing fails
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder:7b",
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
            # -------------------------
            # Step 1: direct float parse
            # -------------------------
            try:
                score = float(raw_output)
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

            # -------------------------
            # Step 2: regex fallback (extract first float)
            # -------------------------
            match = re.search(r"\d*\.\d+|\d+", raw_output)
            if match:
                try:
                    score = float(match.group())
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass

            # -------------------------
            # Step 3: final fallback
            # -------------------------
            return fallback_score

        except requests.exceptions.RequestException:
            # network / timeout / connection error
            return fallback_score

        except Exception:
            # anything unexpected
            return fallback_score
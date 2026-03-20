import requests

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
    def ollama_evaluate_fitness(prompt: str, model: str = "llama3.1:8b", temperature: float = 0.7) -> float:
        evaluation_prompt = f"""
        You are evaluating the fitness of a prompt for an RTS game-playing agent based on its expected performance in a MicroRTS game.

        Requirements:
        - Analyze the provided prompt and assign a fitness score between 0 and 1, where 1 indicates a highly effective prompt that is likely to lead to strong performance in the game, and 0 indicates a poor prompt that is unlikely to perform well.
        - Consider factors such as clarity, strategic depth, adaptability, and alignment with MicroRTS game dynamics when assigning the fitness score.
        - Return ONLY the fitness score as a decimal number between 0 and 1. Do not include any explanations or additional text.

        Prompt to evaluate:
        {prompt}
        """.strip()

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": evaluation_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        fitness_score_str = data["response"].strip()
        try:
            fitness_score = float(fitness_score_str)
            return max(0.0, min(1.0, fitness_score))  # Ensure the score is between 0 and 1
        except ValueError:
            raise ValueError(f"Invalid fitness score returned by LLM: '{fitness_score_str}'")
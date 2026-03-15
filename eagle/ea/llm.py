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
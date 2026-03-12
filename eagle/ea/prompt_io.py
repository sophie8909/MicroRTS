from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """
    Return the MICRORTS repository root.

    File layout assumption:
    MICRORTS/
      eagle/
        ea/
          prompt_io.py

    Therefore:
    prompt_io.py -> ea -> eagle -> MICRORTS
    """
    return Path(__file__).resolve().parents[2]


def get_prompt_txt_path() -> Path:
    """
    Return the path of the prompt.txt file in the repository root.
    """
    return get_repo_root() / "prompt.txt"


def write_prompt_txt(prompt_text: str) -> Path:
    """
    Write the given static prompt to MICRORTS/prompt.txt.

    This file acts as the bridge between Python-based evolution and
    Java-based real game execution.
    """
    path = get_prompt_txt_path()
    path.write_text(prompt_text, encoding="utf-8")
    return path
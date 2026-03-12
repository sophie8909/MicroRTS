from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import re


@dataclass
class JavaEvalConfig:
    """
    Configuration for the Java-based real game runner.
    """
    java_cmd: str = "java"
    classpath: str = "lib/*:bin"
    main_class: str = "rts.MicroRTS"
    timeout_sec: int = 180
    workdir: Optional[str] = None


class JavaGameRunner:
    """
    Run MicroRTS via subprocess and parse stdout/stderr.

    Important:
    - This class assumes the Java side already knows how to read MICRORTS/prompt.txt.
    - You may need to adjust the parsing rules depending on your Java logs.
    """

    def __init__(self, cfg: JavaEvalConfig):
        self.cfg = cfg

    def run_match(self, extra_args: Optional[List[str]] = None) -> Dict:
        """
        Run one real match.

        Returns:
        - parsed metrics
        - raw stdout/stderr
        - return code
        """
        extra_args = extra_args or []

        cmd = [
            self.cfg.java_cmd,
            "-cp",
            self.cfg.classpath,
            self.cfg.main_class,
            *extra_args,
        ]

        result = subprocess.run(
            cmd,
            cwd=self.cfg.workdir,
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout_sec,
        )

        parsed = self._parse_output(result.stdout, result.stderr)
        parsed["stdout"] = result.stdout
        parsed["stderr"] = result.stderr
        parsed["returncode"] = result.returncode
        return parsed

    def _parse_output(self, stdout: str, stderr: str) -> Dict:
        """
        Parse Java output.

        Recommended Java-side log lines:
            winner=0
            enemy_kills=5
            game_length=312

        This parser is intentionally simple and easy to modify.
        """
        winner = None
        enemy_kills = None
        game_length = None

        winner_match = re.search(r"winner\s*[:=]\s*(-?\d+)", stdout, re.IGNORECASE)
        if winner_match:
            winner = int(winner_match.group(1))

        kills_match = re.search(r"enemy_kills\s*[:=]\s*(\d+)", stdout, re.IGNORECASE)
        if kills_match:
            enemy_kills = int(kills_match.group(1))

        length_match = re.search(r"game_length\s*[:=]\s*(\d+)", stdout, re.IGNORECASE)
        if length_match:
            game_length = int(length_match.group(1))

        return {
            "success": winner is not None,
            "winner": winner,
            "enemy_kills": enemy_kills,
            "game_length": game_length,
        }
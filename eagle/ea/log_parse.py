from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# =========================================================
# Regex patterns
# =========================================================

INIT_LOGS_RE = re.compile(r"^\s*initLogsIfNeeded.*$", re.MULTILINE)
GETACTION_START_RE = re.compile(r"^\s*\[(?P<agent>[^\]]+)\.getAction\]\s+start\s*$", re.MULTILINE)

RAW_RESPONSE_HEADER_RE = re.compile(r"^\s*=== Raw LLM Response ===\s*$", re.MULTILINE)
RAW_RESPONSE_FOOTER_RE = re.compile(r"^\s*=+\s*$", re.MULTILINE)

RUNNING_GETACTION_RE = re.compile(r"^\s*Running getAction for Player:\s*(?P<player>\d+)\s*$", re.MULTILINE)
CURRENT_TIME_RE = re.compile(
    r"^\s*current time\s+(?P<time>\d+)\s+p0 player\s+(?P<p0_player>\d+)\((?P<p0_value>\d+)\)\s+p1 player\s+(?P<p1_player>\d+)\((?P<p1_value>\d+)\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
SCOREBOARD_RE = re.compile(
    r"^\s*T:\s*(?P<T>\d+),\s*P0:\s*(?P<P0_player>\d+)\s*\((?P<P0_value>\d+)\),\s*P1:\s*(?P<P1_player>\d+)\s*\((?P<P1_value>\d+)\)\s*$",
    re.MULTILINE,
)
GAMEOVER_RE = re.compile(r"^\s*gs\.gameover\(\)\s*=\s*(?P<value>true|false)\s*$", re.MULTILINE)

APPLY_MOVE_RE = re.compile(
    r"^\s*Applying LLM move:\s*(?P<raw_move>.+?)\s*\|\s*action_type=(?P<action_type>\w+)\s*\|\s*unit=\((?P<ux>-?\d+),(?P<uy>-?\d+)\)\s*type=(?P<unit_type>.+?)\s*$"
)

ACTION_FAILED_RE = re.compile(
    r"^\s*'?(?P<prefix>\w+)'?\s+failed:\s*(?P<reason>.+?)\s*$",
    re.IGNORECASE,
)

NON_OWNED_RE = re.compile(
    r"Can't command non-owned unit at\s*\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)",
    re.IGNORECASE,
)

FALLBACK_NO_MOVES_RE = re.compile(
    r"^\s*\[LLM\]\s+No moves\[\]\s+in response\..*$",
    re.MULTILINE,
)

SKIP_RE = re.compile(
    r"^\s*(?:⚠️\s*)?Skipping\s+.+$",
    re.IGNORECASE,
)


# =========================================================
# Data classes
# =========================================================

@dataclass
class MoveResult:
    segment_index: int
    move_index: int
    agent: str | None
    current_time: int | None
    player: int | None

    llm_move_raw: dict[str, Any] | None
    raw_move: str | None
    unit_type: str | None
    action_type: str | None
    unit_position: list[int] | None

    status: str
    failure_reason: str | None

    has_apply_log: bool
    apply_log: str | None
    result_log: str | None


# =========================================================
# Utility functions
# =========================================================

def split_segments_by_initlogs(log_text: str) -> list[str]:
    """
    Split the log into segments by initLogsIfNeeded.
    """
    matches = list(INIT_LOGS_RE.finditer(log_text))
    if not matches:
        return [log_text.strip()] if log_text.strip() else []

    segments: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(log_text)
        segment = log_text[start:end].strip()
        if segment:
            segments.append(segment)
    return segments


def detect_agent_name(segment: str) -> str | None:
    """
    Detect the agent name from [Agent.getAction] start.
    """
    match = GETACTION_START_RE.search(segment)
    if not match:
        return None
    return match.group("agent")


def is_target_agent_block(segment: str, target_agent: str = "EAGLE") -> bool:
    """
    Check whether the segment belongs to the target agent.
    """
    agent = detect_agent_name(segment)
    if agent is None:
        return False
    return agent.lower() == target_agent.lower()


def extract_raw_llm_response(segment: str) -> tuple[str | None, dict[str, Any] | None]:
    """
    Extract the raw LLM response JSON.
    """
    header = RAW_RESPONSE_HEADER_RE.search(segment)
    if not header:
        return None, None

    after_header = segment[header.end():]
    footer = RAW_RESPONSE_FOOTER_RE.search(after_header)
    raw_text = after_header[:footer.start()].strip() if footer else after_header.strip()

    if not raw_text:
        return None, None

    try:
        raw_json = json.loads(raw_text)
    except json.JSONDecodeError:
        raw_json = None

    return raw_text, raw_json


def split_pre_post(segment: str) -> tuple[str, str]:
    """
    Split the segment into pre-getAction and post-getAction parts.
    """
    match = RUNNING_GETACTION_RE.search(segment)
    if not match:
        return segment.strip(), ""
    return segment[:match.start()].strip(), segment[match.start():].strip()


def parse_post_fields(post_text: str) -> dict[str, Any]:
    """
    Parse player id, current time, and scoreboard.
    """
    result: dict[str, Any] = {
        "player": None,
        "current_time": None,
        "scoreboard": None,
    }

    player_match = RUNNING_GETACTION_RE.search(post_text)
    if player_match:
        result["player"] = int(player_match.group("player"))

    time_match = CURRENT_TIME_RE.search(post_text)
    if time_match:
        result["current_time"] = int(time_match.group("time"))

    scoreboard_match = SCOREBOARD_RE.search(post_text)
    if scoreboard_match:
        result["scoreboard"] = {
            "T": int(scoreboard_match.group("T")),
            "P0": {
                "player": int(scoreboard_match.group("P0_player")),
                "value": int(scoreboard_match.group("P0_value")),
            },
            "P1": {
                "player": int(scoreboard_match.group("P1_player")),
                "value": int(scoreboard_match.group("P1_value")),
            },
        }

    return result


def parse_gameover(pre_text: str) -> bool | None:
    """
    Parse gs.gameover() if present.
    """
    match = GAMEOVER_RE.search(pre_text)
    if not match:
        return None
    return match.group("value").lower() == "true"


def normalize_llm_moves(raw_llm_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Safely return the moves array from the raw LLM response.
    """
    if not isinstance(raw_llm_json, dict):
        return []

    moves = raw_llm_json.get("moves")
    if not isinstance(moves, list):
        return []

    normalized: list[dict[str, Any]] = []
    for move in moves:
        if isinstance(move, dict):
            normalized.append(move)
        else:
            normalized.append({"raw_value": move})
    return normalized


def extract_failure_reason(line: str) -> str:
    """
    Convert a failure log line into a compact reason label.
    """
    lower = line.lower()

    if "non-owned unit" in lower:
        return "non_owned_unit"
    if "non-worker" in lower:
        return "non_worker"
    if "not base/barracks" in lower:
        return "not_base_or_barracks"
    if "structure at" in lower:
        return "structure_blocked"
    if "out of bounds" in lower:
        return "out_of_bounds"
    if "resource" in lower:
        return "resource_issue"

    failed_match = ACTION_FAILED_RE.match(line.strip())
    if failed_match:
        return f"{failed_match.group('prefix').lower()}_failed"

    return "unknown_failure"


# =========================================================
# Core move-result builder
# =========================================================

def build_move_results(
    segment_index: int,
    agent: str | None,
    current_time: int | None,
    player: int | None,
    raw_llm_json: dict[str, Any] | None,
    pre_text: str,
) -> list[MoveResult]:
    """
    Build move-level results based on log events only.

    Categories:
    - direct_failed: failed before apply
    - applied_failed: apply happened, then failure
    - applied_success: apply happened, no failure
    - not_executed: no matching execution log found
    """
    llm_moves = normalize_llm_moves(raw_llm_json)
    lines = [line.rstrip() for line in pre_text.splitlines() if line.strip()]

    results: list[MoveResult] = []
    move_idx = 0
    line_idx = 0

    while move_idx < len(llm_moves):
        move = llm_moves[move_idx]

        raw_move = move.get("raw_move") if isinstance(move.get("raw_move"), str) else None
        unit_type = move.get("unit_type") if isinstance(move.get("unit_type"), str) else None
        action_type = move.get("action_type") if isinstance(move.get("action_type"), str) else None
        unit_position = move.get("unit_position") if isinstance(move.get("unit_position"), list) else None

        status = "not_executed"
        failure_reason: str | None = None
        has_apply_log = False
        apply_log: str | None = None
        result_log: str | None = None

        matched = False

        while line_idx < len(lines):
            line = lines[line_idx].strip()

            if NON_OWNED_RE.search(line):
                status = "direct_failed"
                failure_reason = extract_failure_reason(line)
                result_log = line
                matched = True
                line_idx += 1
                break

            if SKIP_RE.search(line):
                status = "duplicate_skipped"
                failure_reason = extract_failure_reason(line)
                result_log = line
                matched = True
                line_idx += 1
                break

            apply_match = APPLY_MOVE_RE.match(line)
            if apply_match:
                status = "applied_success"
                has_apply_log = True
                apply_log = line
                matched = True
                line_idx += 1

                if line_idx < len(lines):
                    next_line = lines[line_idx].strip()
                    failed_match = ACTION_FAILED_RE.match(next_line)
                    if failed_match:
                        status = "applied_failed"
                        failure_reason = extract_failure_reason(next_line)
                        result_log = next_line
                        line_idx += 1

                break

            line_idx += 1

        if not matched:
            if FALLBACK_NO_MOVES_RE.search(pre_text):
                failure_reason = "fallback_no_moves"
            else:
                failure_reason = "no_matching_execution_log"

        results.append(
            MoveResult(
                segment_index=segment_index,
                move_index=move_idx,
                agent=agent,
                current_time=current_time,
                player=player,
                llm_move_raw=move,
                raw_move=raw_move,
                unit_type=unit_type,
                action_type=action_type,
                unit_position=unit_position,
                status=status,
                failure_reason=failure_reason,
                has_apply_log=has_apply_log,
                apply_log=apply_log,
                result_log=result_log,
            )
        )

        move_idx += 1

    return results


# =========================================================
# Segment parser
# =========================================================

def parse_segment(segment: str, segment_index: int) -> dict[str, Any]:
    """
    Parse one target-agent segment.
    """
    agent = detect_agent_name(segment)
    pre_text, post_text = split_pre_post(segment)
    raw_llm_text, raw_llm_json = extract_raw_llm_response(pre_text)
    post_fields = parse_post_fields(post_text)

    move_results = build_move_results(
        segment_index=segment_index,
        agent=agent,
        current_time=post_fields["current_time"],
        player=post_fields["player"],
        raw_llm_json=raw_llm_json,
        pre_text=pre_text,
    )

    llm_move_count = len(normalize_llm_moves(raw_llm_json))
    direct_failure_count = sum(1 for m in move_results if m.status == "direct_failed")
    duplicate_skipped_count = sum(1 for m in move_results if m.status == "duplicate_skipped")
    applied_failure_count = sum(1 for m in move_results if m.status == "applied_failed")
    applied_success_count = sum(1 for m in move_results if m.status == "applied_success")

    return {
        "segment_index": segment_index,
        "agent": agent,
        "current_time": post_fields["current_time"],
        "player": post_fields["player"],
        "scoreboard": post_fields["scoreboard"],
        "gameover": parse_gameover(pre_text),
        "raw_llm_response_text": raw_llm_text,
        "raw_llm_response_json": raw_llm_json,
        "llm_move_count": llm_move_count,
        "direct_failure_count": direct_failure_count,
        "duplicate_skipped_count": duplicate_skipped_count,
        "applied_failure_count": applied_failure_count,
        "applied_success_count": applied_success_count,
        "move_results": [asdict(m) for m in move_results],
    }


# =========================================================
# Main parser
# =========================================================

def parse_log(log_text: str, target_agent: str = "EAGLE") -> dict[str, Any]:
    """
    Parse the full log and return segment-level and global summaries.
    """
    segments = split_segments_by_initlogs(log_text)

    parsed_segments: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        if is_target_agent_block(segment, target_agent=target_agent):
            parsed_segments.append(parse_segment(segment, i))

    all_moves: list[dict[str, Any]] = []
    for segment in parsed_segments:
        all_moves.extend(segment["move_results"])

    summary = {
        "target_agent": target_agent,
        "segment_count": len(parsed_segments),
        "llm_move_count": sum(s["llm_move_count"] for s in parsed_segments),
        "direct_failure_count": sum(s["direct_failure_count"] for s in parsed_segments),
        "duplicate_skipped_count": sum(s["duplicate_skipped_count"] for s in parsed_segments),
        "applied_failure_count": sum(s["applied_failure_count"] for s in parsed_segments),
        "applied_success_count": sum(s["applied_success_count"] for s in parsed_segments),
    }

    return {
        "summary": summary,
        "segments": parsed_segments,
        "all_move_results": all_moves,
    }


def parse_log_file(file_path: str | Path, target_agent: str = "EAGLE") -> dict[str, Any]:
    """
    Parse a log file from disk.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    return parse_log(text, target_agent=target_agent)


# =========================================================
# Output helpers
# =========================================================

def save_json(data: Any, output_path: str | Path) -> None:
    """
    Save parsed results as JSON.
    """
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_jsonl_move_results(parsed: dict[str, Any], output_path: str | Path) -> None:
    """
    Save move-level results as JSONL.
    """
    with Path(output_path).open("w", encoding="utf-8") as f:
        for row in parsed["all_move_results"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# if __name__ == "__main__":
#     input_path = "game.log"
#     output_json = "parsed_eagle_log.json"
#     output_jsonl = "parsed_eagle_moves.jsonl"

#     parsed = parse_log_file(input_path, target_agent="EAGLE")

#     print(json.dumps(parsed["summary"], ensure_ascii=False, indent=2))

#     save_json(parsed, output_json)
#     save_jsonl_move_results(parsed, output_jsonl)
from __future__ import annotations

import re
import shutil
from pathlib import Path


# Example: run_2026-03-27-00-00-32.log
LOG_PATTERN = re.compile(
    r"^run_(\d{4})-(\d{2})-(\d{2})_\d{2}-\d{2}-\d{2}\.log$"
)

# Example: Response2026-03-31_13-57-48_LLM_Gemini_OneShot.json
# It only relies on the leading date part after "Response"
RESPONSE_PATTERN = re.compile(
    r"^Response(\d{4})-(\d{2})-(\d{2})[_-].*"
)


def safe_move(file_path: Path, target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / file_path.name

    if target_file.exists():
        print(f"[SKIP] Target already exists: {target_file}")
        return False

    shutil.move(str(file_path), str(target_file))
    print(f"[MOVE] {file_path} -> {target_dir}/")
    return True


def organize_logs(logs_dir: str = "logs") -> None:
    base_path = Path(logs_dir)

    if not base_path.exists():
        print(f"[WARN] Logs directory does not exist: {base_path.resolve()}")
        return

    moved_count = 0
    skipped_count = 0

    for item in base_path.iterdir():
        if not item.is_file():
            continue

        match = LOG_PATTERN.match(item.name)
        if not match:
            skipped_count += 1
            continue

        _, month, day = match.groups()
        folder_name = f"{month}{day}"
        target_dir = base_path / folder_name

        if safe_move(item, target_dir):
            moved_count += 1
        else:
            skipped_count += 1

    print(f"[LOGS] moved={moved_count}, skipped={skipped_count}")


def organize_responses(responses_dir: str = "responses") -> None:
    base_path = Path(responses_dir)

    if not base_path.exists():
        print(f"[WARN] Responses directory does not exist: {base_path.resolve()}")
        return

    moved_count = 0
    skipped_count = 0

    for item in base_path.iterdir():
        if not item.is_file():
            continue

        match = RESPONSE_PATTERN.match(item.name)
        if not match:
            skipped_count += 1
            continue

        _, month, day = match.groups()
        folder_name = f"{month}{day}"
        target_dir = base_path / folder_name

        if safe_move(item, target_dir):
            moved_count += 1
        else:
            skipped_count += 1

    print(f"[RESPONSES] moved={moved_count}, skipped={skipped_count}")





def organize_microrts_responses(base_dir: str = "."):
    base_path = Path(base_dir)
    responses_root = base_path / "responses"
    responses_root.mkdir(exist_ok=True)

    moved = 0
    skipped = 0

    for file in base_path.iterdir():
        if not file.is_file():
            continue

        match = RESPONSE_PATTERN.match(file.name)
        if not match:
            continue

        _, month, day = match.groups()
        date_folder = f"{month}{day}"

        target_dir = responses_root / date_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / file.name

        if target_file.exists():
            print(f"[SKIP] {file.name}")
            skipped += 1
            continue

        shutil.move(str(file), str(target_file))
        print(f"[MOVE] {file.name} -> responses/{date_folder}/")
        moved += 1

    print(f"\nDone. moved={moved}, skipped={skipped}")


def main() -> None:
    organize_logs("logs")
    organize_responses("responses")
    organize_microrts_responses(".")
    print("[DONE] Organization finished.")



if __name__ == "__main__":
    main()
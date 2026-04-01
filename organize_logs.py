from __future__ import annotations

import re
import shutil
from pathlib import Path


# Match filenames like: run_2026-03-27-00-00-32.log
LOG_PATTERN = re.compile(r"^run_(\d{4})-(\d{2})-(\d{2})-\d{2}-\d{2}-\d{2}\.log$")


def organize_logs(base_dir: str = "logs") -> None:
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"[ERROR] Directory does not exist: {base_path.resolve()}")
        return

    if not base_path.is_dir():
        print(f"[ERROR] Not a directory: {base_path.resolve()}")
        return

    moved_count = 0
    skipped_count = 0

    for item in base_path.iterdir():
        if not item.is_file():
            continue

        match = LOG_PATTERN.match(item.name)
        if not match:
            skipped_count += 1
            print(f"[SKIP] Not a target log file: {item.name}")
            continue

        year, month, day = match.groups()
        folder_name = f"{month}{day}"
        target_dir = base_path / folder_name
        target_dir.mkdir(exist_ok=True)

        target_file = target_dir / item.name

        # Avoid overwriting existing files
        if target_file.exists():
            skipped_count += 1
            print(f"[SKIP] Target already exists: {target_file}")
            continue

        shutil.move(str(item), str(target_file))
        moved_count += 1
        print(f"[MOVE] {item.name} -> {target_dir}/")

    print("\n=== Done ===")
    print(f"Moved: {moved_count}")
    print(f"Skipped: {skipped_count}")


if __name__ == "__main__":
    organize_logs("logs")
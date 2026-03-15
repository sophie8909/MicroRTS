from .evaluate import Evaluator

def test_parse_fitness():
    evaluator = Evaluator(None)  # We won't use the component pool for this test

    import glob
    import os
    log_files = glob.glob(str(evaluator.repo_root / "logs" / "run_*.log"))
    if not log_files:
        return 0.0
    latest_log_file = sorted(log_files)[-1]
    print(f"Testing parse_fitness with log file: {latest_log_file}")
    with open(latest_log_file, "r", encoding="utf-8") as f:
        log_content = f.read()
    # parse the log content to get the fitness score
    fitness = evaluator.parse_fitness(log_content)
    print(f"Parsed fitness: {fitness}")

if __name__ == "__main__":
    test_parse_fitness()
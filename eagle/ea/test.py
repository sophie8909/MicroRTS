import ast
from .config import EAConfig
from .component_pool import ComponentPool
from .evaluate import Evaluator
from .individual import Individual

from .main import OPPONENT_LIST

def _split_top_level_fields(individual_str: str) -> list[str]:
    fields = []
    start = 0
    depth = 0
    in_string = False
    string_quote = ""
    escaped = False

    for i, char in enumerate(individual_str):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
            continue

        if char in ("'", '"'):
            in_string = True
            string_quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            fields.append(individual_str[start:i].strip())
            start = i + 1

    tail = individual_str[start:].strip()
    if tail:
        fields.append(tail)
    return fields


def _parse_literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_ea_log(log_file: str):
    with open(log_file, "r") as f:
        lines = f.readlines()
    individuals = []
    # Extract the Individuals
    for line in lines:
        if line.startswith("Individual"):
            print(line.strip())
            # Extract the part after "Individual(" and before the closing ")"
            start_idx = line.find("Individual(") + len("Individual(")
            end_idx = line.rfind(")")
            if start_idx != -1 and end_idx != -1:
                individual_str = line[start_idx:end_idx]
                # Split key-value pairs only on top-level commas so nested
                # dictionaries such as strategy={...} stay intact.
                components = _split_top_level_fields(individual_str)
                individual_data = {}
                for component in components:
                    if "=" in component:
                        key, value = component.split("=", 1)
                        individual_data[key.strip()] = _parse_literal(value.strip())
        
                individuals.append(Individual(**individual_data))
    return individuals


if __name__ == "__main__":
    # test final prompt
    

    log_id = "20260330_142045"  # Replace with the actual log ID
    experiment_log_folder = f"logs/{log_id}"  # Replace with the actual log folder name
    evaluator = Evaluator(ComponentPool.from_json(f"{experiment_log_folder}/component_pool.json"))
    
    last_gen = 30
    last_generation_log = f"{experiment_log_folder}/generation_{last_gen}_mo.txt"  # Replace with the actual generation log file name
    individuals = parse_ea_log(last_generation_log)


    for individual in individuals:
        print(f"Testing individual: {individual}")
        prompt = evaluator.construct_prompt(individual)
        print(f"Constructed Prompt:\n{prompt}\n")
        # for opponent in OPPONENT_LIST:
        #     print(f"Testing against opponent: {opponent}")
        #     evaluator.set_opponent(opponent)
            
        #     evaluator.run_simulation()

        #     #get run log file
        #     latest_log_file = evaluator.get_latest_log_file()
        #     if latest_log_file:
        #         print(f"Testing parse_fitness with log file: {latest_log_file}")

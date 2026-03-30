

if __name__ == "__main__":
    # test final prompt
    from .config import EAConfig
    from .component_pool import ComponentPool
    from .evaluate import Evaluator

    from .main import OPPONENT_LIST

    evaluator = Evaluator(ComponentPool.from_json("prompts/components.json"))
    for opponent in OPPONENT_LIST:
        print(f"Testing against opponent: {opponent}")
        evaluator.set_opponent(opponent)
        
        evaluator.run_simulation()

        #get run log file
        latest_log_file = evaluator.get_latest_log_file()
        if latest_log_file:
            print(f"Testing parse_fitness with log file: {latest_log_file}")

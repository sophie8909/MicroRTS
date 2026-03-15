"""
Main entry point for running the evolutionary algorithm to optimize prompt components for MicroRTS. 
This script initializes the experiment configuration, loads the prompt components, and executes the selected evolutionary algorithm to evolve effective prompts for guiding agent behavior in MicroRTS.
"""

if __name__ == "__main__":
    # load configuration
    from .config import EAConfig
    config = EAConfig()
    # load prompt components    
    from .component_pool import ComponentPool
    component_pool = ComponentPool.from_json("prompts/components.json")
    # run evolutionary algorithm    
    if config.algorithm == "ga":
        from .ga import GA
        ga = GA(config, component_pool)
        ga.run()
    # elif config.algorithm == "nsga2":
    #     from .nsga2 import NSGA2
    #     nsga2 = NSGA2(config, component_pool)
    #     nsga2.run()
    # elif config.algorithm == "moead":
    #     from .moead import MOEAD
    #     moead = MOEAD(config, component_pool)
    #     moead.run()
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    
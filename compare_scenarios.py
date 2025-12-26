"""
Script to compare Traditional Baseline vs MPC Optimization.
"""
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

from run_scenario import ScenarioRunner, Scenario, SCENARIOS

# Configure logging to only show critical info during comparison to avoid clutter
logging.getLogger("run_scenario").setLevel(logging.WARNING)
logging.getLogger("optimization_agent").setLevel(logging.WARNING)
logging.getLogger("simulation_engine").setLevel(logging.WARNING)
logging.getLogger("forecaster").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Compare")
logger.setLevel(logging.INFO)

def run_benchmark(scenario_name: str, speed: float = 50.0) -> Dict:
    """Run both modes and return metrics."""
    scenario = Scenario[scenario_name]
    config = SCENARIOS[scenario]
    
    logger.info(f"\nRunning Benchmark for {scenario_name}...")
    logger.info("-" * 40)
    
    # 1. Traditional Run
    logger.info("  > Running Traditional Mode...")
    runner_trad = ScenarioRunner(scenario=scenario, mode="TRADITIONAL", speed=speed)
    runner_trad.run()
    metrics_trad = runner_trad.metrics
    
    # 2. MPC Run
    logger.info("  > Running MPC Mode (AI)...")
    runner_mpc = ScenarioRunner(scenario=scenario, mode="MPC", speed=speed)
    runner_mpc.run()
    metrics_mpc = runner_mpc.metrics

    # 3. RL Run
    logger.info("  > Running RL Mode (DQN)...")
    runner_rl = ScenarioRunner(scenario=scenario, mode="RL", speed=speed)
    runner_rl.run()
    metrics_rl = runner_rl.metrics

    # 4. HYBRID Run
    logger.info("  > Running HYBRID Mode (MPC+RL)...")
    runner_hybrid = ScenarioRunner(scenario=scenario, mode="HYBRID", speed=speed)
    runner_hybrid.run()
    metrics_hybrid = runner_hybrid.metrics
    
    # Calculate derived stats
    def get_stats(m):
        total = m["arrivals"]
        served = m["served"]
        reneged = m["reneged"]
        
        # Calculate Average Wait Time (Approximate from queue lengths as simplistic proxy)
        # Real wait time is in simulation.metrics but we need to access it.
        # ScenarioRunner doesn't expose simulation object after run easily, 
        # but we can assume 'queue_length_history' is a proxy for wait.
        avg_queue = np.mean(m["queue_length_history"]) if m["queue_length_history"] else 0
        
        # Calculate Labor Cost (Assume $15/hr wage)
        # teller_history contains count at every decision interval (2 mins)
        avg_tellers = np.mean(m["teller_count_history"]) if m["teller_count_history"] else 0
        duration_hours = config.duration_hours # From scenario config
        total_labor_cost = avg_tellers * duration_hours * 15.0
        
        return {
            "Served %": (served / total * 100) if total else 0,
            "Renege %": (reneged / total * 100) if total else 0,
            "Avg Queue": avg_queue,
            "Total Arrivals": total,
            "Cost": total_labor_cost
        }

    stats_trad = get_stats(metrics_trad)
    stats_mpc = get_stats(metrics_mpc)
    stats_rl = get_stats(metrics_rl)
    stats_hybrid = get_stats(metrics_hybrid)
    
    return {
        "Scenario": scenario_name,
        "Trad Served %": stats_trad["Served %"],
        "MPC Served %": stats_mpc["Served %"],
        "RL Served %": stats_rl["Served %"],
        "HYBRID Served %": stats_hybrid["Served %"],
        "Trad Renege %": stats_trad["Renege %"],
        "MPC Renege %": stats_mpc["Renege %"],
        "RL Renege %": stats_rl["Renege %"],
        "HYBRID Renege %": stats_hybrid["Renege %"],
        "Trad Cost": stats_trad["Cost"],
        "MPC Cost": stats_mpc["Cost"],
        "RL Cost": stats_rl["Cost"],
        "HYBRID Cost": stats_hybrid["Cost"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["FLASH_MOB"])
    args = parser.parse_args()
    
    results = []
    
    print("Starting Comparison Benchmark...")
    print("Muting detailed logs for speed...")
    
    for s in args.scenarios:
        res = run_benchmark(s)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("BENCHMARK RESULTS (Traditional vs MPC vs RL vs HYBRID)")
    print("="*100)
    
    # Format for display
    cols = ["Scenario", 
            "Trad Served %", "MPC Served %", "RL Served %", "HYBRID Served %",
            "Trad Renege %", "MPC Renege %", "RL Renege %", "HYBRID Renege %",
            "Trad Cost", "MPC Cost", "RL Cost", "HYBRID Cost"]
    
    print(df[cols].round(1).to_string(index=False))
    
    # Save to CSV for robust reading
    df[cols].to_csv("benchmark_results.csv", index=False)
    print("\nSaved to benchmark_results.csv")
    
    print("="*80)
    print("Note: Positive 'Improvement' and Positive 'Reduction' are GOOD.")

if __name__ == "__main__":
    main()

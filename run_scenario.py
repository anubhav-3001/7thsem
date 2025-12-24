"""
Scenario Runner for Affective Digital Twin
==========================================

This module defines various test scenarios with different arrival patterns
to test the optimizer's ability to dynamically scale teller count.

Scenarios:
1. FLASH_MOB - Sudden extreme rush
2. LUNCH_RUSH - Predictable lunch peak
3. PAYDAY - End of month surge
4. HOLIDAY_EVE - Gradually building crowd
5. QUIET_DAY - Low traffic baseline

Usage:
    python run_scenario.py --scenario FLASH_MOB --speed 0.5
"""

import argparse
import time
import threading
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Callable
from enum import Enum

import numpy as np

from producer import CustomerEventGenerator
from simulation_engine import AffectiveSimulationEngine
from forecaster import BayesianForecaster
from optimization_agent import OptimizationAgent, SystemState, Action

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Scenario(Enum):
    """Available test scenarios."""
    FLASH_MOB = "flash_mob"
    LUNCH_RUSH = "lunch_rush"
    PAYDAY = "payday"
    HOLIDAY_EVE = "holiday_eve"
    QUIET_DAY = "quiet_day"
    STRESS_TEST = "stress_test"


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    description: str
    duration_hours: float  # Simulation duration
    initial_tellers: int
    arrival_schedule: List[Dict]  # [(start_min, end_min, rate), ...]
    

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS: Dict[Scenario, ScenarioConfig] = {
    
    Scenario.FLASH_MOB: ScenarioConfig(
        name="Flash Mob Scenario",
        description="""
        Normal morning → SUDDEN MASSIVE RUSH (200/hr) → Normal afternoon
        Tests: Rapid teller scaling, emergency response
        Expected: Tellers jump from 3 to 10+ during rush
        """,
        duration_hours=2.0,
        initial_tellers=3,
        arrival_schedule=[
            {"start": 0, "end": 20, "rate": 20},     # Normal opening
            {"start": 20, "end": 25, "rate": 50},    # Building
            {"start": 25, "end": 40, "rate": 200},   # FLASH MOB - extreme rush!
            {"start": 40, "end": 50, "rate": 100},   # Subsiding
            {"start": 50, "end": 80, "rate": 30},    # Recovery
            {"start": 80, "end": 120, "rate": 15},   # Quiet afternoon
        ]
    ),
    
    Scenario.LUNCH_RUSH: ScenarioConfig(
        name="Lunch Rush Scenario", 
        description="""
        Classic bank pattern: Morning ramp → Lunch peak → Afternoon decline
        Tests: Predictable scaling, smooth transitions
        Expected: Gradual teller increase, then decrease
        """,
        duration_hours=3.0,
        initial_tellers=3,
        arrival_schedule=[
            {"start": 0, "end": 30, "rate": 15},     # Morning slow
            {"start": 30, "end": 60, "rate": 30},    # Morning pickup
            {"start": 60, "end": 90, "rate": 60},    # PRE-LUNCH rush building
            {"start": 90, "end": 120, "rate": 90},   # LUNCH PEAK
            {"start": 120, "end": 150, "rate": 50},  # Post-lunch decline
            {"start": 150, "end": 180, "rate": 20},  # Afternoon quiet
        ]
    ),
    
    Scenario.PAYDAY: ScenarioConfig(
        name="Payday Scenario",
        description="""
        End of month payday: Very high sustained traffic all day
        Tests: Maximum capacity handling, break scheduling under load
        Expected: Maximum tellers for extended period
        """,
        duration_hours=2.5,
        initial_tellers=5,
        arrival_schedule=[
            {"start": 0, "end": 15, "rate": 40},     # Already busy at open
            {"start": 15, "end": 45, "rate": 80},    # Morning surge
            {"start": 45, "end": 90, "rate": 100},   # SUSTAINED HIGH LOAD
            {"start": 90, "end": 120, "rate": 90},   # Still very busy
            {"start": 120, "end": 150, "rate": 60},  # Finally calming
        ]
    ),
    
    Scenario.HOLIDAY_EVE: ScenarioConfig(
        name="Holiday Eve Scenario",
        description="""
        Day before holiday: Slow start → Massive afternoon rush → Close early
        Tests: Late-day surge handling, capacity limits
        Expected: Dramatic increase in late afternoon
        """,
        duration_hours=2.5,
        initial_tellers=3,
        arrival_schedule=[
            {"start": 0, "end": 30, "rate": 10},     # Very quiet morning
            {"start": 30, "end": 60, "rate": 25},    # Slow pickup
            {"start": 60, "end": 90, "rate": 50},    # Building
            {"start": 90, "end": 120, "rate": 120},  # AFTERNOON RUSH
            {"start": 120, "end": 140, "rate": 150}, # PEAK RUSH before close
            {"start": 140, "end": 150, "rate": 40},  # Quick dropoff (closing)
        ]
    ),
    
    Scenario.QUIET_DAY: ScenarioConfig(
        name="Quiet Day Scenario",
        description="""
        Very low traffic day: Tests teller reduction and break optimization
        Tests: Removing unnecessary tellers, efficient break scheduling
        Expected: Minimal tellers, many breaks given
        """,
        duration_hours=1.5,
        initial_tellers=5,
        arrival_schedule=[
            {"start": 0, "end": 30, "rate": 10},     # Very quiet
            {"start": 30, "end": 60, "rate": 15},    # Still quiet
            {"start": 60, "end": 90, "rate": 12},    # Maintained low
        ]
    ),
    
    Scenario.STRESS_TEST: ScenarioConfig(
        name="Stress Test Scenario",
        description="""
        Extreme oscillations: Tests rapid scaling up and down
        Tests: System stability under wild fluctuations
        Expected: Oscillating teller count
        """,
        duration_hours=2.0,
        initial_tellers=3,
        arrival_schedule=[
            {"start": 0, "end": 10, "rate": 20},
            {"start": 10, "end": 20, "rate": 150},   # Spike 1
            {"start": 20, "end": 30, "rate": 10},    # Drop
            {"start": 30, "end": 40, "rate": 180},   # Spike 2 (higher)
            {"start": 40, "end": 55, "rate": 15},    # Drop
            {"start": 55, "end": 70, "rate": 200},   # Spike 3 (maximum)
            {"start": 70, "end": 90, "rate": 25},    # Recovery
            {"start": 90, "end": 120, "rate": 100},  # Sustained moderate
        ]
    ),
}


class ScenarioRunner:
    """
    Runs test scenarios with custom arrival patterns.
    """
    
    def __init__(
        self,
        scenario: Scenario,
        speed: float = 1.0,
        seed: int = 42
    ):
        self.config = SCENARIOS[scenario]
        self.speed = speed
        self.seed = seed
        self.shutdown_event = threading.Event()
        
        # Metrics tracking
        self.metrics = {
            "teller_count_history": [],
            "queue_length_history": [],
            "decisions": [],
            "served": 0,
            "reneged": 0,
            "arrivals": 0,
        }
        
        # Current simulation time (in minutes)
        self.sim_time = 0.0
        
    def get_current_rate(self) -> float:
        """Get arrival rate based on current simulation time."""
        for period in self.config.arrival_schedule:
            if period["start"] <= self.sim_time < period["end"]:
                return period["rate"]
        return 5.0  # Default low rate
    
    def run(self):
        """Run the scenario."""
        logger.info("=" * 60)
        logger.info(f"SCENARIO: {self.config.name}")
        logger.info("=" * 60)
        logger.info(self.config.description)
        logger.info(f"Duration: {self.config.duration_hours} hours")
        logger.info(f"Initial tellers: {self.config.initial_tellers}")
        logger.info(f"Speed: {self.speed}x")
        logger.info("=" * 60)
        
        # Initialize modules
        np.random.seed(self.seed)
        
        forecaster = BayesianForecaster(sequence_length=10)
        try:
            forecaster.load_model('forecaster_weights.pth')
            logger.info("✓ Loaded pre-trained forecaster weights")
        except:
            logger.info("⚠ No pre-trained weights, using untrained model")
        
        simulation = AffectiveSimulationEngine(
            num_tellers=self.config.initial_tellers,
            seed=self.seed
        )
        
        # Start service processes for each teller
        simulation.running = True
        simulation.env.process(simulation._service_process())
        simulation.env.process(simulation._anger_update_process())
        
        optimizer = OptimizationAgent()
        
        # Run simulation
        logger.info("\n📊 Starting simulation...")
        logger.info("-" * 60)
        
        decision_interval = 2.0  # minutes
        real_interval = decision_interval / self.speed
        duration_minutes = self.config.duration_hours * 60
        
        while self.sim_time < duration_minutes and not self.shutdown_event.is_set():
            # Get current arrival rate from schedule
            current_rate = self.get_current_rate()
            
            # Generate arrivals for this interval
            expected_arrivals = current_rate * (decision_interval / 60)
            actual_arrivals = np.random.poisson(expected_arrivals)
            
            # Add customers to simulation
            for _ in range(actual_arrivals):
                # Create Customer object directly (generator returns dict, but we need dataclass)
                from simulation_engine import Customer
                import uuid
                customer = Customer(
                    customer_id=str(uuid.uuid4())[:8],
                    arrival_time=simulation.env.now,
                    patience_limit=np.random.exponential(15),  # 15 min avg patience
                    task_complexity=np.clip(np.random.exponential(1.0), 0.5, 3.0),
                    contagion_factor=np.random.beta(2, 5)
                )
                simulation.add_customer(customer)
                self.metrics["arrivals"] += 1
            
            # Step simulation
            simulation.env.run(until=simulation.env.now + decision_interval)
            
            # Update forecaster history
            forecaster.update_history({
                "arrivals": actual_arrivals,
                "hour": int(9 + self.sim_time / 60),  # Start at 9 AM
                "day": 2,  # Wednesday
                "avg_anger": simulation.anger_tracker.current_anger
            })
            
            # Get prediction
            prediction = forecaster.predict_with_uncertainty(num_samples=30)
            if prediction is None:
                prediction = {"mean": current_rate/12, "ucb": current_rate/6, "std": 2.0}
            
            # Get fatigue summary
            fatigue = {t.teller_id: t.fatigue for t in simulation.tellers}
            num_tellers = len(simulation.tellers)
            avg_fatigue = np.mean(list(fatigue.values())) if fatigue else 0
            max_fatigue = max(fatigue.values()) if fatigue else 0
            burnt_out = sum(1 for f in fatigue.values() if f > 0.8)
            
            # Build state
            state = SystemState(
                num_tellers=num_tellers,
                current_queue=len(simulation.waiting_customers),
                avg_fatigue=avg_fatigue,
                max_fatigue=max_fatigue,
                burnt_out_count=burnt_out,
                teller_fatigue=fatigue,
                lobby_anger=simulation.anger_tracker.current_anger,
                predicted_arrivals_mean=prediction["mean"],
                predicted_arrivals_ucb=prediction["ucb"],
                prediction_uncertainty=prediction["std"],
                current_wait=simulation.metrics.total_wait_time / max(1, simulation.metrics.total_served) if simulation.metrics.total_served > 0 else 0
            )
            
            # Get optimizer decision
            action, command = optimizer.decide(state)
            
            # Execute action
            if action == Action.ADD_TELLER:
                simulation.add_teller()
            elif action == Action.REMOVE_TELLER:
                simulation.remove_teller()
            elif action == Action.GIVE_BREAK:
                teller_id = command.get("teller_id")
                if teller_id is not None:
                    simulation.give_teller_break(teller_id, 5.0)
            
            # Track metrics
            self.metrics["teller_count_history"].append(num_tellers)
            self.metrics["queue_length_history"].append(state.current_queue)
            self.metrics["decisions"].append(action.value)
            self.metrics["served"] = simulation.metrics.total_served
            self.metrics["reneged"] = simulation.metrics.total_reneged
            
            # Log progress
            time_str = f"{int(9 + self.sim_time/60):02d}:{int(self.sim_time%60):02d}"
            logger.info(
                f"[{time_str}] Rate:{current_rate:3.0f}/hr | "
                f"Queue:{state.current_queue:3d} | "
                f"Tellers:{num_tellers:2d} | "
                f"Anger:{simulation.anger_tracker.current_anger:.1f} | "
                f"Decision:{action.value}"
            )
            
            # Advance time
            self.sim_time += decision_interval
            time.sleep(real_interval)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print scenario summary statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("SCENARIO SUMMARY")
        logger.info("=" * 60)
        
        total = self.metrics["arrivals"]
        served = self.metrics["served"]
        reneged = self.metrics["reneged"]
        
        logger.info(f"Total Arrivals: {total}")
        logger.info(f"Served: {served}")
        logger.info(f"Reneged: {reneged}")
        if total > 0:
            logger.info(f"Service Rate: {served/total*100:.1f}%")
            logger.info(f"Renege Rate: {reneged/total*100:.1f}%")
        
        # Teller statistics
        tellers = self.metrics["teller_count_history"]
        if tellers:
            logger.info(f"\nTeller Count:")
            logger.info(f"  Min: {min(tellers)}")
            logger.info(f"  Max: {max(tellers)}")
            logger.info(f"  Avg: {np.mean(tellers):.1f}")
        
        # Decision counts
        from collections import Counter
        decisions = Counter(self.metrics["decisions"])
        logger.info(f"\nDecision Distribution:")
        for action, count in decisions.most_common():
            logger.info(f"  {action}: {count}")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run test scenarios")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="FLASH_MOB",
        choices=[s.name for s in Scenario],
        help="Scenario to run"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="Simulation speed multiplier (default: 2.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Scenarios:")
        print("=" * 60)
        for scenario in Scenario:
            config = SCENARIOS[scenario]
            print(f"\n{scenario.name}:")
            print(f"  {config.description.strip()}")
        return
    
    scenario = Scenario[args.scenario]
    runner = ScenarioRunner(
        scenario=scenario,
        speed=args.speed,
        seed=args.seed
    )
    
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("\n⚠ Scenario interrupted")
        runner._print_summary()


if __name__ == "__main__":
    main()

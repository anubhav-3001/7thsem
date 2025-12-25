"""
Module 4: Optimization Agent (optimization_agent.py)
=====================================================

A rational decision-maker balancing cost, wait time, and burnout.
This combines control theory with operations research principles.

Cost Function (Multi-Objective):
Z = C_labor × N + C_wait × W_norm + C_burnout × burnout_count

Where:
- N = number of active tellers
- W_norm = W / W_ref (normalized wait time)
- burnout_count = Σ I(fatigue > 0.8) - count of burnt-out tellers

Cost normalization: "All cost terms are normalized to comparable magnitudes."

Action Space (5 actions):
1. DO_NOTHING - Maintain current state
2. ADD_TELLER - Increase capacity
3. REMOVE_TELLER - Decrease capacity (cost savings)
4. GIVE_BREAK - Send fatigued teller for recovery
5. DELAY_DECISION - Wait one interval before acting

Stability Note:
"The 'DELAY_DECISION' action reduces control oscillations and improves 
system stability under high uncertainty."

Output Contract (Kafka 'bank_commands'):
{
    "action": str,
    "timestamp": ISO8601,
    "reason": str,
    "cost_analysis": {...},
    "teller_id": int (if applicable)
}
"""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Reference values for normalization
W_REF = 5.0  # Reference wait time (minutes)

# Cost weights (tunable hyperparameters)
# Wait cost increased to prioritize service over cost savings
C_LABOR = 2.0       # Reduced from 5 - much cheaper to add tellers in crisis
C_WAIT = 25.0       # Increased from 15 - heavily penalize long waits
C_BURNOUT = 50.0    # Cost per burnt-out teller (highest!)

# Emergency thresholds
QUEUE_EMERGENCY_THRESHOLD = 10  # Force ADD_TELLER above this queue size
QUEUE_HIGH_THRESHOLD = 6        # Consider adding teller

# Decision interval
DECISION_INTERVAL_MINUTES = 2.0  # Reduced from 5 for faster response

# Fluid approximation parameters
MU_SERVICE = 1.0 / 3.0  # Service rate (customers/minute)


# =============================================================================
# ACTION DEFINITIONS
# =============================================================================

class Action(Enum):
    """
    Available optimization actions.
    
    5-action space for stable control:
    - DO_NOTHING: No change
    - ADD_TELLER: +1 capacity
    - REMOVE_TELLER: -1 capacity
    - GIVE_BREAK: Recover fatigued worker
    - DELAY_DECISION: Defer action (stability)
    """
    DO_NOTHING = "DO_NOTHING"
    ADD_TELLER = "ADD_TELLER"
    REMOVE_TELLER = "REMOVE_TELLER"
    GIVE_BREAK = "GIVE_BREAK"
    DELAY_DECISION = "DELAY_DECISION"


@dataclass
class SystemState:
    """Current system state for decision making."""
    num_tellers: int
    current_queue: int
    avg_fatigue: float
    max_fatigue: float
    burnt_out_count: int  # Σ I(fatigue > 0.8)
    teller_fatigue: Dict[int, float]
    lobby_anger: float
    predicted_arrivals_mean: float
    predicted_arrivals_ucb: float
    prediction_uncertainty: float
    current_wait: float  # Current average wait


@dataclass
class CostBreakdown:
    """Breakdown of cost components."""
    labor_cost: float
    wait_cost: float
    burnout_cost: float
    total_cost: float
    
    def to_dict(self) -> Dict:
        return {
            "labor_cost": round(self.labor_cost, 2),
            "wait_cost": round(self.wait_cost, 2),
            "burnout_cost": round(self.burnout_cost, 2),
            "total_cost": round(self.total_cost, 2)
        }


# =============================================================================
# OPTIMIZATION AGENT
# =============================================================================

class OptimizationAgent:
    """
    Multi-objective optimization agent for staffing decisions.
    
    Decision Loop (every 5 minutes):
    1. Fetch UCB arrivals from forecaster (Module 1)
    2. Fetch fatigue state from simulation (Module 3)
    3. Estimate wait time using fluid approximation
    4. Evaluate all 5 actions
    5. Choose minimum-cost action
    6. Publish command to Kafka
    """
    
    def __init__(
        self,
        c_labor: float = C_LABOR,
        c_wait: float = C_WAIT,
        c_burnout: float = C_BURNOUT,
        w_ref: float = W_REF
    ):
        self.c_labor = c_labor
        self.c_wait = c_wait
        self.c_burnout = c_burnout
        self.w_ref = w_ref
        
        # Decision history for auditing
        self.decision_history: List[Dict] = []
        
        # Last action to prevent oscillation
        self.last_action: Optional[Action] = None
        self.delay_remaining: int = 0
        
        # Kafka integration
        self.kafka_producer: Optional[KafkaProducer] = None
        
    def connect_kafka(
        self,
        bootstrap_servers: str = "localhost:9092"
    ) -> bool:
        """Connect to Kafka for command publishing."""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 5, 0)
            )
            logger.info("Optimization agent connected to Kafka")
            return True
        except NoBrokersAvailable:
            logger.warning("Kafka not available for optimizer")
            return False
            
    def _estimate_wait_time(
        self,
        state: SystemState,
        num_tellers: int,
        avg_efficiency: float = 1.0
    ) -> float:
        """
        Estimate wait time using fluid approximation.
        
        For stable system: W ≈ Q / (cμη - λ)
        For overloaded system: W → ∞
        
        Where:
        - Q = current queue length
        - c = number of tellers
        - μ = service rate (customers/minute)
        - η = average efficiency
        - λ = arrival rate (customers/minute)
        """
        if num_tellers <= 0:
            return float('inf')
        
        # CRITICAL: Convert arrivals per interval to per-minute rate
        # This fixes dimensional correctness for queueing formulas
        lambda_eff = state.predicted_arrivals_ucb / DECISION_INTERVAL_MINUTES
        
        # Current load (customers in queue)
        current_load = state.current_queue
        
        # Effective service capacity (customers/minute)
        effective_capacity = num_tellers * MU_SERVICE * avg_efficiency
        
        # Proper fluid approximation from queueing theory
        if effective_capacity > lambda_eff:
            # System is stable: W = Q / (cμη - λ)
            drain_rate = effective_capacity - lambda_eff
            estimated_wait = current_load / drain_rate
        else:
            # System is overloaded - queue grows indefinitely
            # Penalty tied to decision interval for interpretability
            growth_rate = lambda_eff - effective_capacity
            estimated_wait = (
                current_load / max(effective_capacity, 1e-3)
                + growth_rate * DECISION_INTERVAL_MINUTES
            )
            
        return max(0, estimated_wait)
        
    def _calculate_efficiency_after_action(
        self,
        state: SystemState,
        action: Action,
        target_teller: Optional[int] = None
    ) -> float:
        """Calculate average efficiency after taking action."""
        fatigue_dict = dict(state.teller_fatigue)
        
        if action == Action.GIVE_BREAK and target_teller is not None:
            # Break reduces fatigue by exponential decay
            if target_teller in fatigue_dict:
                fatigue_dict[target_teller] *= np.exp(-0.1 * 10)  # 10 min break
                
        if action == Action.ADD_TELLER:
            # New teller has 0 fatigue
            fatigue_dict[max(fatigue_dict.keys(), default=-1) + 1] = 0.0
            
        if action == Action.REMOVE_TELLER:
            # Remove least fatigued
            if fatigue_dict:
                min_teller = min(fatigue_dict, key=fatigue_dict.get)
                del fatigue_dict[min_teller]
                
        if not fatigue_dict:
            return 1.0
        
        # Use MIN efficiency for pessimistic/risk-averse planning
        # One exhausted teller can bottleneck service
        # This is more conservative than averaging which hides outliers
        efficiencies = [max(0.3, 1.0 - 0.6 * f) for f in fatigue_dict.values()]
        return min(efficiencies)
        
    def _count_burnout_after_action(
        self,
        state: SystemState,
        action: Action,
        target_teller: Optional[int] = None
    ) -> int:
        """
        Count burnt-out tellers after action.
        burnout_count = Σ I(fatigue > 0.8)
        """
        fatigue_dict = dict(state.teller_fatigue)
        
        if action == Action.GIVE_BREAK and target_teller is not None:
            if target_teller in fatigue_dict:
                fatigue_dict[target_teller] *= np.exp(-0.1 * 10)
                
        if action == Action.ADD_TELLER:
            fatigue_dict[max(fatigue_dict.keys(), default=-1) + 1] = 0.0
            
        if action == Action.REMOVE_TELLER and fatigue_dict:
            min_teller = min(fatigue_dict, key=fatigue_dict.get)
            del fatigue_dict[min_teller]
        
        # Soft burnout penalty: gradual cost increase from 0.6 to 0.8+
        # This prevents treating fatigue=0.79 same as fatigue=0.2
        # More realistic for human factors modeling
        soft_burnout = sum(
            max(0, (f - 0.6) / 0.4) for f in fatigue_dict.values()
        )
        hard_burnout = sum(1 for f in fatigue_dict.values() if f > 0.8)
        
        # Return combined score (hard count + 0.3 * soft penalty)
        return hard_burnout + 0.3 * soft_burnout
        
    def _calculate_cost(
        self,
        state: SystemState,
        action: Action,
        target_teller: Optional[int] = None
    ) -> CostBreakdown:
        """
        Calculate cost for a given action.
        
        Z = C_labor × N + C_wait × W_norm + C_burnout × burnout_count
        
        Where W_norm = W / W_ref (normalized wait time)
        """
        # Determine resulting number of tellers
        # Note: GIVE_BREAK temporarily reduces effective capacity
        if action == Action.ADD_TELLER:
            num_tellers = state.num_tellers + 1
        elif action == Action.REMOVE_TELLER:
            num_tellers = max(1, state.num_tellers - 1)
        elif action == Action.GIVE_BREAK:
            # Correctly account for capacity loss during break
            # Otherwise GIVE_BREAK looks "free" which is unrealistic
            num_tellers = max(1, state.num_tellers - 1)
        else:
            num_tellers = state.num_tellers
            
        # Calculate efficiency
        efficiency = self._calculate_efficiency_after_action(
            state, action, target_teller
        )
        
        # Estimate wait time
        wait_time = self._estimate_wait_time(state, num_tellers, efficiency)
        
        # Normalize wait time
        wait_norm = wait_time / self.w_ref
        
        # Count burnouts
        burnout_count = self._count_burnout_after_action(
            state, action, target_teller
        )
        
        # Calculate cost components
        labor_cost = self.c_labor * num_tellers
        wait_cost = self.c_wait * wait_norm
        burnout_cost = self.c_burnout * burnout_count
        
        # Add uncertainty penalty for DO_NOTHING and DELAY_DECISION
        # This makes delay meaningful only when uncertainty is high
        if action in [Action.DO_NOTHING, Action.DELAY_DECISION]:
            wait_cost += state.prediction_uncertainty
        
        # Add penalty for GIVE_BREAK when fatigue is low (wasteful break)
        # Only give breaks when fatigue > 0.5 is worthwhile
        if action == Action.GIVE_BREAK and target_teller is not None:
            fatigue = state.teller_fatigue.get(target_teller, 0)
            if fatigue < 0.5:
                # High penalty for unnecessary break (capacity loss)
                labor_cost += 20.0 * (0.5 - fatigue)  # Up to +10 penalty
        
        total = labor_cost + wait_cost + burnout_cost
        
        return CostBreakdown(
            labor_cost=labor_cost,
            wait_cost=wait_cost,
            burnout_cost=burnout_cost,
            total_cost=total
        )
        
    def _get_most_fatigued_teller(self, state: SystemState) -> Optional[int]:
        """Find teller with highest fatigue."""
        if not state.teller_fatigue:
            return None
        return max(state.teller_fatigue, key=state.teller_fatigue.get)
        
    def decide(self, state: SystemState) -> Tuple[Action, Dict]:
        """
        Make staffing decision based on current state.
        
        Algorithm:
        1. If delay is active, continue waiting
        2. Evaluate cost for each possible action
        3. Select minimum cost action
        4. Record decision for audit trail
        
        Returns:
            (action, command_dict)
        """
        timestamp = datetime.now().isoformat()
        
        # Check if we're in delay mode
        if self.delay_remaining > 0:
            # EMERGENCY OVERRIDE: Skip delay if queue is critical
            if state.current_queue >= QUEUE_EMERGENCY_THRESHOLD:
                self.delay_remaining = 0  # Cancel delay - emergency!
            else:
                self.delay_remaining -= 1
                command = {
                    "action": Action.DELAY_DECISION.value,
                    "timestamp": timestamp,
                    "reason": f"Delay active ({self.delay_remaining} intervals remaining)",
                    "cost_analysis": None
                }
                return Action.DELAY_DECISION, command
        
        # EMERGENCY: Immediately add teller if queue is critical
        if state.current_queue >= QUEUE_EMERGENCY_THRESHOLD:
            command = {
                "action": Action.ADD_TELLER.value,
                "timestamp": timestamp,
                "reason": f"EMERGENCY: Queue={state.current_queue} >= {QUEUE_EMERGENCY_THRESHOLD}",
                "cost_analysis": None
            }
            logger.info(f"🚨 EMERGENCY ADD_TELLER: Queue={state.current_queue}")
            return Action.ADD_TELLER, command
            
        # Evaluate all actions
        evaluations: Dict[Action, CostBreakdown] = {}
        target_teller = self._get_most_fatigued_teller(state)
        
        for action in Action:
            if action == Action.REMOVE_TELLER and state.num_tellers <= 3:
                # Keep minimum 3 tellers for adequate service
                continue
            # Don't remove tellers if queue is high
            if action == Action.REMOVE_TELLER and state.current_queue >= QUEUE_HIGH_THRESHOLD:
                continue
            if action == Action.GIVE_BREAK and target_teller is None:
                continue
            # Don't give breaks if queue is high
            if action == Action.GIVE_BREAK and state.current_queue >= QUEUE_HIGH_THRESHOLD:
                continue
            # Don't delay if queue is building
            if action == Action.DELAY_DECISION and state.current_queue >= QUEUE_HIGH_THRESHOLD:
                continue
                
            cost = self._calculate_cost(
                state, action,
                target_teller if action == Action.GIVE_BREAK else None
            )
            evaluations[action] = cost
            
        # Find minimum cost action
        best_action = min(evaluations, key=lambda a: evaluations[a].total_cost)
        best_cost = evaluations[best_action]
        
        # Anti-oscillation: if same action as last time, consider delay
        if (best_action == self.last_action and 
            best_action in [Action.ADD_TELLER, Action.REMOVE_TELLER]):
            # Check if DELAY_DECISION has similar cost
            if Action.DELAY_DECISION in evaluations:
                delay_cost = evaluations[Action.DELAY_DECISION]
                if delay_cost.total_cost <= best_cost.total_cost * 1.1:  # 10% tolerance
                    best_action = Action.DELAY_DECISION
                    best_cost = delay_cost
                    self.delay_remaining = 1
                    
        self.last_action = best_action
        
        # Build command
        command = {
            "action": best_action.value,
            "timestamp": timestamp,
            "reason": self._generate_reason(best_action, best_cost, state),
            "cost_analysis": best_cost.to_dict(),
            "all_costs": {a.value: c.to_dict() for a, c in evaluations.items()}
        }
        
        # Add teller_id if applicable
        if best_action == Action.GIVE_BREAK:
            command["teller_id"] = target_teller
            
        # Record for audit
        self.decision_history.append(command)
        
        # Publish to Kafka
        if self.kafka_producer:
            self.kafka_producer.send('bank_commands', command)
            self.kafka_producer.flush()
            
        return best_action, command
        
    def _generate_reason(
        self,
        action: Action,
        cost: CostBreakdown,
        state: SystemState
    ) -> str:
        """Generate human-readable reason for decision."""
        reasons = {
            Action.DO_NOTHING: (
                f"Current staffing ({state.num_tellers} tellers) is optimal. "
                f"Wait cost: {cost.wait_cost:.1f}, Burnout risk: {state.burnt_out_count}"
            ),
            Action.ADD_TELLER: (
                f"High demand (UCB={state.predicted_arrivals_ucb:.1f}) or wait "
                f"({state.current_wait:.1f}min) exceeds threshold. Adding capacity."
            ),
            Action.REMOVE_TELLER: (
                f"Low demand (UCB={state.predicted_arrivals_ucb:.1f}) and stable queue. "
                f"Reducing labor cost."
            ),
            Action.GIVE_BREAK: (
                f"Teller burnout detected (max fatigue={state.max_fatigue:.2f}). "
                f"Prioritizing worker wellbeing over short-term efficiency."
            ),
            Action.DELAY_DECISION: (
                f"High uncertainty (σ={state.prediction_uncertainty:.2f}) or "
                f"potential oscillation. Deferring decision for stability."
            )
        }
        return reasons.get(action, "Optimal action selected")
        
    def get_decision_trace(self) -> List[Dict]:
        """
        Get decision history for dashboard audit panel.
        
        Returns list of:
        {time, UCB, action, cost}
        """
        return [
            {
                "time": d.get("timestamp", ""),
                "action": d.get("action", ""),
                "cost": d.get("cost_analysis", {}).get("total_cost", 0)
                    if d.get("cost_analysis") else 0,
                "reason": d.get("reason", "")[:50]  # Truncate for display
            }
            for d in self.decision_history[-20:]  # Last 20 decisions
        ]


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

class FixedStaffBaseline:
    """
    Fixed-staff baseline for comparison.
    
    Definition:
    "The fixed-staff baseline uses a constant number of tellers equal to 
    the time-averaged staffing level chosen by the optimization agent, 
    without breaks or adaptive control."
    """
    
    def __init__(self, fixed_tellers: int = 3):
        self.fixed_tellers = fixed_tellers
        self.metrics = {
            "total_wait": 0.0,
            "total_customers": 0,
            "total_reneges": 0
        }
        
    def update(self, wait_time: float, reneged: bool = False) -> None:
        """Update baseline metrics."""
        self.metrics["total_wait"] += wait_time
        self.metrics["total_customers"] += 1
        if reneged:
            self.metrics["total_reneges"] += 1
            
    def get_comparison(self, optimized_metrics: Dict) -> Dict:
        """
        Compare optimized performance vs baseline.
        
        Returns metrics for evaluation section.
        """
        baseline_avg_wait = (
            self.metrics["total_wait"] / 
            max(1, self.metrics["total_customers"])
        )
        optimized_avg_wait = optimized_metrics.get("avg_wait", 0)
        
        wait_reduction = (
            (baseline_avg_wait - optimized_avg_wait) / 
            max(0.01, baseline_avg_wait) * 100
        )
        
        baseline_renege = (
            self.metrics["total_reneges"] / 
            max(1, self.metrics["total_customers"]) * 100
        )
        optimized_renege = optimized_metrics.get("renege_rate", 0)
        
        return {
            "baseline_avg_wait": round(baseline_avg_wait, 2),
            "optimized_avg_wait": round(optimized_avg_wait, 2),
            "wait_reduction_pct": round(wait_reduction, 1),
            "baseline_renege_rate": round(baseline_renege, 1),
            "optimized_renege_rate": round(optimized_renege, 1),
            "baseline_tellers": self.fixed_tellers
        }


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Demo: Optimization agent decision making."""
    logger.info("=" * 60)
    logger.info("Module 4: Optimization Agent Demo")
    logger.info("=" * 60)
    
    agent = OptimizationAgent()
    
    # Simulate different scenarios
    scenarios = [
        SystemState(
            num_tellers=3,
            current_queue=5,
            avg_fatigue=0.3,
            max_fatigue=0.4,
            burnt_out_count=0,
            teller_fatigue={0: 0.3, 1: 0.2, 2: 0.4},
            lobby_anger=2.0,
            predicted_arrivals_mean=10,
            predicted_arrivals_ucb=15,
            prediction_uncertainty=2.5,
            current_wait=3.0
        ),
        SystemState(
            num_tellers=3,
            current_queue=20,
            avg_fatigue=0.7,
            max_fatigue=0.85,
            burnt_out_count=1,
            teller_fatigue={0: 0.7, 1: 0.6, 2: 0.85},
            lobby_anger=7.0,
            predicted_arrivals_mean=40,
            predicted_arrivals_ucb=55,
            prediction_uncertainty=8.0,
            current_wait=12.0
        ),
        SystemState(
            num_tellers=5,
            current_queue=2,
            avg_fatigue=0.2,
            max_fatigue=0.3,
            burnt_out_count=0,
            teller_fatigue={0: 0.1, 1: 0.2, 2: 0.15, 3: 0.3, 4: 0.2},
            lobby_anger=0.5,
            predicted_arrivals_mean=5,
            predicted_arrivals_ucb=8,
            prediction_uncertainty=1.5,
            current_wait=1.0
        )
    ]
    
    scenario_names = [
        "Normal Operations",
        "High Load + Burnout Risk",
        "Low Demand (Overstaffed)"
    ]
    
    for name, state in zip(scenario_names, scenarios):
        logger.info(f"\n--- Scenario: {name} ---")
        logger.info(f"Queue: {state.current_queue}, Tellers: {state.num_tellers}")
        logger.info(f"Avg Fatigue: {state.avg_fatigue:.2f}, Burnt Out: {state.burnt_out_count}")
        logger.info(f"UCB Arrivals: {state.predicted_arrivals_ucb:.1f}")
        
        action, command = agent.decide(state)
        
        logger.info(f"\n✅ Decision: {action.value}")
        logger.info(f"Reason: {command['reason']}")
        if command['cost_analysis']:
            costs = command['cost_analysis']
            logger.info(f"Cost Breakdown: Labor={costs['labor_cost']:.1f}, "
                       f"Wait={costs['wait_cost']:.1f}, Burnout={costs['burnout_cost']:.1f}")
            logger.info(f"Total Cost: {costs['total_cost']:.1f}")
            
    # Show decision trace
    logger.info("\n" + "=" * 60)
    logger.info("Decision Trace (for Dashboard)")
    logger.info("=" * 60)
    for trace in agent.get_decision_trace():
        logger.info(f"  {trace['time'][-8:]}: {trace['action']:15} Cost={trace['cost']:.1f}")


if __name__ == "__main__":
    run_demo()

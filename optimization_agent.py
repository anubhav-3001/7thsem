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

# Cost weights (MPC Objective Function - AGGRESSIVE SERVICE)
C_LABOR = 5.0       # slashed from 15.0 to prioritize service over cost
C_WAIT = 20.0       # quadrupled from 5.0 to severely penalize any wait
C_BURNOUT = 50.0    # Cost per burnt-out teller
C_SWITCH = 1.0      # Reduced damping to allow rapid scaling

# Resource Constraints
MAX_TELLERS = 20    # Increased cap (was 15) to handle extreme lunch peaks

# MPC Parameters
MPC_HORIZON = 5     # Look ahead 5 steps (e.g., 10 minutes)
MPC_DISCOUNT = 0.9  # Future costs are slightly less important

# Emergency thresholds (Safety net, limits MPC search space)
QUEUE_EMERGENCY_THRESHOLD = 25  # Force max capacity calculation
QUEUE_LOW_THRESHOLD = 1         # Min capacity calculation

# Decision interval
DECISION_INTERVAL_MINUTES = 2.0

# START PREVIOUSLY DELETED CONSTANTS NEEDED BY decide() METHOD
QUEUE_PER_TELLER_ADD = 2.0      # More aggressive add (was 3.0)
QUEUE_PER_TELLER_REMOVE = 0.5   # Remove teller if queue/tellers < this
QUEUE_HIGH_THRESHOLD = 15       # Don't remove if queue > 15
# END RESTORED CONSTANTS

# Queue Dynamics Model
# Service rate mu (customers/minute/teller)
MU_SERVICE = 1.0 / 3.0  # 3 minutes per customer -> 0.33 cust/min


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
        
    def _predict_state(self, current_queue: float, num_tellers: int, arrival_rate: float) -> Tuple[float, float]:
        """
        Simulate queue evolution for one time step.
        Returns: (next_queue, wait_time)
        """
        capacity = max(1, num_tellers) * MU_SERVICE * DECISION_INTERVAL_MINUTES
        arrivals = arrival_rate * DECISION_INTERVAL_MINUTES
        
        # Queue dynamics: Q_next = max(0, Q_curr + arrivals - capacity)
        next_queue = max(0.0, current_queue + arrivals - capacity)
        
        # Wait estimation (Little's Law approximation)
        effective_capacity = max(1, num_tellers) * MU_SERVICE
        wait_time = next_queue / max(0.1, effective_capacity)
        
        return next_queue, wait_time

    def _optimize_horizon(self, state: SystemState) -> Tuple[Action, CostBreakdown]:
        """
        Model Predictive Control: Evaluate actions over a future horizon.
        """
        best_initial_action = Action.DO_NOTHING
        min_cumulative_cost = float('inf')
        best_breakdown = None  # Store cost breakdown for logging
        
        # Simplification: Assume arrival rate is constant over horizon (Zero-Order Hold)
        # In a full system, we'd query forecaster for t+1, t+2...
        arrival_rate = state.predicted_arrivals_ucb / DECISION_INTERVAL_MINUTES
        
        # Explore initial actions (Beam Search width 1)
        # We only branch on the first step, then assume "DO_NOTHING" (keep level) for the rest
        # This is a standard simplified control strategy
        candidates = [Action.DO_NOTHING, Action.ADD_TELLER, Action.REMOVE_TELLER]
        
        # Add GIVE_BREAK if applicable
        target_teller = self._get_most_fatigued_teller(state)
        fatigue = state.teller_fatigue.get(target_teller, 0) if target_teller is not None else 0
        if target_teller is not None and fatigue > 0.6:
            candidates.append(Action.GIVE_BREAK)
            
        for action in candidates:
            # 1. Apply Initial Action
            sim_queue = state.current_queue
            sim_tellers = state.num_tellers
            cumulative_cost = 0.0
            
            # Action logic for first step
            if action == Action.ADD_TELLER:
                if sim_tellers >= MAX_TELLERS: continue # Cap resource usage
                sim_tellers += 1
                switch_cost = C_SWITCH
            elif action == Action.REMOVE_TELLER:
                if sim_tellers <= 3: continue # Constraint
                sim_tellers -= 1
                switch_cost = C_SWITCH
            elif action == Action.GIVE_BREAK:
                # Temporarily lose 1 teller capacity
                sim_tellers -= 1 
                switch_cost = 0 # Breaks are necessary, don't penalize as switching
                # Bonus: Reset fatigue cost (simplified)
            else: # DO_NOTHING / DELAY
                switch_cost = 0
                
            cumulative_cost += switch_cost
            
            # 2. Simulate Horizon
            step_breakdown = None
            
            for t in range(MPC_HORIZON):
                # Update State
                sim_queue, sim_wait = self._predict_state(sim_queue, sim_tellers, arrival_rate)
                
                # Calculate Costs (Step t)
                labor_cost = C_LABOR * sim_tellers
                wait_cost = C_WAIT * (sim_wait / W_REF)
                
                # Burnout cost (static approximation)
                burnout_cost = self.c_burnout * state.burnt_out_count if t == 0 else 0
                
                # Breaking logic adjustment for first step
                if action == Action.GIVE_BREAK and t == 0:
                     # Simulate benefit of break (reduced burnout risk)
                     burnout_cost = 0 
                
                step_total = labor_cost + wait_cost + burnout_cost
                cumulative_cost += step_total * (MPC_DISCOUNT ** t)
                
                # Capture first step cost for reporting
                if t == 0:
                    step_breakdown = CostBreakdown(
                        labor_cost=labor_cost,
                        wait_cost=wait_cost,
                        burnout_cost=burnout_cost,
                        total_cost=step_total
                    )
            
            # 3. Select Best
            if cumulative_cost < min_cumulative_cost:
                min_cumulative_cost = cumulative_cost
                best_initial_action = action
                best_breakdown = step_breakdown
                
        return best_initial_action, best_breakdown

    def _decide_traditional(self, state: SystemState) -> Tuple[Action, Dict]:
        """
        Traditional Reactive Logic (Baseline).
        
        Rules:
        - If Queue/Teller > 5: Add Teller
        - If Queue/Teller < 2: Remove Teller
        - If any teller fatigue > 0.7: Give Break (reactive)
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate load metric
        load_per_teller = state.current_queue / max(1, state.num_tellers)
        
        # Check for fatigued tellers needing breaks (reactive)
        if state.teller_fatigue:
            for teller_id, fatigue in state.teller_fatigue.items():
                if fatigue > 0.7 and state.num_tellers > 2:  # Don't give break if too few tellers
                    return Action.GIVE_BREAK, {
                        "action": Action.GIVE_BREAK.value,
                        "reason": f"Reactive: Teller {str(teller_id)[-4:]} fatigue={fatigue:.0%}",
                        "timestamp": timestamp,
                        "teller_id": teller_id
                    }
        
        if load_per_teller > 5.0 and state.num_tellers < MAX_TELLERS:
            return Action.ADD_TELLER, {
                "action": Action.ADD_TELLER.value,
                "reason": f"Reactive: Load {load_per_teller:.1f} > 5.0",
                "timestamp": timestamp
            }
            
        if load_per_teller < 2.0 and state.num_tellers > 3:
             return Action.REMOVE_TELLER, {
                "action": Action.REMOVE_TELLER.value,
                "reason": f"Reactive: Load {load_per_teller:.1f} < 2.0",
                "timestamp": timestamp
            }
            
        return Action.DO_NOTHING, {
            "action": Action.DO_NOTHING.value,
            "reason": "Reactive: Load within normal bounds",
            "timestamp": timestamp
        }

    def decide(self, state: SystemState, mode: str = "MPC", scenario_name: str = "FLASH_MOB") -> Tuple[Action, Dict]:
        """
        Make staffing decision.
        mode: "MPC" (AI), "TRADITIONAL" (Reactive), "RL" (DQN), or "HYBRID" (MPC+RL)
        scenario_name: Name of scenario for loading correct RL model
        """
        if mode == "TRADITIONAL":
            return self._decide_traditional(state)
            
        if mode == "RL":
            return self._decide_rl(state, scenario_name)
            
        if mode == "HYBRID":
            return self._decide_hybrid(state, scenario_name)
            
        timestamp = datetime.now().isoformat()
        
        # SAFETY OVERRIDES (Constraints outside MPC)
        if state.current_queue >= QUEUE_EMERGENCY_THRESHOLD:
             # CRITICAL FIX: Respect MAX_TELLERS even in emergency
             if state.num_tellers < MAX_TELLERS:
                 command = {
                    "action": Action.ADD_TELLER.value,
                    "timestamp": timestamp,
                    "reason": f"EMERGENCY: Queue={state.current_queue}",
                    "cost_analysis": None
                }
                 logger.info(f"🚨 EMERGENCY ADD: Queue={state.current_queue}")
                 return Action.ADD_TELLER, command
             
        if state.current_queue <= QUEUE_LOW_THRESHOLD and state.num_tellers > 5:
             command = {
                "action": Action.REMOVE_TELLER.value,
                "timestamp": timestamp,
                "reason": f"LOW LOAD: Queue={state.current_queue}",
                "cost_analysis": None
            }
             return Action.REMOVE_TELLER, command

        # FATIGUE CHECK - Give breaks to exhausted tellers
        if state.teller_fatigue and state.num_tellers > 2:
            for teller_id, fatigue in state.teller_fatigue.items():
                if fatigue > 0.7:  # 70% fatigue threshold
                    return Action.GIVE_BREAK, {
                        "action": Action.GIVE_BREAK.value,
                        "timestamp": timestamp,
                        "reason": f"MPC: Teller {str(teller_id)[-4:]} fatigue={fatigue:.0%}",
                        "cost_analysis": None,
                        "teller_id": teller_id
                    }

        # MPC OPTIMIZATION
        best_action, cost_analysis = self._optimize_horizon(state)
        
        command = {
            "action": best_action.value,
            "timestamp": timestamp,
            "reason": f"MPC Optimal (H={MPC_HORIZON})",
            "cost_analysis": cost_analysis.to_dict() if cost_analysis else None
        }
        
        logger.info(f"🤖 MPC Decision: {best_action.value} (Queue={state.current_queue}, Pred={state.predicted_arrivals_ucb:.1f})")
        
        return best_action, command

    def _decide_rl(self, state: SystemState, scenario_name: str = "FLASH_MOB") -> Tuple[Action, Dict]:
        """Deep Q-Learning Decision."""
        # Lazy Load Agent (reload if scenario changes)
        current_model = getattr(self, "_rl_model_name", None)
        target_model = f"rl_model_{scenario_name}.pth"
        
        if not hasattr(self, "rl_agent") or current_model != target_model:
            try:
                from rl_agent import DQNAgent
                # State size must match training (4)
                self.rl_agent = DQNAgent(input_size=4, action_size=3)
                # Try loading scenario-specific model
                try:
                    self.rl_agent.load(target_model)
                    self.rl_agent.epsilon = 0.01 # Greedy for inference
                    self._rl_model_name = target_model
                    logger.info(f"✓ Loaded RL Model: {scenario_name}")
                except:
                    # Fallback to flash_mob model
                    try:
                        self.rl_agent.load("rl_model_FLASH_MOB.pth")
                        self._rl_model_name = "rl_model_FLASH_MOB.pth"
                        logger.warning(f"⚠ No RL model for {scenario_name}, using FLASH_MOB")
                    except:
                        logger.warning("⚠ No RL model found, using untrained agent")
            except ImportError:
                 logger.error("Could not import DQNAgent")
                 return Action.DO_NOTHING, {}

        # Construct State Vector [Queue/50, Tellers/20, Forecast/20, Fatigue]
        # Must match train_rl.py normalization
        s_vec = [
            state.current_queue / 50.0,
            state.num_tellers / 20.0,
            state.predicted_arrivals_ucb / 20.0,
            state.avg_fatigue
        ]
        
        action_idx = self.rl_agent.select_action(s_vec)
        
        # Map: 0->DO_NOTHING, 1->ADD, 2->REMOVE
        action_map = {0: Action.DO_NOTHING, 1: Action.ADD_TELLER, 2: Action.REMOVE_TELLER}
        best_action = action_map.get(action_idx, Action.DO_NOTHING)
        
        # Safety Overrides still apply? Maybe partially.
        # Let's trust pure RL for now unless physically impossible
        if best_action == Action.ADD_TELLER and state.num_tellers >= MAX_TELLERS:
            best_action = Action.DO_NOTHING
        if best_action == Action.REMOVE_TELLER and state.num_tellers <= 1:
            best_action = Action.DO_NOTHING

        timestamp = datetime.now().isoformat()
        command = {
            "action": best_action.value,
            "timestamp": timestamp,
            "reason": f"RL Policy ({scenario_name})",
            "cost_analysis": None
        }
        return best_action, command

    def _decide_hybrid(self, state: SystemState, scenario_name: str = "FLASH_MOB") -> Tuple[Action, Dict]:
        """
        HYBRID Decision: Combines MPC and RL intelligently.
        
        Strategy:
        - Use MPC when: queue > 15 OR predicted arrivals > 50 (peak/busy periods)
        - Use RL when: queue <= 15 AND predicted arrivals <= 50 (low traffic)
        
        This combines MPC's forecasting advantage during peaks with 
        RL's cost efficiency during quiet periods.
        """
        # Determine current load level
        is_peak = (state.current_queue > 15 or 
                   state.predicted_arrivals_ucb > 50 or
                   state.lobby_anger > 5.0)
        
        timestamp = datetime.now().isoformat()
        
        if is_peak:
            # Use MPC for peak periods (better service)
            action, cmd = self._decide_mpc_internal(state)
            cmd["reason"] = f"HYBRID→MPC (Queue={state.current_queue}, Pred={state.predicted_arrivals_ucb:.0f})"
            logger.info(f"🔀 HYBRID: Peak detected → Using MPC")
            return action, cmd
        else:
            # Use RL for low-traffic periods (cost efficient)
            action, cmd = self._decide_rl(state, scenario_name)
            cmd["reason"] = f"HYBRID→RL (Queue={state.current_queue}, Pred={state.predicted_arrivals_ucb:.0f})"
            logger.info(f"🔀 HYBRID: Low traffic → Using RL")
            return action, cmd
    
    def _decide_mpc_internal(self, state: SystemState) -> Tuple[Action, Dict]:
        """Internal MPC decision without logging (for hybrid use)."""
        timestamp = datetime.now().isoformat()
        
        # SAFETY OVERRIDES
        if state.current_queue >= QUEUE_EMERGENCY_THRESHOLD:
            if state.num_tellers < MAX_TELLERS:
                return Action.ADD_TELLER, {
                    "action": Action.ADD_TELLER.value,
                    "timestamp": timestamp,
                    "reason": f"EMERGENCY: Queue={state.current_queue}",
                    "cost_analysis": None
                }
                
        if state.current_queue <= QUEUE_LOW_THRESHOLD and state.num_tellers > 5:
            return Action.REMOVE_TELLER, {
                "action": Action.REMOVE_TELLER.value,
                "timestamp": timestamp,
                "reason": f"LOW LOAD: Queue={state.current_queue}",
                "cost_analysis": None
            }

        # FATIGUE CHECK - Give breaks to exhausted tellers
        if state.teller_fatigue and state.num_tellers > 2:
            for teller_id, fatigue in state.teller_fatigue.items():
                if fatigue > 0.7:  # 70% fatigue threshold
                    return Action.GIVE_BREAK, {
                        "action": Action.GIVE_BREAK.value,
                        "timestamp": timestamp,
                        "reason": f"HYBRID-MPC: Teller {str(teller_id)[-4:]} fatigue={fatigue:.0%}",
                        "cost_analysis": None,
                        "teller_id": teller_id
                    }

        # MPC OPTIMIZATION
        best_action, cost_analysis = self._optimize_horizon(state)
        
        return best_action, {
            "action": best_action.value,
            "timestamp": timestamp,
            "reason": f"MPC Optimal (H={MPC_HORIZON})",
            "cost_analysis": cost_analysis.to_dict() if cost_analysis else None
        }
        
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

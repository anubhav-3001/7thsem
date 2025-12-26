"""
Train the DQN Agent using the Simulation as an Environment.
"""
import logging
import numpy as np
import torch
import pandas as pd
from run_scenario import SCENARIOS, Scenario
from simulation_engine import AffectiveSimulationEngine
from optimization_agent import OptimizationAgent, SystemState, Action
from rl_agent import DQNAgent

# Mute simulation logs
logging.getLogger("simulation_engine").setLevel(logging.WARNING)
logging.getLogger("run_scenario").setLevel(logging.WARNING)

from forecaster import BayesianForecaster

def train_rl(scenario_name="FLASH_MOB", episodes=150):
    print(f"Starting RL Training for {scenario_name} ({episodes} episodes)...")
    
    # Init Forecaster
    forecaster = BayesianForecaster()
    try:
        # Try specific weights first
        forecaster.load_model(f"forecaster_weights_{scenario_name.lower()}.pth")
    except:
        try: 
            forecaster.load_model("forecaster_weights.pth")
        except:
            print("Warning: No forecaster weights found.")

    # Hyperparams
    state_size = 4  # [Queue, Tellers, Forecast, Fatigue]
    action_size = 3 # [DO_NOTHING, ADD, REMOVE]
    
    agent = DQNAgent(state_size, action_size)
    
    scenario_config = SCENARIOS[Scenario[scenario_name]]
    
    # IMPROVED Reward Weights - SERVICE FOCUSED (v2)
    C_WAIT = 40.0    # Per-customer wait penalty per step
    C_LABOR = 1.0    # Very low - prioritize service over cost
    C_RENEGE = 75.0  # TRIPLED: Heavy penalty per reneged customer
    C_SERVICE = 15.0 # TRIPLED: Strong bonus per served customer
    
    for e in range(episodes):
        # Reset Sim
        sim = AffectiveSimulationEngine(num_tellers=scenario_config.initial_tellers, seed=42+e)
        
        # Track reneges/served for reward shaping
        prev_served = 0
        prev_reneged = 0
        
        # Manually drive sim
        total_reward = 0
        done = False
        
        # Initial State
        sim.env.run(until=1) 
        
        state_vec = [0, len(sim.tellers), 0, 0] # Init placeholder
        
        # Simulation Loop (Step by Step)
        decision_interval = 2
        sim_duration_minutes = scenario_config.duration_hours * 60
        
        # Helper to get rate
        def get_rate(time_min):
            for period in scenario_config.arrival_schedule:
                if period["start"] <= time_min < period["end"]:
                    return period["rate"]
            return 5.0
        
        while not done:
            now = sim.env.now
            if now >= sim_duration_minutes:
                done = True
                break
            
            # 1. Generate Arrivals for this step (CRITICAL FOR TRAINING)
            current_rate = get_rate(now)
            expected_arrivals = current_rate * (decision_interval / 60)
            actual_arrivals = np.random.poisson(expected_arrivals)
            
            for _ in range(actual_arrivals):
                # Create synthetic customer
                from simulation_engine import Customer
                import uuid
                c = Customer(
                    customer_id=str(uuid.uuid4())[:8],
                    arrival_time=now,
                    patience_limit=np.random.exponential(15),
                    task_complexity=np.random.normal(1.0, 0.2),
                    contagion_factor=0.5
                )
                sim.add_customer(c)
                
            # Observe State
            pred_dict = forecaster.predict_with_uncertainty(now/60.0)
            pred = pred_dict["ucb"] if pred_dict else 50.0
            pred_mean = pred_dict["mean"] if pred_dict else 50.0
            pred_std = pred_dict["std"] if pred_dict else 0.0
            
            # Fatigue Stats
            fatigues = [t.fatigue for t in sim.tellers]
            avg_fatigue = np.mean(fatigues) if fatigues else 0
            max_fatigue = max(fatigues) if fatigues else 0
            burnt_out = sum(1 for f in fatigues if f > 0.8)
            
            sys_state = SystemState(
                num_tellers=len(sim.tellers),
                current_queue=len(sim.waiting_customers),
                avg_fatigue=avg_fatigue,
                max_fatigue=max_fatigue,
                burnt_out_count=burnt_out,
                teller_fatigue={t.teller_id: t.fatigue for t in sim.tellers},
                lobby_anger=sim.anger_tracker.current_anger,
                predicted_arrivals_mean=pred_mean,
                predicted_arrivals_ucb=pred,
                prediction_uncertainty=pred_std,
                current_wait=sim.metrics.total_wait_time / max(1, sim.metrics.total_served) 
            )
            
            # Normalize State for NN
            s_curr = [
                sys_state.current_queue / 50.0, # Norm
                sys_state.num_tellers / 20.0,
                sys_state.predicted_arrivals_ucb / 20.0,
                sys_state.avg_fatigue # Already 0-1
            ]
            
            # Action
            action_idx = agent.select_action(s_curr)
            
            # Map action idx to Action enum
            action_map = {0: Action.DO_NOTHING, 1: Action.ADD_TELLER, 2: Action.REMOVE_TELLER}
            action_enum = action_map[action_idx]
            
            # Execute Action in Sim
            if action_enum == Action.ADD_TELLER:
                sim.add_teller()
            elif action_enum == Action.REMOVE_TELLER:
                sim.remove_teller()
                
            # Step Sim
            step_duration = decision_interval
            sim.env.run(until=now + step_duration)
            
            # Calculate IMPROVED Reward (Service-Focused v2)
            current_q = len(sim.waiting_customers)
            
            # Base costs
            labor_cost = len(sim.tellers) * (C_LABOR/60 * step_duration)
            wait_cost = current_q * (C_WAIT/60 * step_duration)
            
            # SERVICE SHAPING: Reward for serving, penalize reneges
            new_served = sim.metrics.total_served - prev_served
            new_reneged = sim.metrics.total_reneged - prev_reneged
            service_reward = new_served * C_SERVICE
            renege_penalty = new_reneged * C_RENEGE
            
            prev_served = sim.metrics.total_served
            prev_reneged = sim.metrics.total_reneged
            
            reward = -(labor_cost + wait_cost + renege_penalty) + service_reward
            
            # EXPONENTIAL QUEUE PENALTY (new!)
            # Penalty grows faster as queue gets longer
            if current_q > 10:
                queue_penalty = (current_q - 10) ** 1.5  # Exponential growth
                reward -= queue_penalty
            
            # Crisis escalation
            if current_q > 30:
                reward -= 100  # Panic threshold lowered
            if current_q > 50:
                reward -= 200  # Crisis penalty
                
            total_reward += reward
            
            # Next State
            next_pred_dict = forecaster.predict_with_uncertainty((now+step_duration)/60.0)
            next_pred = next_pred_dict["ucb"] if next_pred_dict else 50.0
            next_pred_mean = next_pred_dict["mean"] if next_pred_dict else 50.0
            next_pred_std = next_pred_dict["std"] if next_pred_dict else 0.0
            
            # Fatigue Stats Next
            fatigues_next = [t.fatigue for t in sim.tellers]
            avg_fatigue_n = np.mean(fatigues_next) if fatigues_next else 0
            max_fatigue_n = max(fatigues_next) if fatigues_next else 0
            burnt_out_n = sum(1 for f in fatigues_next if f > 0.8)
            
            next_sys_state = SystemState(
                num_tellers=len(sim.tellers),
                current_queue=len(sim.waiting_customers),
                avg_fatigue=avg_fatigue_n,
                max_fatigue=max_fatigue_n,
                burnt_out_count=burnt_out_n,
                teller_fatigue={t.teller_id: t.fatigue for t in sim.tellers},
                lobby_anger=sim.anger_tracker.current_anger,
                predicted_arrivals_mean=next_pred_mean,
                predicted_arrivals_ucb=next_pred,
                prediction_uncertainty=next_pred_std,
                current_wait=sim.metrics.total_wait_time / max(1, sim.metrics.total_served)
            )
            
            s_next = [
                next_sys_state.current_queue / 50.0,
                next_sys_state.num_tellers / 20.0,
                next_sys_state.predicted_arrivals_ucb / 20.0,
                np.mean(list(next_sys_state.teller_fatigue.values())) / 100.0 if next_sys_state.teller_fatigue else 0
            ]
            
            # Store & Train
            agent.memory.push(s_curr, action_idx, reward, s_next, done)
            loss = agent.update()
            
        # Update Target Net
        if e % 5 == 0:
            agent.update_target_network()
            
        print(f"Episode {e+1}/{episodes} | Score: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Tellers: {len(sim.tellers)}")
        
    # Save
    agent.save(f"rl_model_{scenario_name}.pth")
    print(f"Simultion Complete. Model saved to rl_model_{scenario_name}.pth")

if __name__ == "__main__":
    train_rl()

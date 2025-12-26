"""
Comparison Dashboard: Traditional vs Dynamic Queuing
=====================================================

Side-by-side comparison of fixed-teller traditional queuing vs our dynamic optimizer.

Usage:
    streamlit run comparison_dashboard.py --server.port 8503
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
import json
from datetime import datetime
from typing import Dict, Optional
from enum import Enum
from pathlib import Path

# Import simulation components
from simulation_engine import AffectiveSimulationEngine, Customer
from forecaster import BayesianForecaster
from optimization_agent import OptimizationAgent, SystemState, Action

# State files for both simulations
STATE_FILE_TRAD = Path(__file__).parent / ".comparison_traditional.json"
STATE_FILE_DYN = Path(__file__).parent / ".comparison_dynamic.json"
STOP_FILE = Path(__file__).parent / ".comparison_stop"

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

class Scenario(Enum):
    FLASH_MOB = "Flash Mob"
    LUNCH_RUSH = "Lunch Rush"
    PAYDAY = "Payday"
    STRESS_TEST = "Stress Test"
    FULL_DAY = "Marathon (Full Day)"

SCENARIO_CONFIGS = {
    Scenario.FLASH_MOB: {
        "name": "Flash Mob",
        "description": "Sudden rush: 20/hr → 200/hr → 15/hr",
        "duration": 2.0,
        "fixed_tellers": 8,  # Traditional approach: fixed count
        "initial_tellers": 3,  # Dynamic approach: starts low
        "schedule": [
            {"start": 0, "end": 20, "rate": 20},
            {"start": 20, "end": 25, "rate": 50},
            {"start": 25, "end": 40, "rate": 200},
            {"start": 40, "end": 50, "rate": 100},
            {"start": 50, "end": 80, "rate": 30},
            {"start": 80, "end": 120, "rate": 15},
        ]
    },
    Scenario.LUNCH_RUSH: {
        "name": "Lunch Rush",
        "description": "12-2pm peak: 15/hr → 120/hr → 20/hr",
        "duration": 6.0,
        "fixed_tellers": 10,
        "initial_tellers": 3,
        "schedule": [
            {"start": 0, "end": 60, "rate": 15},
            {"start": 60, "end": 120, "rate": 25},
            {"start": 120, "end": 180, "rate": 40},
            {"start": 180, "end": 210, "rate": 80},
            {"start": 210, "end": 270, "rate": 120},
            {"start": 270, "end": 300, "rate": 70},
            {"start": 300, "end": 360, "rate": 20},
        ]
    },
    Scenario.PAYDAY: {
        "name": "Payday",
        "description": "Sustained high: 40-100/hr all day",
        "duration": 2.5,
        "fixed_tellers": 12,
        "initial_tellers": 5,
        "schedule": [
            {"start": 0, "end": 15, "rate": 40},
            {"start": 15, "end": 45, "rate": 80},
            {"start": 45, "end": 90, "rate": 100},
            {"start": 90, "end": 120, "rate": 90},
            {"start": 120, "end": 150, "rate": 60},
        ]
    },
    Scenario.STRESS_TEST: {
        "name": "Stress Test",
        "description": "Extreme oscillations: 20↔200/hr",
        "duration": 2.0,
        "fixed_tellers": 15,
        "initial_tellers": 3,
        "schedule": [
            {"start": 0, "end": 10, "rate": 20},
            {"start": 10, "end": 20, "rate": 150},
            {"start": 20, "end": 30, "rate": 10},
            {"start": 30, "end": 40, "rate": 180},
            {"start": 40, "end": 55, "rate": 15},
            {"start": 55, "end": 70, "rate": 200},
            {"start": 70, "end": 90, "rate": 25},
            {"start": 90, "end": 120, "rate": 100},
        ]
    },
    Scenario.FULL_DAY: {
        "name": "Marathon (Full Day)",
        "description": "12-hour simulation: Morning → Lunch → Evening pushes",
        "duration": 12.0,       # 12 hours
        "fixed_tellers": 12,    # High baseline cost
        "initial_tellers": 4,   # Start lean
        "schedule": [
            # Morning Opening (8am-10am)
            {"start": 0, "end": 120, "rate": 80},  
            # Morning Lull (10am-11:30am)
            {"start": 120, "end": 210, "rate": 30},
            # Lunch Peak (11:30am-2pm) - HIGH STRESS
            {"start": 210, "end": 360, "rate": 150},
            # Afternoon Lull (2pm-4pm)
            {"start": 360, "end": 480, "rate": 40},
            # Closing Rush (4pm-6pm)
            {"start": 480, "end": 600, "rate": 100},
            # After Hours (6pm-8pm)
            {"start": 600, "end": 720, "rate": 10},
        ]
    },
}


def get_default_state(system_type: str):
    return {
        'system_type': system_type,
        'time_points': [],
        'queue_lengths': [],
        'teller_counts': [],
        'anger_levels': [],
        'served_counts': [],
        'reneged_counts': [],
        'wait_times': [],  # Average wait time per interval
        'current_time': '09:00',
        'current_queue': 0,
        'current_tellers': 0,
        'current_anger': 0.0,
        'current_avg_wait': 0.0,
        'total_served': 0,
        'total_reneged': 0,
        'total_arrivals': 0,
        'total_labor_cost': 0.0,
        'total_wait_time': 0.0,
        'teller_hours': 0.0,
        'is_running': False,
        'is_complete': False,
    }


import filelock  # Add to imports at top

# Thread-safe file operations
_file_locks = {}

def get_file_lock(state_file):
    lock_path = str(state_file) + ".lock"
    if lock_path not in _file_locks:
        try:
            from filelock import FileLock
            _file_locks[lock_path] = FileLock(lock_path, timeout=5)
        except ImportError:
            # Fallback if filelock not installed
            class DummyLock:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            _file_locks[lock_path] = DummyLock()
    return _file_locks[lock_path]


def save_state(data, state_file):
    """Save state with retry logic"""
    for attempt in range(3):
        try:
            with open(state_file, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            time.sleep(0.1)
    return False


def load_state(state_file, system_type):
    """Load state with retry logic"""
    for attempt in range(3):
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    content = f.read()
                    if content.strip():  # Only parse if file has content
                        data = json.loads(content)
                        if data.get('time_points'):  # Ensure it's valid data
                            return data
        except Exception as e:
            time.sleep(0.1)
    return get_default_state(system_type)


def should_stop():
    return STOP_FILE.exists()


def request_stop():
    STOP_FILE.touch()


def clear_stop():
    if STOP_FILE.exists():
        STOP_FILE.unlink()


# =============================================================================
# TRADITIONAL SIMULATION (Fixed Tellers)
# =============================================================================

def run_traditional_simulation(scenario: Scenario, speed: float):
    """Run traditional queue with fixed number of tellers"""
    config = SCENARIO_CONFIGS[scenario]
    # Load the pre-initialized state (created by run_both_simulations)
    data = load_state(STATE_FILE_TRAD, "Traditional")
    data['is_running'] = True
    
    # Use SEPARATE RandomState for thread safety (same seed = same arrivals)
    rng = np.random.RandomState(42)
    
    # Initialize simulation with FIXED teller count
    simulation = AffectiveSimulationEngine(
        num_tellers=config['fixed_tellers'],
        seed=42
    )
    simulation.running = True
    simulation.env.process(simulation._service_process())
    simulation.env.process(simulation._anger_update_process())
    
    sim_time = 0.0
    decision_interval = 2.0
    duration_minutes = config['duration'] * 60
    cost_per_hour = 25.0
    
    def get_rate(t):
        for period in config['schedule']:
            if period['start'] <= t < period['end']:
                return period['rate']
        return 5.0
    
    while sim_time < duration_minutes and not should_stop():
        current_rate = get_rate(sim_time)
        
        # Generate arrivals using LOCAL random state
        expected = current_rate * (decision_interval / 60)
        actual_arrivals = rng.poisson(expected)
        
        for _ in range(actual_arrivals):
            import uuid
            customer = Customer(
                customer_id=str(uuid.uuid4())[:8],
                arrival_time=simulation.env.now,
                patience_limit=rng.exponential(25),
                task_complexity=np.clip(rng.exponential(0.8), 0.3, 2.0),
                contagion_factor=rng.beta(2, 8)
            )
            simulation.add_customer(customer)
        
        # Step simulation - NO OPTIMIZER, fixed tellers
        simulation.env.run(until=simulation.env.now + decision_interval)
        
        # TRADITIONAL: Mandatory breaks when tellers are fatigued (realistic constraint)
        # Tellers in traditional system still get tired and need breaks
        for teller in simulation.tellers:
            # If teller is very fatigued (>0.75) and not on break, force a break
            if teller.fatigue > 0.75 and not teller.on_break:
                simulation.give_teller_break(teller.teller_id, 5.0)  # 5 min break
        
        # Update state
        time_str = f"{int(9 + sim_time/60):02d}:{int(sim_time%60):02d}"
        data['time_points'].append(time_str)
        data['queue_lengths'].append(len(simulation.waiting_customers))
        data['teller_counts'].append(config['fixed_tellers'])  # Always fixed
        data['anger_levels'].append(float(simulation.anger_tracker.current_anger))
        data['served_counts'].append(simulation.metrics.total_served)
        data['reneged_counts'].append(simulation.metrics.total_reneged)
        
        # Calculate average wait time
        avg_wait = simulation.metrics.total_wait_time / max(1, simulation.metrics.total_served) if simulation.metrics.total_served > 0 else 0
        data['wait_times'].append(avg_wait)
        
        data['current_time'] = time_str
        data['current_queue'] = len(simulation.waiting_customers)
        data['current_tellers'] = config['fixed_tellers']
        data['current_anger'] = float(simulation.anger_tracker.current_anger)
        data['current_avg_wait'] = avg_wait
        data['total_served'] = simulation.metrics.total_served
        data['total_reneged'] = simulation.metrics.total_reneged
        data['total_arrivals'] = simulation.metrics.total_arrivals
        data['total_wait_time'] = simulation.metrics.total_wait_time
        
        # Calculate cost
        interval_hours = decision_interval / 60.0
        data['teller_hours'] += config['fixed_tellers'] * interval_hours
        data['total_labor_cost'] = data['teller_hours'] * cost_per_hour
        
        save_state(data, STATE_FILE_TRAD)
        sim_time += decision_interval
        time.sleep(decision_interval / speed)
    
    data['is_running'] = False
    data['is_complete'] = True
    save_state(data, STATE_FILE_TRAD)


# =============================================================================
# DYNAMIC SIMULATION (Our Optimizer)
# =============================================================================

def run_dynamic_simulation(scenario: Scenario, speed: float):
    """Run dynamic queue with our optimizer"""
    config = SCENARIO_CONFIGS[scenario]
    # Load the pre-initialized state (created by run_both_simulations)
    data = load_state(STATE_FILE_DYN, "Dynamic (Ours)")
    data['is_running'] = True
    
    # Use SEPARATE RandomState for thread safety (same seed = same arrivals as Traditional)
    rng = np.random.RandomState(42)
    
    # Initialize simulation
    simulation = AffectiveSimulationEngine(
        num_tellers=config['initial_tellers'],
        seed=42
    )
    simulation.running = True
    simulation.env.process(simulation._service_process())
    simulation.env.process(simulation._anger_update_process())
    
    # Load forecaster and optimizer
    forecaster = BayesianForecaster(sequence_length=10)
    try:
        forecaster.load_model('forecaster_weights.pth')
    except:
        pass
    optimizer = OptimizationAgent()
    
    sim_time = 0.0
    decision_interval = 2.0
    duration_minutes = config['duration'] * 60
    cost_per_hour = 25.0
    
    def get_rate(t):
        for period in config['schedule']:
            if period['start'] <= t < period['end']:
                return period['rate']
        return 5.0
    
    while sim_time < duration_minutes and not should_stop():
        current_rate = get_rate(sim_time)
        
        # Generate arrivals using LOCAL random state (same as Traditional)
        expected = current_rate * (decision_interval / 60)
        actual_arrivals = rng.poisson(expected)
        
        for _ in range(actual_arrivals):
            import uuid
            customer = Customer(
                customer_id=str(uuid.uuid4())[:8],
                arrival_time=simulation.env.now,
                patience_limit=rng.exponential(25),
                task_complexity=np.clip(rng.exponential(0.8), 0.3, 2.0),
                contagion_factor=rng.beta(2, 8)
            )
            simulation.add_customer(customer)
        
        simulation.env.run(until=simulation.env.now + decision_interval)
        
        # Update forecaster
        forecaster.update_history({
            "arrivals": actual_arrivals,
            "hour": int(9 + sim_time / 60),
            "day": 2,
            "avg_anger": simulation.anger_tracker.current_anger
        })
        
        # Get prediction
        prediction = forecaster.predict_with_uncertainty(num_samples=30)
        if prediction is None:
            prediction = {"mean": current_rate/12, "ucb": current_rate/6, "std": 2.0}
        
        # Build optimizer state
        num_tellers = len(simulation.tellers)
        fatigue = {t.teller_id: t.fatigue for t in simulation.tellers}
        avg_fatigue = np.mean(list(fatigue.values())) if fatigue else 0
        max_fatigue = max(fatigue.values()) if fatigue else 0
        burnt_out = sum(1 for f in fatigue.values() if f > 0.8)
        
        opt_state = SystemState(
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
        action, command = optimizer.decide(opt_state)
        
        if action == Action.ADD_TELLER:
            simulation.add_teller()
        elif action == Action.REMOVE_TELLER:
            simulation.remove_teller()
        elif action == Action.GIVE_BREAK:
            teller_id = command.get("teller_id")
            if teller_id is not None:
                simulation.give_teller_break(teller_id, 5.0)
        
        # Update state
        time_str = f"{int(9 + sim_time/60):02d}:{int(sim_time%60):02d}"
        data['time_points'].append(time_str)
        data['queue_lengths'].append(len(simulation.waiting_customers))
        data['teller_counts'].append(len(simulation.tellers))
        data['anger_levels'].append(float(simulation.anger_tracker.current_anger))
        data['served_counts'].append(simulation.metrics.total_served)
        data['reneged_counts'].append(simulation.metrics.total_reneged)
        
        # Calculate average wait time
        avg_wait = simulation.metrics.total_wait_time / max(1, simulation.metrics.total_served) if simulation.metrics.total_served > 0 else 0
        data['wait_times'].append(avg_wait)
        
        data['current_time'] = time_str
        data['current_queue'] = len(simulation.waiting_customers)
        data['current_tellers'] = len(simulation.tellers)
        data['current_anger'] = float(simulation.anger_tracker.current_anger)
        data['current_avg_wait'] = avg_wait
        data['total_served'] = simulation.metrics.total_served
        data['total_reneged'] = simulation.metrics.total_reneged
        data['total_arrivals'] = simulation.metrics.total_arrivals
        data['total_wait_time'] = simulation.metrics.total_wait_time
        
        # Calculate cost
        interval_hours = decision_interval / 60.0
        data['teller_hours'] += len(simulation.tellers) * interval_hours
        data['total_labor_cost'] = data['teller_hours'] * cost_per_hour
        
        save_state(data, STATE_FILE_DYN)
        sim_time += decision_interval
        time.sleep(decision_interval / speed)
    
    data['is_running'] = False
    data['is_complete'] = True
    save_state(data, STATE_FILE_DYN)


def run_both_simulations(scenario: Scenario, speed: float):
    """Run both simulations in parallel threads"""
    clear_stop()
    
    config = SCENARIO_CONFIGS[scenario]
    
    # Initialize state files BEFORE starting threads (prevents race condition)
    trad_state = get_default_state("Traditional")
    trad_state['current_tellers'] = config['fixed_tellers']
    trad_state['is_running'] = True
    save_state(trad_state, STATE_FILE_TRAD)
    
    dyn_state = get_default_state("Dynamic (Ours)")
    dyn_state['current_tellers'] = config['initial_tellers']
    dyn_state['is_running'] = True
    save_state(dyn_state, STATE_FILE_DYN)
    
    # Small delay to ensure files are written
    time.sleep(0.2)
    
    # Start both threads
    t1 = threading.Thread(target=run_traditional_simulation, args=(scenario, speed), daemon=True)
    t2 = threading.Thread(target=run_dynamic_simulation, args=(scenario, speed), daemon=True)
    
    t1.start()
    t2.start()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_chart(trad_data, dyn_data, metric='queue_lengths', title="Queue Length"):
    """Create side-by-side comparison chart"""
    fig = go.Figure()
    
    # Get metric data with fallback to empty list
    trad_metric = trad_data.get(metric, [])
    dyn_metric = dyn_data.get(metric, [])
    
    if trad_data.get('time_points') and trad_metric:
        fig.add_trace(go.Scatter(
            x=trad_data['time_points'][-500:],
            y=trad_metric[-500:],
            name='Traditional (Fixed)',
            line=dict(color='#FF6B6B', width=3),
            mode='lines+markers'
        ))
    
    if dyn_data.get('time_points') and dyn_metric:
        fig.add_trace(go.Scatter(
            x=dyn_data['time_points'][-500:],
            y=dyn_metric[-500:],
            name='Dynamic (Ours)',
            line=dict(color='#4ECDC4', width=3),
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        height=280,
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    st.set_page_config(
        page_title="Comparison: Traditional vs Dynamic",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Comparison: Traditional vs Dynamic Queuing")
    st.markdown("*Side-by-side comparison of fixed-teller vs our optimizer*")
    
    # Load states
    trad_data = load_state(STATE_FILE_TRAD, "Traditional")
    dyn_data = load_state(STATE_FILE_DYN, "Dynamic (Ours)")
    
    is_running = trad_data.get('is_running', False) or dyn_data.get('is_running', False)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        scenario = st.selectbox(
            "Scenario",
            list(Scenario),
            format_func=lambda x: x.value,
            disabled=is_running
        )
        
        config = SCENARIO_CONFIGS[scenario]
        st.caption(config['description'])
        st.info(f"Traditional: **{config['fixed_tellers']} fixed tellers**\nDynamic: Starts with {config['initial_tellers']}, scales as needed")
        
        speed = st.slider("Speed", 10, 50, 30, disabled=is_running)
        
        if st.button("▶️ Start Comparison", disabled=is_running, type="primary", use_container_width=True):
            run_both_simulations(scenario, float(speed))
            time.sleep(0.3)
            st.rerun()
        
        if st.button("⏹️ Stop", disabled=not is_running, use_container_width=True):
            request_stop()
            time.sleep(0.5)
            st.rerun()
        
        if st.button("🔄 Reset", use_container_width=True):
            clear_stop()
            if STATE_FILE_TRAD.exists():
                STATE_FILE_TRAD.unlink()
            if STATE_FILE_DYN.exists():
                STATE_FILE_DYN.unlink()
            st.rerun()
        
        st.divider()
        if is_running:
            st.success("🔄 Running...")
        elif trad_data.get('is_complete') or dyn_data.get('is_complete'):
            st.success("✅ Complete!")
        else:
            st.info("Ready")
    
    # Header metrics - side by side
    st.markdown("### 📊 Real-time Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Traditional (Fixed Tellers)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Queue", trad_data['current_queue'])
        c2.metric("Tellers", trad_data['current_tellers'])
        c3.metric("Served", trad_data['total_served'])
        c4.metric("Wait", f"{trad_data.get('current_avg_wait', 0):.1f}m")
        c5.metric("Cost", f"${trad_data.get('total_labor_cost', 0):.0f}")
    
    with col2:
        st.markdown("#### 🟢 Dynamic (Our Optimizer)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Queue", dyn_data['current_queue'])
        c2.metric("Tellers", dyn_data['current_tellers'])
        c3.metric("Served", dyn_data['total_served'])
        c4.metric("Wait", f"{dyn_data.get('current_avg_wait', 0):.1f}m")
        c5.metric("Cost", f"${dyn_data.get('total_labor_cost', 0):.0f}")
    
    st.markdown("---")
    
    # Comparison Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_comparison_chart(trad_data, dyn_data, 'queue_lengths', '📊 Queue Length Over Time'),
            use_container_width=True, key="queue"
        )
    
    with col2:
        st.plotly_chart(
            create_comparison_chart(trad_data, dyn_data, 'teller_counts', '🧑‍💼 Teller Count Over Time'),
            use_container_width=True, key="tellers"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_comparison_chart(trad_data, dyn_data, 'wait_times', '⏱️ Average Wait Time (min)'),
            use_container_width=True, key="wait"
        )
    
    with col2:
        st.plotly_chart(
            create_comparison_chart(trad_data, dyn_data, 'reneged_counts', '❌ Cumulative Reneges'),
            use_container_width=True, key="reneged"
        )
    
    # Final comparison (when complete)
    if trad_data.get('is_complete') and dyn_data.get('is_complete'):
        st.markdown("### 🏆 Final Results Comparison")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Create comparison table
            trad_service_rate = trad_data['total_served'] / max(1, trad_data['total_arrivals']) * 100
            dyn_service_rate = dyn_data['total_served'] / max(1, dyn_data['total_arrivals']) * 100
            
            trad_cost_per_cust = trad_data['total_labor_cost'] / max(1, trad_data['total_served'])
            dyn_cost_per_cust = dyn_data['total_labor_cost'] / max(1, dyn_data['total_served'])
            
            trad_avg_wait = trad_data.get('total_wait_time', 0) / max(1, trad_data['total_served'])
            dyn_avg_wait = dyn_data.get('total_wait_time', 0) / max(1, dyn_data['total_served'])
            
            comparison_data = {
                'Metric': ['Total Served', 'Total Reneged', 'Service Rate', 'Avg Wait Time', 'Total Cost', 'Cost/Customer', 'Peak Tellers'],
                'Traditional': [
                    trad_data['total_served'],
                    trad_data['total_reneged'],
                    f"{trad_service_rate:.1f}%",
                    f"{trad_avg_wait:.1f} min",
                    f"${trad_data['total_labor_cost']:.0f}",
                    f"${trad_cost_per_cust:.2f}",
                    max(trad_data['teller_counts']) if trad_data['teller_counts'] else 0,
                ],
                'Dynamic (Ours)': [
                    dyn_data['total_served'],
                    dyn_data['total_reneged'],
                    f"{dyn_service_rate:.1f}%",
                    f"{dyn_avg_wait:.1f} min",
                    f"${dyn_data['total_labor_cost']:.0f}",
                    f"${dyn_cost_per_cust:.2f}",
                    max(dyn_data['teller_counts']) if dyn_data['teller_counts'] else 0,
                ],
                'Winner': []
            }
            
            # Determine winners
            winners = []
            # Served - higher is better
            winners.append('🟢 Dynamic' if dyn_data['total_served'] > trad_data['total_served'] else '🔴 Traditional' if trad_data['total_served'] > dyn_data['total_served'] else 'Tie')
            # Reneged - lower is better
            winners.append('🟢 Dynamic' if dyn_data['total_reneged'] < trad_data['total_reneged'] else '🔴 Traditional' if trad_data['total_reneged'] < dyn_data['total_reneged'] else 'Tie')
            # Service rate - higher is better
            winners.append('🟢 Dynamic' if dyn_service_rate > trad_service_rate else '🔴 Traditional' if trad_service_rate > dyn_service_rate else 'Tie')
            # Wait time - lower is better
            winners.append('🟢 Dynamic' if dyn_avg_wait < trad_avg_wait else '🔴 Traditional' if trad_avg_wait < dyn_avg_wait else 'Tie')
            # Cost - lower is better
            winners.append('🟢 Dynamic' if dyn_data['total_labor_cost'] < trad_data['total_labor_cost'] else '🔴 Traditional' if trad_data['total_labor_cost'] < dyn_data['total_labor_cost'] else 'Tie')
            # Cost per customer - lower is better
            winners.append('🟢 Dynamic' if dyn_cost_per_cust < trad_cost_per_cust else '🔴 Traditional' if trad_cost_per_cust < dyn_cost_per_cust else 'Tie')
            # Peak tellers - context dependent
            winners.append('N/A')
            
            comparison_data['Winner'] = winners
            
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Summary
            dynamic_wins = sum(1 for w in winners if '🟢' in w)
            trad_wins = sum(1 for w in winners if '🔴' in w)
            
            if dynamic_wins > trad_wins:
                st.success(f"### 🏆 Dynamic System Wins! ({dynamic_wins} vs {trad_wins})")
            elif trad_wins > dynamic_wins:
                st.warning(f"### Traditional System Wins ({trad_wins} vs {dynamic_wins})")
            else:
                st.info("### It's a Tie!")
    
    # Auto-refresh
    if is_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()

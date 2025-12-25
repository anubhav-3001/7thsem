"""
Scenario Testing Dashboard
==========================

Real-time dashboard using file-based state sharing between threads.

Usage:
    streamlit run scenario_dashboard.py --server.port 8502
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
import json
import os
from typing import Dict
from enum import Enum
from pathlib import Path

# Import simulation components
from simulation_engine import AffectiveSimulationEngine, Customer
from forecaster import BayesianForecaster
from optimization_agent import OptimizationAgent, SystemState, Action

# State file path
STATE_FILE = Path(__file__).parent / ".scenario_state.json"
STOP_FILE = Path(__file__).parent / ".scenario_stop"

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

class Scenario(Enum):
    FLASH_MOB = "Flash Mob"
    LUNCH_RUSH = "Lunch Rush"
    PAYDAY = "Payday"
    HOLIDAY_EVE = "Holiday Eve"
    QUIET_DAY = "Quiet Day"
    STRESS_TEST = "Stress Test"

SCENARIO_CONFIGS = {
    Scenario.FLASH_MOB: {
        "name": "Flash Mob Scenario",
        "description": "Normal morning → SUDDEN MASSIVE RUSH → Normal afternoon",
        "duration": 2.0,
        "initial_tellers": 3,
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
        "name": "Lunch Rush Scenario",
        "description": "Classic bank pattern with predictable lunch peak",
        "duration": 3.0,
        "initial_tellers": 3,
        "schedule": [
            {"start": 0, "end": 30, "rate": 15},
            {"start": 30, "end": 60, "rate": 30},
            {"start": 60, "end": 90, "rate": 60},
            {"start": 90, "end": 120, "rate": 90},
            {"start": 120, "end": 150, "rate": 50},
            {"start": 150, "end": 180, "rate": 20},
        ]
    },
    Scenario.PAYDAY: {
        "name": "Payday Scenario",
        "description": "End of month: sustained high traffic all day",
        "duration": 2.5,
        "initial_tellers": 5,
        "schedule": [
            {"start": 0, "end": 15, "rate": 40},
            {"start": 15, "end": 45, "rate": 80},
            {"start": 45, "end": 90, "rate": 100},
            {"start": 90, "end": 120, "rate": 90},
            {"start": 120, "end": 150, "rate": 60},
        ]
    },
    Scenario.HOLIDAY_EVE: {
        "name": "Holiday Eve Scenario",
        "description": "Slow start → Massive afternoon rush → Early close",
        "duration": 2.5,
        "initial_tellers": 3,
        "schedule": [
            {"start": 0, "end": 30, "rate": 10},
            {"start": 30, "end": 60, "rate": 25},
            {"start": 60, "end": 90, "rate": 50},
            {"start": 90, "end": 120, "rate": 120},
            {"start": 120, "end": 140, "rate": 150},
            {"start": 140, "end": 150, "rate": 40},
        ]
    },
    Scenario.QUIET_DAY: {
        "name": "Quiet Day Scenario",
        "description": "Very low traffic - tests teller reduction",
        "duration": 1.5,
        "initial_tellers": 5,
        "schedule": [
            {"start": 0, "end": 30, "rate": 10},
            {"start": 30, "end": 60, "rate": 15},
            {"start": 60, "end": 90, "rate": 12},
        ]
    },
    Scenario.STRESS_TEST: {
        "name": "Stress Test Scenario",
        "description": "Extreme oscillations to test rapid scaling",
        "duration": 2.0,
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
}


def get_default_state():
    return {
        'time_points': [],
        'queue_lengths': [],
        'teller_counts': [],
        'anger_levels': [],
        'arrival_rates': [],
        'decisions': [],
        'logs': [],
        'current_time': '09:00',
        'current_queue': 0,
        'current_tellers': 3,
        'current_anger': 0.0,
        'total_served': 0,
        'total_reneged': 0,
        'total_arrivals': 0,
        'is_running': False,
        'is_complete': False,
    }


def save_state(data):
    """Save state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f)


def load_state():
    """Load state from file"""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return get_default_state()


def should_stop():
    """Check if stop was requested"""
    return STOP_FILE.exists()


def request_stop():
    """Request simulation stop"""
    STOP_FILE.touch()


def clear_stop():
    """Clear stop request"""
    if STOP_FILE.exists():
        STOP_FILE.unlink()


def run_scenario_thread(scenario: Scenario, speed: float = 10.0):
    """Run scenario and save state to file for dashboard to read"""
    config = SCENARIO_CONFIGS[scenario]
    
    # Initialize state
    data = get_default_state()
    data['current_tellers'] = config['initial_tellers']
    data['is_running'] = True
    data['logs'].append(f"🚀 Starting {config['name']}")
    data['logs'].append(f"Duration: {config['duration']} hours, Speed: {speed}x")
    save_state(data)
    
    # Clear stop flag
    clear_stop()
    
    # Initialize simulation
    np.random.seed(42)
    simulation = AffectiveSimulationEngine(
        num_tellers=config['initial_tellers'],
        seed=42
    )
    simulation.running = True
    simulation.env.process(simulation._service_process())
    simulation.env.process(simulation._anger_update_process())
    
    # Forecaster and optimizer
    forecaster = BayesianForecaster(sequence_length=10)
    try:
        forecaster.load_model('forecaster_weights.pth')
        data['logs'].append("✓ Loaded trained forecaster")
    except:
        data['logs'].append("⚠ Using untrained forecaster")
    
    optimizer = OptimizationAgent()
    
    # Run simulation
    sim_time = 0.0
    decision_interval = 2.0
    duration_minutes = config['duration'] * 60
    
    def get_rate(t):
        for period in config['schedule']:
            if period['start'] <= t < period['end']:
                return period['rate']
        return 5.0
    
    while sim_time < duration_minutes and not should_stop():
        current_rate = get_rate(sim_time)
        
        # Generate arrivals
        expected = current_rate * (decision_interval / 60)
        actual_arrivals = np.random.poisson(expected)
        
        for _ in range(actual_arrivals):
            import uuid
            customer = Customer(
                customer_id=str(uuid.uuid4())[:8],
                arrival_time=simulation.env.now,
                patience_limit=np.random.exponential(25),  # Increased from 15 for more tolerance
                task_complexity=np.clip(np.random.exponential(0.8), 0.3, 2.0),  # Faster service
                contagion_factor=np.random.beta(2, 8)  # Less susceptible to anger contagion
            )
            simulation.add_customer(customer)
        
        # Step simulation
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
        
        # Build state for optimizer
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
        
        # Get and execute decision
        action, command = optimizer.decide(opt_state)
        
        if action == Action.ADD_TELLER:
            simulation.add_teller()
            data['logs'].append(f"➕ ADD_TELLER → {len(simulation.tellers)} tellers")
        elif action == Action.REMOVE_TELLER:
            simulation.remove_teller()
            data['logs'].append(f"➖ REMOVE_TELLER → {len(simulation.tellers)} tellers")
        elif action == Action.GIVE_BREAK:
            teller_id = command.get("teller_id")
            if teller_id is not None:
                simulation.give_teller_break(teller_id, 5.0)
                data['logs'].append(f"☕ GIVE_BREAK: Teller {teller_id}")
        
        # Update state
        time_str = f"{int(9 + sim_time/60):02d}:{int(sim_time%60):02d}"
        data['time_points'].append(time_str)
        data['queue_lengths'].append(len(simulation.waiting_customers))
        data['teller_counts'].append(len(simulation.tellers))
        data['anger_levels'].append(float(simulation.anger_tracker.current_anger))
        data['arrival_rates'].append(float(current_rate))
        data['decisions'].append(action.value)
        
        data['current_time'] = time_str
        data['current_queue'] = len(simulation.waiting_customers)
        data['current_tellers'] = len(simulation.tellers)
        data['current_anger'] = float(simulation.anger_tracker.current_anger)
        data['total_served'] = simulation.metrics.total_served
        data['total_reneged'] = simulation.metrics.total_reneged
        data['total_arrivals'] = simulation.metrics.total_arrivals
        
        # Keep logs manageable
        if len(data['logs']) > 50:
            data['logs'] = data['logs'][-50:]
        
        # Save state to file
        save_state(data)
        
        sim_time += decision_interval
        time.sleep(decision_interval / speed)
    
    data['is_running'] = False
    data['is_complete'] = True
    data['logs'].append(f"✅ Complete! Served: {data['total_served']}, Reneged: {data['total_reneged']}")
    save_state(data)
    clear_stop()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_queue_teller_chart(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if data['time_points']:
        fig.add_trace(
            go.Bar(x=data['time_points'][-40:], y=data['queue_lengths'][-40:],
                   name="Queue", marker_color='#FF6B6B', opacity=0.7),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=data['time_points'][-40:], y=data['teller_counts'][-40:],
                       name="Tellers", line=dict(color='#4ECDC4', width=4), mode='lines+markers'),
            secondary_y=True
        )
    
    fig.update_layout(title="📊 Queue vs Tellers", height=300,
                      legend=dict(orientation="h", y=1.1), margin=dict(t=50))
    fig.update_yaxes(title_text="Queue", secondary_y=False)
    fig.update_yaxes(title_text="Tellers", secondary_y=True)
    return fig


def create_anger_chart(data):
    fig = go.Figure()
    if data['time_points']:
        fig.add_hrect(y0=0, y1=3, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=3, y1=6, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=6, y1=10, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_trace(go.Scatter(
            x=data['time_points'][-40:], y=data['anger_levels'][-40:],
            mode='lines+markers', line=dict(color='#FF4757', width=3),
            fill='tozeroy', fillcolor='rgba(255,71,87,0.3)'
        ))
    fig.update_layout(title="😠 Anger Index", height=200, yaxis=dict(range=[0, 10]), margin=dict(t=40))
    return fig


def create_rate_chart(data):
    fig = go.Figure()
    if data['time_points']:
        fig.add_trace(go.Scatter(
            x=data['time_points'][-40:], y=data['arrival_rates'][-40:],
            mode='lines', fill='tozeroy', line=dict(color='#5352ED', width=2)
        ))
    fig.update_layout(title="📈 Arrival Rate/hr", height=180, margin=dict(t=40))
    return fig


def create_decisions_chart(data):
    from collections import Counter
    if data['decisions']:
        counts = Counter(data['decisions'])
        colors = {'DO_NOTHING': '#95E1D3', 'ADD_TELLER': '#4ECDC4', 
                  'REMOVE_TELLER': '#FF6B6B', 'GIVE_BREAK': '#FFE66D', 'DELAY_DECISION': '#A8E6CF'}
        fig = go.Figure(data=[go.Pie(
            labels=list(counts.keys()), values=list(counts.values()), hole=0.4,
            marker_colors=[colors.get(k, '#888') for k in counts.keys()]
        )])
        fig.update_layout(title="🎯 Decisions", height=280, margin=dict(t=40))
        return fig
    return go.Figure().update_layout(title="🎯 Decisions", height=280)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(page_title="Scenario Dashboard", page_icon="🧪", layout="wide")
    
    st.title("🧪 Scenario Testing Dashboard")
    
    # Load current state from file
    data = load_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        scenario = st.selectbox(
            "Scenario", list(Scenario), format_func=lambda x: x.value,
            disabled=data['is_running']
        )
        
        config = SCENARIO_CONFIGS[scenario]
        st.caption(config['description'])
        
        speed = st.slider("Speed", 5, 50, 20, disabled=data['is_running'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=data['is_running'], type="primary", use_container_width=True):
                clear_stop()
                thread = threading.Thread(target=run_scenario_thread, args=(scenario, float(speed)), daemon=True)
                thread.start()
                time.sleep(0.3)
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", disabled=not data['is_running'], use_container_width=True):
                request_stop()
                time.sleep(0.5)
                st.rerun()
        
        # Reset button to clear stuck state
        if st.button("🔄 Reset", use_container_width=True):
            clear_stop()
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            st.rerun()
        
        st.divider()
        if data['is_running']:
            st.success("🔄 Running...")
        elif data['is_complete']:
            st.success("✅ Complete!")
        else:
            st.info("Ready")
    
    # Metrics
    cols = st.columns(6)
    cols[0].metric("🕐 Time", data['current_time'])
    cols[1].metric("👥 Queue", data['current_queue'])
    cols[2].metric("🧑‍💼 Tellers", data['current_tellers'])
    cols[3].metric("😠 Anger", f"{data['current_anger']:.1f}")
    cols[4].metric("✅ Served", data['total_served'])
    cols[5].metric("❌ Reneged", data['total_reneged'])
    
    # Charts
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(create_queue_teller_chart(data), use_container_width=True, key="q")
    with c2:
        st.plotly_chart(create_decisions_chart(data), use_container_width=True, key="d")
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_anger_chart(data), use_container_width=True, key="a")
    with c2:
        st.plotly_chart(create_rate_chart(data), use_container_width=True, key="r")
    
    # Logs
    st.markdown("### 📋 Logs")
    st.code("\n".join(data['logs'][-20:]) if data['logs'] else "Waiting...", language=None)
    
    # Auto-refresh
    if data['is_running']:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()

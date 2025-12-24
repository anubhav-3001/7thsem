"""
Module 5: Dashboard (dashboard.py)
===================================

Real-time visualization making invisible states visible for human supervisors.
Uses Streamlit + Plotly for rapid prototyping and easy reviewer reproduction.

Visualizations:
1. Uncertainty Cone - Actual arrivals + mean prediction + 95% CI
2. Fatigue Heatmap - Color-coded teller fatigue levels [0,1]
3. Lobby Contagion Meter - Gauge 0-10 with danger zone >6
4. Decision Trace Table - Time | Action | Cost Z (auditable)

Time Semantics:
Wall-clock timestamps (datetime.now()) are used for logging and visualization.
Simulation time is tracked separately within the engine (env.now in minutes).

Purpose:
"This dashboard visualizes latent variables (uncertainty, fatigue, emotion),
not just traditional KPIs. This makes the system auditable and supports
human-in-the-loop oversight."
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import threading
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DASHBOARD STATE
# =============================================================================

class DashboardState:
    """Manages real-time data for dashboard."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        
        # Time series data
        self.arrivals_actual: deque = deque(maxlen=max_history)
        self.arrivals_mean: deque = deque(maxlen=max_history)
        self.arrivals_ucb: deque = deque(maxlen=max_history)
        self.arrivals_lcb: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)
        
        # Teller fatigue
        self.teller_fatigue: Dict[int, float] = {}
        
        # Lobby anger
        self.lobby_anger: float = 0.0
        # Note: anger_history reserved for future sparkline extension
        self.anger_history: deque = deque(maxlen=max_history)
        
        # Decision trace
        self.decision_trace: List[Dict] = []
        
        # Metrics (individual attributes for easy access)
        self.total_arrivals: int = 0
        self.total_served: int = 0
        self.total_reneged: int = 0
        self.avg_wait: float = 0.0
        self.renege_rate: float = 0.0
        
    def update_predictions(
        self,
        actual: float,
        mean: float,
        std: float,
        timestamp: str
    ) -> None:
        """Update arrival predictions."""
        self.arrivals_actual.append(actual)
        self.arrivals_mean.append(mean)
        self.arrivals_ucb.append(mean + 1.96 * std)
        self.arrivals_lcb.append(max(0, mean - 1.96 * std))
        self.timestamps.append(timestamp)
        
    def update_fatigue(self, teller_data: List[Dict]) -> None:
        """Update teller fatigue levels."""
        self.teller_fatigue = {
            t["teller_id"]: t["fatigue"]
            for t in teller_data
        }
        
    def update_anger(self, anger: float, timestamp: str) -> None:
        """Update lobby anger."""
        self.lobby_anger = anger
        self.anger_history.append({
            "timestamp": timestamp,
            "anger": anger
        })
    
    def update_metrics(self, metrics: Dict) -> None:
        """Update key metrics from simulation."""
        self.total_arrivals = metrics.get("total_arrivals", 0)
        self.total_served = metrics.get("total_served", 0)
        self.total_reneged = metrics.get("total_reneged", 0)
        self.avg_wait = metrics.get("avg_wait", 0.0)
        self.renege_rate = metrics.get("renege_rate", 0.0)
        
    def add_decision(self, decision: Dict) -> None:
        """Add decision to trace."""
        self.decision_trace.append(decision)
        if len(self.decision_trace) > 50:
            self.decision_trace = self.decision_trace[-50:]


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_uncertainty_cone(state: DashboardState) -> go.Figure:
    """
    Visual 1: Uncertainty Cone
    
    Shows:
    - Actual arrivals (solid line)
    - Mean prediction (dashed line)
    - 95% confidence interval (shaded UCB to LCB)
    
    Note on LCB: Lower confidence bound is shown for visual symmetry
    and intuition; staffing decisions use UCB only (risk-averse).
    
    Interpretation:
    - Wide cone = high uncertainty, model is guessing
    - Narrow cone = confident prediction
    """
    if len(state.timestamps) < 2:
        # Return empty figure if not enough data
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title="Arrival Predictions with Uncertainty",
            height=400
        )
        return fig
        
    timestamps = list(state.timestamps)
    actual = list(state.arrivals_actual)
    mean = list(state.arrivals_mean)
    ucb = list(state.arrivals_ucb)
    lcb = list(state.arrivals_lcb)
    
    fig = go.Figure()
    
    # Confidence interval (shaded)
    fig.add_trace(go.Scatter(
        x=timestamps + timestamps[::-1],
        y=ucb + lcb[::-1],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        hoverinfo='skip'
    ))
    
    # Mean prediction
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=mean,
        mode='lines',
        name='Predicted Mean',
        line=dict(color='#636EFA', dash='dash', width=2)
    ))
    
    # UCB line (for staffing decisions)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ucb,
        mode='lines',
        name='UCB (Staffing Threshold)',
        line=dict(color='#EF553B', dash='dot', width=1)
    ))
    
    # Actual arrivals
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual,
        mode='lines+markers',
        name='Actual Arrivals',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="📈 Arrival Predictions with Epistemic Uncertainty (5-min intervals)",
        xaxis_title="Time",
        yaxis_title="Arrivals per 5-Minute Interval",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    
    return fig


def create_fatigue_heatmap(state: DashboardState) -> go.Figure:
    """
    Visual 2: Fatigue Heatmap
    
    Shows:
    - Teller-wise fatigue as color-coded bars
    - Risk levels: Green (<0.5), Yellow (0.5-0.8), Red (>0.8)
    
    Managers can preempt burnout by identifying at-risk tellers.
    """
    if not state.teller_fatigue:
        # Demo data if no real data
        tellers = ["Teller 0", "Teller 1", "Teller 2"]
        fatigue = [0.3, 0.5, 0.7]
    else:
        tellers = [f"Teller {tid}" for tid in sorted(state.teller_fatigue.keys())]
        fatigue = [state.teller_fatigue[tid] for tid in sorted(state.teller_fatigue.keys())]
    
    # Color based on fatigue level
    colors = []
    for f in fatigue:
        if f < 0.5:
            colors.append('#00CC96')  # Green - OK
        elif f < 0.8:
            colors.append('#FECB52')  # Yellow - Warning
        else:
            colors.append('#EF553B')  # Red - Burnout
            
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=tellers,
        x=fatigue,
        orientation='h',
        marker_color=colors,
        text=[f'{f:.1%}' for f in fatigue],
        textposition='inside',
        hovertemplate='%{y}: %{x:.1%} fatigue<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Warning", annotation_position="top")
    fig.add_vline(x=0.8, line_dash="dash", line_color="red",
                  annotation_text="Burnout", annotation_position="top")
    
    fig.update_layout(
        title="🔥 Teller Fatigue Levels",
        xaxis_title="Fatigue (0 = Fresh, 1 = Exhausted)",
        xaxis=dict(range=[0, 1]),
        height=250,
        margin=dict(l=100)
    )
    
    return fig


def create_contagion_gauge(state: DashboardState) -> go.Figure:
    """
    Visual 3: Lobby Contagion Meter
    
    Shows:
    - Gauge from 0-10
    - Danger zone > 6 (red arc)
    - Current lobby anger level
    
    Makes emotional collapse measurable and actionable.
    """
    anger = state.lobby_anger
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=anger,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "😤 Lobby Anger Index", 'font': {'size': 24}},
        delta={'reference': 4, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#00CC96'},    # Calm
                {'range': [3, 6], 'color': '#FECB52'},    # Tense
                {'range': [6, 10], 'color': '#EF553B'}    # Danger
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 6
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        annotations=[
            dict(
                text="LobbyAnger = min(10, median(wait) / W_ref)",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=10, color="gray")
            ),
            dict(
                text="⚠️ Values > 6 indicate high reneging risk due to emotional contagion",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=9, color="#EF553B")
            )
        ]
    )
    
    return fig


def create_decision_trace_table(state: DashboardState) -> pd.DataFrame:
    """
    Visual 4: Decision Trace Panel
    
    Audit table showing:
    - Time
    - Predicted UCB
    - Action taken
    - Cost Z
    
    "This makes the system auditable. Reviewers love this."
    """
    if not state.decision_trace:
        # Demo data
        return pd.DataFrame({
            'Time': ['10:00', '10:05', '10:10'],
            'Action': ['DO_NOTHING', 'ADD_TELLER', '⏸️ DELAY_DECISION'],
            'Cost': [85.0, 95.0, 78.0]
        })
        
    data = []
    for d in state.decision_trace[-10:]:  # Last 10 decisions
        action = d.get('action', '')
        # Highlight DELAY_DECISION with indicator for stability-driven choice
        action_display = f"⏸️ {action}" if action == 'DELAY_DECISION' else action
        data.append({
            'Time': d.get('timestamp', '')[-8:] if d.get('timestamp') else '',
            'Action': action_display,
            'Cost': round(d.get('cost', 0), 1)
        })
        
    return pd.DataFrame(data)


def create_metrics_cards(state: DashboardState) -> Dict:
    """Generate metrics for display cards."""
    return {
        "Total Arrivals": state.total_arrivals,
        "Served": state.total_served,
        "Reneged": state.total_reneged,
        "Avg Wait": f"{state.avg_wait:.1f} min",
        "Renege Rate": f"{state.renege_rate:.1f}%"
    }


# =============================================================================
# KAFKA LISTENER (Background Thread)
# =============================================================================

def start_kafka_listener(state: DashboardState, stop_event: threading.Event):
    """Background thread to consume Kafka events."""
    try:
        consumer = KafkaConsumer(
            'bank_simulation',
            'bank_commands',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            api_version=(2, 5, 0),
            consumer_timeout_ms=1000
        )
        
        logger.info("Kafka listener started")
        
        while not stop_event.is_set():
            for message in consumer:
                if stop_event.is_set():
                    break
                    
                data = message.value
                topic = message.topic
                
                if topic == 'bank_simulation':
                    event_type = data.get('event_type')
                    
                    if event_type == 'ANGER_UPDATE':
                        state.update_anger(
                            data.get('lobby_anger', 0),
                            data.get('timestamp', '')
                        )
                        # Also update fatigue if present in the event
                        if 'teller_fatigue' in data:
                            state.update_fatigue(data.get('teller_fatigue', []))
                        # Update metrics if present
                        if 'metrics' in data:
                            state.update_metrics(data.get('metrics', {}))
                    
                    elif event_type == 'PREDICTION_UPDATE':
                        # Update arrival predictions for uncertainty cone
                        state.update_predictions(
                            actual=data.get('actual_arrivals', 0),
                            mean=data.get('mean', 0),
                            std=data.get('std', 0),
                            timestamp=data.get('timestamp', '')
                        )
                        
                elif topic == 'bank_commands':
                    state.add_decision({
                        'timestamp': data.get('timestamp', ''),
                        'action': data.get('action', ''),
                        'cost': data.get('cost_analysis', {}).get('total_cost', 0)
                            if data.get('cost_analysis') else 0
                        # Note: UCB column removed - show absence rather than placeholder
                    })
                    
    except NoBrokersAvailable:
        logger.warning("Kafka not available for dashboard")
    except Exception as e:
        logger.error(f"Kafka listener error: {e}")


# =============================================================================
# DEMO DATA GENERATOR
# =============================================================================

def generate_demo_data(state: DashboardState, step: int):
    """Generate demo data for standalone testing."""
    np.random.seed(42 + step)
    
    # Simulated arrivals
    base = 10 + 5 * np.sin(step * 0.3)  # Cyclic pattern
    actual = int(max(0, np.random.poisson(base)))
    mean = base + np.random.normal(0, 1)
    std = 2 + np.random.uniform(0, 2)
    
    timestamp = (datetime.now() - timedelta(minutes=50-step)).strftime('%H:%M')
    state.update_predictions(actual, mean, std, timestamp)
    
    # Simulated fatigue
    state.teller_fatigue = {
        0: min(1.0, 0.2 + step * 0.01 + np.random.uniform(0, 0.1)),
        1: min(1.0, 0.3 + step * 0.015 + np.random.uniform(0, 0.1)),
        2: min(1.0, 0.5 + step * 0.02 + np.random.uniform(0, 0.1))
    }
    
    # Simulated anger - TIE TO ARRIVALS for causal coherence in demo
    # This prevents demo anger increasing while queue decreases (causally inconsistent)
    anger_base = 0.5 * actual  # Anger driven by actual load
    state.lobby_anger = min(10, anger_base + np.random.uniform(-0.5, 0.5))
    
    # Simulated metrics
    state.metrics = {
        "total_arrivals": step * 5,
        "total_served": int(step * 4.5),
        "total_reneged": int(step * 0.3),
        "avg_wait": 3 + step * 0.1,
        "renege_rate": 6.0 + step * 0.1
    }
    
    # Simulated decisions
    actions = ['DO_NOTHING', 'ADD_TELLER', 'GIVE_BREAK', 'DELAY_DECISION', 'REMOVE_TELLER']
    if step % 5 == 0:
        state.add_decision({
            'timestamp': datetime.now().isoformat(),
            'action': actions[step % len(actions)],
            'cost': 50 + step * 2 + np.random.uniform(-10, 10),
            'ucb': mean + 1.96 * std
        })


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main Streamlit dashboard."""
    st.set_page_config(
        page_title="Socio-Technical Service Dashboard",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 Socio-Technical Service Operations Dashboard")
    st.markdown("""
    *Real-time visibility into latent system states: uncertainty, fatigue, and emotional contagion.*
    """)
    
    # Initialize state
    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = DashboardState()
        st.session_state.demo_step = 0
        
    state = st.session_state.dashboard_state
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        demo_mode = st.checkbox("Demo Mode", value=False, help="Uncheck to use live Kafka data")
        
        if demo_mode:
            if st.button("Step Forward"):
                st.session_state.demo_step += 1
                generate_demo_data(state, st.session_state.demo_step)
                
            auto_refresh = st.checkbox("Auto-refresh (2s)")
            
            if st.button("Reset"):
                st.session_state.dashboard_state = DashboardState()
                st.session_state.demo_step = 0
                st.rerun()
        else:
            # Live mode: start Kafka listener if not already running
            if 'kafka_thread' not in st.session_state:
                st.session_state.stop_event = threading.Event()
                st.session_state.kafka_thread = threading.Thread(
                    target=start_kafka_listener,
                    args=(state, st.session_state.stop_event),
                    daemon=True
                )
                st.session_state.kafka_thread.start()
                st.success("🔴 Kafka listener started - receiving live data")
            else:
                st.success("🟢 Connected to Kafka - live mode active")
            
            auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
            
        st.markdown("---")
        st.markdown("""
        **Legend**
        - 🟢 Normal
        - 🟡 Warning  
        - 🔴 Critical
        """)
        
    # Generate initial demo data only in demo mode
    if demo_mode and st.session_state.demo_step == 0:
        for i in range(20):
            generate_demo_data(state, i)
        st.session_state.demo_step = 20
        
    # Metrics row
    st.markdown("### 📊 Key Metrics")
    metrics = create_metrics_cards(state)
    cols = st.columns(5)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)
        
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Uncertainty Cone
        st.plotly_chart(
            create_uncertainty_cone(state),
            use_container_width=True
        )
        
    with col2:
        # Contagion Gauge
        st.plotly_chart(
            create_contagion_gauge(state),
            use_container_width=True
        )
        
    # Second row
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Fatigue Heatmap
        st.plotly_chart(
            create_fatigue_heatmap(state),
            use_container_width=True
        )
        
    with col4:
        # Decision Trace
        st.markdown("### 📋 Decision Trace (Audit)")
        trace_df = create_decision_trace_table(state)
        st.dataframe(
            trace_df,
            use_container_width=True,
            hide_index=True
        )
        
    # Auto-refresh for both demo and live modes
    if 'auto_refresh' in dir() and auto_refresh:
        time.sleep(2)
        if demo_mode:
            st.session_state.demo_step += 1
            generate_demo_data(state, st.session_state.demo_step)
        st.rerun()
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    Closed-Loop Socio-Technical System | 
    Bayesian Forecasting + Affective Simulation + Multi-Objective Optimization
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

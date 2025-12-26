# 🏦 Bank Queue Simulation Dashboard

A real-time simulation dashboard for testing bank queue optimization strategies using AI-powered decision making.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Install dependencies
pip install streamlit plotly numpy torch kafka-python simpy

# Optional: Install auto-refresh component for smoother updates
pip install streamlit-autorefresh
```

### Running the Dashboard

```bash
cd theoryel
streamlit run scenario_dashboard.py
```

Open http://localhost:8501 in your browser.

## 📊 Usage

1. **Select a Scenario** - Choose from 6 pre-configured scenarios:
   - Flash Mob - Sudden massive rush
   - Lunch Rush - Classic peak pattern
   - Stress Test - Extended high load
   - Payday - Sustained high traffic
   - Holiday Eve - Late surge
   - Quiet Day - Low traffic baseline

2. **Select a Mode**:
   - **MPC** - Model Predictive Control with AI optimization
   - **HYBRID** - MPC during peaks, RL during low traffic

3. **Adjust Speed** - Control simulation speed (5x-50x)

4. **Click Start** - Watch the simulation in real-time!

## 🎯 Features

- **Real-time Metrics**: Queue length, teller count, anger index, cost
- **Teller Fatigue Tracking**: Visual indicators (🟢🟡🟠🔴) for each teller
- **Automatic Break Scheduling**: Tellers get breaks when fatigue > 70%
- **Interactive Charts**: Queue vs Tellers, Anger Index, Arrival Rate
- **Decision Breakdown**: Pie chart showing ADD_TELLER, REMOVE_TELLER, GIVE_BREAK decisions

## 📁 Project Structure

```
theoryel/
├── scenario_dashboard.py    # Main Streamlit dashboard
├── optimization_agent.py    # MPC/HYBRID decision logic
├── simulation_engine.py     # Bank simulation with fatigue model
├── forecaster.py            # Bayesian LSTM forecaster
├── rl_agent.py              # Deep Q-Network agent
├── train_rl.py              # RL training script
└── rl_model_*.pth           # Pre-trained RL models
```

## 🔧 Configuration

Key parameters in `simulation_engine.py`:
- `FATIGUE_ACCUMULATION_RATE = 0.05` - How fast tellers get tired
- `BASE_SERVICE_TIME = 3.0` - Minutes per customer

Key parameters in `optimization_agent.py`:
- `MAX_TELLERS = 15` - Maximum staffing level
- `MPC_HORIZON = 6` - Planning horizon steps

## 📈 Training RL Models

To train RL models for each scenario:

```bash
python train_rl.py --scenario LUNCH_RUSH --episodes 300
python train_rl.py --all  # Train all scenarios
```

## 🛠️ Troubleshooting

**Dashboard stuck on "Running..."**
```bash
# Clear state and restart
del .scenario_state.json
streamlit run scenario_dashboard.py
```

**Kafka connection warnings**
- These are non-blocking; the dashboard works without Kafka
- To enable Kafka: `docker-compose up -d`

## 📝 License

MIT License

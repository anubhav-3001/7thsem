"""
Script to train the Bayesian Forecaster on specific scenarios.
"""
import argparse
import logging
import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple

from forecaster import BayesianForecaster, BayesianLSTM, create_training_data, train_model
from run_scenario import SCENARIOS, Scenario

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(scenario_name: str, days: int = 30) -> Dict[str, List]:
    """
    Generate synthetic training data based on a scenario's arrival schedule.
    """
    if scenario_name not in Scenario.__members__:
        raise ValueError(f"Unknown scenario: {scenario_name}")
        
    scenario_enum = Scenario[scenario_name]
    config = SCENARIOS[scenario_enum]
    
    # logger.info(f"Generating {days} days of data for scenario: {config.name}")
    
    arrivals = []
    hours = []
    days_list = []
    anger_scores = []
    
    for day in range(days):
        day_of_week = day % 7
        duration_minutes = int(config.duration_hours * 60)
        step = 5 # minutes
        
        # Scenario Timeline
        for t in range(0, duration_minutes, step):
            rate = 5.0 # default
            for period in config.arrival_schedule:
                if period["start"] <= t < period["end"]:
                    rate = period["rate"]
                    break
            
            # Convert hourly rate to interval rate
            obs_arrivals = np.random.poisson(rate * (step / 60))
            
            # Time of day: Start at 9:00 AM
            current_hour = 9 + int(t / 60)
            
            # Anger: correlates with load (heuristic for training)
            base_anger = 1.0 + (obs_arrivals / 20.0) 
            anger = np.clip(np.random.normal(base_anger, 0.5), 0, 10)
            
            arrivals.append(obs_arrivals)
            hours.append(current_hour)
            days_list.append(day_of_week)
            anger_scores.append(anger)
            
    return {
        "arrivals": arrivals,
        "hours": hours,
        "days": days_list,
        "anger_scores": anger_scores
    }

def evaluate_model(model: BayesianLSTM, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
    """Evaluate model performance on test set."""
    model.eval()
    with torch.no_grad():
        # Get point predictions (mean of distribution)
        # For evaluation, we can just do one pass with dropout disabled? 
        # No, BayesianLSTM relies on dropout for prediction too usually (MC dropout).
        # But for "goodness of fit" let's check the deterministic output or mean of MC.
        
        # Let's do 20 samples MC dropout 
        preds = []
        for _ in range(20):
            preds.append(model(X_test).numpy())
        
        preds = np.mean(np.array(preds), axis=0) # (test_size, 1)
        
    y_true = y_test.numpy()
    
    mse = np.mean((preds - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - y_true))
    
    return {"RMSE": rmse, "MAE": mae}

def train_and_evaluate(scenario: str, epochs: int, days: int) -> Dict:
    """Train and evaluate a single scenario."""
    
    # 1. Generate Data
    data = generate_synthetic_data(scenario, days)
    
    X, y = create_training_data(
        data["arrivals"], 
        data["hours"], 
        data["days"], 
        data["anger_scores"],
        sequence_length=10
    )
    
    # 2. Split Train/Test (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize Model (uses new default hidden_size=128 from forecaster.py)
    model = BayesianLSTM(input_size=4, hidden_size=128, num_layers=2, dropout_prob=0.3)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Start with higher LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = torch.nn.MSELoss()
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation for scheduler
        model.eval()
        with torch.no_grad():
            val_out = model(X_test)
            val_loss = criterion(val_out, y_test)
        model.train()
        
        scheduler.step(val_loss)
        
        if (epoch+1) % 10 == 0:
            # logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
            pass
            
    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. Save
    filename = f"forecaster_weights_{scenario.lower()}.pth"
    torch.save(model.state_dict(), filename)
    
    return {
        "Scenario": scenario,
        "Final Loss": loss.item(),
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "Saved To": filename
    }

def main():
    parser = argparse.ArgumentParser(description="Train Forecaster for scenarios")
    parser.add_argument("--scenario", type=str, help="Scenario name (e.g., FLASH_MOB)")
    parser.add_argument("--all", action="store_true", help="Train all scenarios")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")  # Increased default
    parser.add_argument("--days", type=int, default=200, help="Days of synthetic data") # Increased default
    
    args = parser.parse_args()
    
    if args.all:
        scenarios_to_run = [s.name for s in Scenario]
    elif args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        print("Please specify --scenario or --all")
        return

    results = []
    logger.info(f"Starting training for {len(scenarios_to_run)} scenarios...")
    
    for s in scenarios_to_run:
        logger.info(f"Training {s}...")
        res = train_and_evaluate(s, args.epochs, args.days)
        results.append(res)
        logger.info(f"  -> RMSE: {res['RMSE']:.2f}")

    # Print Report
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL PERFORMANCE REPORT")
    print("="*60)
    print(df[["Scenario", "RMSE", "MAE", "Saved To"]].to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()

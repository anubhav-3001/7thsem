"""
Train the Bayesian LSTM Forecaster
===================================

This script generates synthetic bank arrival data and trains the LSTM model
to predict future arrivals with uncertainty quantification.

Usage:
    python train_forecaster.py
    
After training, the model weights are saved to 'forecaster_weights.pth'
and can be loaded in main.py for live predictions.
"""

import numpy as np
import torch
import json
from datetime import datetime, timedelta
from typing import List, Tuple

from forecaster import (
    BayesianLSTM, 
    BayesianForecaster,
    create_training_data, 
    train_model
)


def generate_synthetic_bank_data(
    num_days: int = 30,
    intervals_per_day: int = 96  # 15-minute intervals
) -> Tuple[List[int], List[int], List[int], List[float]]:
    """
    Generate realistic synthetic bank arrival data.
    
    Models:
    - Time-of-day patterns (morning ramp, lunch peak, afternoon decline)
    - Day-of-week effects (Mon/Fri busier)
    - Random noise
    
    Returns:
        arrivals, hours, days, anger_scores
    """
    np.random.seed(42)
    
    arrivals = []
    hours = []
    days = []
    anger_scores = []
    
    for day in range(num_days):
        day_of_week = day % 7  # 0=Monday
        
        for interval in range(intervals_per_day):
            # Convert interval to hour (0-23)
            hour = (interval * 15) // 60  # 15-min intervals
            minute_frac = (interval * 15) % 60 / 60
            hour_float = hour + minute_frac
            
            # Base arrival rate based on time of day
            if 9 <= hour_float < 11:
                # Morning ramp-up
                base_rate = 10 + 20 * (hour_float - 9) / 2
            elif 11 <= hour_float < 13:
                # Lunch rush
                base_rate = 60
            elif 13 <= hour_float < 15:
                # Post-lunch decline
                base_rate = 60 - 40 * (hour_float - 13) / 2
            elif 15 <= hour_float < 17:
                # Late afternoon
                base_rate = 15
            else:
                # Off-hours
                base_rate = 5
            
            # Day-of-week effect (Monday/Friday 20% busier)
            if day_of_week in [0, 4]:  # Monday, Friday
                base_rate *= 1.2
            elif day_of_week == 6:  # Sunday (closed or minimal)
                base_rate *= 0.3
            
            # Convert hourly rate to 15-minute interval
            interval_rate = base_rate / 4
            
            # Generate Poisson arrivals
            arrivals_count = np.random.poisson(interval_rate)
            arrivals.append(arrivals_count)
            hours.append(hour)
            days.append(day_of_week)
            
            # Anger correlates with arrivals (more crowded = more angry)
            base_anger = min(arrivals_count / 5, 8)
            anger = np.clip(base_anger + np.random.normal(0, 1), 0, 10)
            anger_scores.append(anger)
    
    return arrivals, hours, days, anger_scores


def main():
    print("=" * 60)
    print("Training Bayesian LSTM Forecaster")
    print("=" * 60)
    
    # Generate synthetic training data
    print("\n📊 Generating synthetic bank arrival data (30 days)...")
    arrivals, hours, days, anger_scores = generate_synthetic_bank_data(num_days=30)
    print(f"   Generated {len(arrivals)} data points")
    print(f"   Avg arrivals per interval: {np.mean(arrivals):.2f}")
    print(f"   Max arrivals: {max(arrivals)}")
    
    # Create training sequences
    print("\n🔧 Creating training sequences...")
    sequence_length = 10  # Match the FORECASTER_SEQUENCE_LEN in main.py
    X, y = create_training_data(arrivals, hours, days, anger_scores, sequence_length)
    print(f"   Training samples: {X.shape[0]}")
    print(f"   Sequence length: {X.shape[1]}")
    print(f"   Features: {X.shape[2]}")
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"   Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Initialize model
    print("\n🧠 Initializing Bayesian LSTM...")
    model = BayesianLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        dropout_prob=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\n🚀 Training model (100 epochs)...")
    print("-" * 40)
    losses = train_model(
        model=model,
        X=X_train,
        y=y_train,
        epochs=100,
        learning_rate=0.001,
        batch_size=32
    )
    print("-" * 40)
    
    # Validate
    print("\n📈 Validation...")
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = torch.nn.MSELoss()(val_predictions, y_val)
        print(f"   Validation MSE: {val_loss.item():.4f}")
        print(f"   Validation RMSE: {np.sqrt(val_loss.item()):.4f}")
    
    # Save model weights
    weights_path = "forecaster_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"\n💾 Model saved to: {weights_path}")
    
    # Test uncertainty quantification
    print("\n🎯 Testing uncertainty quantification...")
    forecaster = BayesianForecaster(model=model, sequence_length=sequence_length)
    
    # Feed last sequence_length observations
    for i in range(-sequence_length, 0):
        forecaster.update_history({
            "arrivals": arrivals[i],
            "hour": hours[i],
            "day": days[i],
            "avg_anger": anger_scores[i]
        })
    
    # Get prediction
    result = forecaster.predict_with_uncertainty(num_samples=100)
    print(f"\n   Sample Prediction:")
    print(f"   Mean: {result['mean']:.2f}")
    print(f"   Std (uncertainty): {result['std']:.2f}")
    print(f"   UCB (95%): {result['ucb']:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete! The model is ready for use.")
    print("=" * 60)
    print("\nTo use in main.py, add this after forecaster initialization:")
    print("    forecaster.load_model('forecaster_weights.pth')")


if __name__ == "__main__":
    main()

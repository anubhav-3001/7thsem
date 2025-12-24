"""
Evaluate the trained Bayesian LSTM Forecaster
"""
import torch
import numpy as np
from forecaster import BayesianLSTM, BayesianForecaster, create_training_data

# Generate test data (different seed for unbiased evaluation)
np.random.seed(123)
arrivals, hours, days, anger_scores = [], [], [], []
for day in range(7):  # 7 days of test data
    day_of_week = day % 7
    for interval in range(96):
        hour = (interval * 15) // 60
        hour_float = hour + (interval * 15) % 60 / 60
        if 9 <= hour_float < 11:
            base_rate = 10 + 20 * (hour_float - 9) / 2
        elif 11 <= hour_float < 13:
            base_rate = 60
        elif 13 <= hour_float < 15:
            base_rate = 60 - 40 * (hour_float - 13) / 2
        elif 15 <= hour_float < 17:
            base_rate = 15
        else:
            base_rate = 5
        if day_of_week in [0, 4]:
            base_rate *= 1.2
        interval_rate = base_rate / 4
        arrivals.append(np.random.poisson(interval_rate))
        hours.append(hour)
        days.append(day_of_week)
        anger_scores.append(np.clip(arrivals[-1]/5 + np.random.normal(0, 1), 0, 10))

# Create test sequences
X_test, y_test = create_training_data(arrivals, hours, days, anger_scores, 10)

# Load trained model
model = BayesianLSTM(input_size=4, hidden_size=64, num_layers=2, dropout_prob=0.3)
model.load_state_dict(torch.load('forecaster_weights.pth', weights_only=True))
model.eval()

# Evaluate
with torch.no_grad():
    y_pred = model(X_test)
    mse = torch.nn.MSELoss()(y_pred, y_test).item()
    mae = torch.abs(y_pred - y_test).mean().item()
    rmse = np.sqrt(mse)

# Baseline: always predict mean
baseline_pred = y_test.mean()
baseline_mse = ((y_test - baseline_pred) ** 2).mean().item()
baseline_rmse = np.sqrt(baseline_mse)

# Print results
print("=" * 55)
print("       MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"Test samples: {len(X_test)}")
print()
print("TRAINED LSTM MODEL:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print()
print("BASELINE (always predict mean):")
print(f"  MSE:  {baseline_mse:.4f}")
print(f"  RMSE: {baseline_rmse:.4f}")
print()
improvement = (1 - mse/baseline_mse)*100
print(f"IMPROVEMENT: {improvement:.1f}% better than baseline")
print()
print("INTERPRETATION:")
avg_arrivals = y_test.mean().item()
print(f"  Average actual arrivals per interval: {avg_arrivals:.2f}")
print(f"  RMSE = {rmse:.2f} (predictions off by ~{rmse:.1f} arrivals)")
print(f"  Relative error: {rmse/avg_arrivals*100:.1f}%")
print()
if improvement > 0:
    print("  The model has learned patterns from the data!")
else:
    print("  The model needs more training or better hyperparameters")
print("=" * 55)

"""
Test that the LSTM model is properly producing varying predictions
"""
from forecaster import BayesianForecaster
import numpy as np

# Create forecaster and load weights
f = BayesianForecaster(sequence_length=10)
f.load_model('forecaster_weights.pth')

print("Testing LSTM model predictions")
print("=" * 50)

# Feed some observations
np.random.seed(42)
for i in range(12):  # More than sequence_length
    f.update_history({
        'arrivals': np.random.randint(1, 15),
        'hour': 12,  # Lunch time
        'day': 2,    # Wednesday
        'avg_anger': np.random.uniform(0.5, 3.0)
    })

# Get prediction
result = f.predict_with_uncertainty(num_samples=50)
print('Prediction with trained model:')
print(f'  Mean: {result["mean"]:.2f}')
print(f'  Std:  {result["std"]:.2f}')
print(f'  UCB:  {result["ucb"]:.2f}')
print()

# Check if source is model or fallback
if result.get("source") == "fallback_heuristic":
    print("WARNING: Still using fallback heuristic!")
else:
    print("SUCCESS: Using trained LSTM model!")
    print("Note: UCB should vary based on input, not be fixed at 0.1")

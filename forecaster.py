"""
Module 1: Bayesian Uncertainty Engine
======================================

This module models EPISTEMIC UNCERTAINTY arising from limited data,
not aleatoric noise. The distinction is critical:
- Epistemic: reducible with more data (what we capture)
- Aleatoric: irreducible randomness in the process

Key Technique: Monte Carlo Dropout
- Dropout remains active during inference
- Multiple forward passes sample from approximate posterior
- Mean = point prediction, Std = epistemic uncertainty

Output Contract:
{
    "mean": float,      # Expected arrivals
    "std": float,       # Epistemic uncertainty (σ)
    "ucb": float,       # Upper Confidence Bound (μ + 1.96σ)
    "timestamp": str    # ISO8601 format
}
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json


class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM using Monte Carlo Dropout for uncertainty quantification.
    
    Architecture:
    - Input: 4D vector (arrivals, hour, day, avg_anger)
    - Layer 1: LSTM capturing short-term temporal patterns
    - Layer 2: LSTM capturing higher-level trends (rush vs lull)
    - Dropout (p=0.5): Core Bayesian approximation trick
    - Linear: Maps hidden state to scalar prediction
    
    Why 2 LSTM layers?
    - First layer: local patterns (minute-level fluctuations)
    - Second layer: global trends (hourly demand curves)
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.5
    ):
        super(BayesianLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        # LSTM inter-layer dropout (regularization only, not Bayesian)
        # Note: This dropout only applies between LSTM layers during training
        # and behaves differently across PyTorch versions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # POST-LSTM MC Dropout - THIS is the Bayesian mechanism
        # Remains active during inference via model.train()
        # This transforms a standard LSTM into an approximate BNN
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, 4)
               Features: [arrivals, hour, day, avg_anger]
               
        Returns:
            Predicted arrivals (batch, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Using last timestep output instead of h_n[-1] for clarity
        # For batch_first=True, lstm_out[:, -1, :] == h_n[-1] in most cases
        # We prefer explicit indexing for interpretability
        last_hidden = lstm_out[:, -1, :]
        
        # Apply MC Dropout (ALWAYS, even during inference)
        dropped = self.dropout(last_hidden)
        
        # Project to scalar
        output = self.fc(dropped)
        
        return output


class BayesianForecaster:
    """
    Wrapper class providing uncertainty-aware predictions.
    
    Key method: predict_with_uncertainty()
    - Runs N stochastic forward passes
    - Aggregates into mean + std + UCB
    - UCB = μ + 1.96σ (95% confidence for staffing decisions)
    """
    
    def __init__(
        self,
        model: Optional[BayesianLSTM] = None,
        sequence_length: int = 15,
        device: str = "cpu"
    ):
        self.model = model if model else BayesianLSTM()
        self.model.to(device)
        self.device = device
        self.sequence_length = sequence_length
        
        # Historical buffer for sequence construction
        # NOTE: This buffer contains OBSERVED arrivals, not forecasts
        # In deployment, ensure you're feeding actual observations to avoid leakage
        self.history_buffer: List[Dict] = []
        
    def update_history(self, observation: Dict) -> None:
        """
        Add new observation to history buffer.
        
        Expected observation format:
        {
            "arrivals": int,
            "hour": int (0-23),
            "day": int (0-6, Monday=0),
            "avg_anger": float (0-10)
        }
        """
        self.history_buffer.append(observation)
        
        # Keep only last sequence_length observations
        if len(self.history_buffer) > self.sequence_length:
            self.history_buffer = self.history_buffer[-self.sequence_length:]
            
    def _prepare_input(self) -> Optional[torch.Tensor]:
        """
        Convert history buffer to model input tensor.
        
        Returns:
            Tensor of shape (1, seq_len, 4) or None if insufficient data
        """
        if len(self.history_buffer) < self.sequence_length:
            return None
            
        # Extract features from buffer
        features = []
        for obs in self.history_buffer[-self.sequence_length:]:
            features.append([
                obs["arrivals"],
                obs["hour"] / 23.0,      # Normalize hour
                obs["day"] / 6.0,        # Normalize day
                obs["avg_anger"] / 10.0  # Normalize anger
            ])
            
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32)
        x = x.unsqueeze(0)  # Add batch dimension
        
        return x.to(self.device)
        
    def predict_with_uncertainty(
        self,
        num_samples: int = 100
    ) -> Optional[Dict]:
        """
        Generate prediction with epistemic uncertainty quantification.
        
        Algorithm:
        1. Enable stochasticity (model.train())
        2. Run N forward passes with same input
        3. Each pass drops different neurons → samples posterior
        4. Aggregate: μ = mean(samples), σ = std(samples)
        5. UCB = μ + 1.96σ (95% confidence bound)
        
        Args:
            num_samples: Number of MC samples (default 100)
            
        Returns:
            Output contract dict or None if insufficient history
        """
        x = self._prepare_input()
        if x is None:
            # Fallback: return heuristic prediction based on typical rates
            # This ensures optimizer always has some arrival estimate
            return self._fallback_prediction()
            
        # CRITICAL: Keep model in train mode for MC Dropout
        self.model.train()
        
        # Collect predictions from multiple stochastic forward passes
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(x)
                predictions.append(pred.item())
                
        predictions = np.array(predictions)
        
        # Statistical aggregation
        mean_pred = float(np.mean(predictions))
        # Use sample std (ddof=1) for unbiased uncertainty estimation
        # This is more defensible for Bayesian sampling than population std
        std_pred = float(np.std(predictions, ddof=1))
        
        # Upper Confidence Bound for risk-aware staffing
        # 1.96 corresponds to 95% confidence interval
        ucb = mean_pred + 1.96 * std_pred
        
        # Construct output following contract
        output = {
            "mean": max(0, mean_pred),  # Arrivals can't be negative
            "std": std_pred,
            "ucb": max(0, ucb),
            "timestamp": datetime.now().isoformat()
        }
        
        return output
        
    def to_json(self, prediction: Dict) -> str:
        """Serialize prediction to JSON string."""
        return json.dumps(prediction)
    
    def _fallback_prediction(self) -> Dict:
        """
        Generate fallback prediction when insufficient history is available.
        
        Uses time-of-day heuristics based on typical bank patterns:
        - Morning (9-11): 20 customers/interval
        - Lunch (11-13): 40 customers/interval (peak)
        - Afternoon (13-17): 15 customers/interval
        - Uncertainty is high (50%) to reflect lack of data
        """
        from datetime import datetime
        hour = datetime.now().hour
        
        # Time-of-day based estimates (per 5-minute interval)
        if 11 <= hour < 13:
            mean_estimate = 8.0  # Lunch rush: ~100/hr / 12 intervals
        elif 9 <= hour < 11:
            mean_estimate = 5.0  # Morning: ~60/hr / 12 intervals  
        else:
            mean_estimate = 3.0  # Off-peak: ~36/hr / 12 intervals
        
        # High uncertainty since this is a heuristic
        std_estimate = mean_estimate * 0.5
        ucb = mean_estimate + 1.96 * std_estimate
        
        return {
            "mean": mean_estimate,
            "std": std_estimate,
            "ucb": ucb,
            "timestamp": datetime.now().isoformat(),
            "source": "fallback_heuristic"  # Flag for debugging
        }
        
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# Training utilities
def create_training_data(
    arrivals: List[int],
    hours: List[int],
    days: List[int],
    anger_scores: List[float],
    sequence_length: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training sequences from historical data.
    
    Returns:
        X: (num_samples, seq_len, 4)
        y: (num_samples, 1)
    """
    n = len(arrivals)
    X_list, y_list = [], []
    
    for i in range(sequence_length, n):
        # Input sequence
        seq = []
        for j in range(i - sequence_length, i):
            seq.append([
                arrivals[j],
                hours[j] / 23.0,
                days[j] / 6.0,
                anger_scores[j] / 10.0
            ])
        X_list.append(seq)
        
        # Target: next arrival count
        y_list.append([arrivals[i]])
        
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    
    return X, y


def train_model(
    model: BayesianLSTM,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> List[float]:
    """
    Train the Bayesian LSTM model.
    
    Args:
        model: BayesianLSTM instance
        X: Training features (num_samples, seq_len, 4)
        y: Training targets (num_samples, 1)
        epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate
        batch_size: Mini-batch size
        
    Returns:
        List of training losses per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Note: We use MSE loss for simplicity; epistemic uncertainty is captured
    # via MC Dropout during inference, not through a probabilistic loss
    criterion = nn.MSELoss()
    
    n_samples = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Mini-batch training
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end - start)
            
        epoch_loss /= n_samples
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
    return losses


if __name__ == "__main__":
    # Demo: Create forecaster and show uncertainty quantification
    print("=" * 60)
    print("Module 1: Bayesian Uncertainty Engine Demo")
    print("=" * 60)
    
    forecaster = BayesianForecaster()
    
    # Simulate some history
    np.random.seed(42)
    for i in range(15):
        forecaster.update_history({
            "arrivals": int(np.random.poisson(10)),
            "hour": 10,
            "day": 2,
            "avg_anger": np.random.uniform(1, 5)
        })
    
    # Get prediction with uncertainty
    result = forecaster.predict_with_uncertainty(num_samples=100)
    
    print("\nPrediction Output Contract:")
    print(json.dumps(result, indent=2))
    print("\nInterpretation:")
    print(f"  Expected arrivals: {result['mean']:.2f}")
    print(f"  Epistemic uncertainty (σ): {result['std']:.2f}")
    print(f"  95% UCB for staffing: {result['ucb']:.2f}")

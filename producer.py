"""
Module 2: Emo-Synthetic Generator (producer.py)
================================================

Purpose: Generate realistic, emotionally rich customer streams using a
Non-Homogeneous Poisson Process with psychological attribute generation.

This replaces static datasets with a psychologically plausible arrival process
that models real-world demand curves and individual customer differences.

Key Features:
- Time-varying arrival rates λ(t)
- Psychological attributes: patience, complexity, contagion susceptibility
- Standardized timestamps (ISO8601 + hour + day_of_week)
- Seed control for reproducible experiments
- Kafka integration for streaming to simulation

Output Contract (per customer event):
{
    "customer_id": str,
    "arrival_time": ISO8601,
    "hour": int (0-23),
    "day_of_week": int (0-6, Monday=0),
    "patience_limit": float (minutes),
    "task_complexity": float (centered at 1.0),
    "contagion_factor": float (0-1),
    "is_lunch_rush": bool
}

Reproducibility Note:
Each experiment run is reproducible via seed control:
    np.random.seed(run_id)
"""

import numpy as np
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Generator, Optional
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArrivalRateSchedule:
    """
    Defines time-varying arrival rate λ(t) for Non-Homogeneous Poisson Process.
    
    Models realistic bank demand curves:
    - Morning ramp-up: 09:00-11:00 → λ: 10→30
    - Lunch rush: 11:00-13:00 → λ: 60 (peak)
    - Afternoon decline: 13:00-15:00 → λ: 20
    - Late afternoon: 15:00-17:00 → λ: 15
    """
    
    def __init__(self):
        # Define rate schedule as (hour_start, hour_end, lambda_value)
        self.schedule = [
            (9, 11, self._linear_ramp(10, 30)),   # Morning ramp-up
            (11, 13, lambda h: 60),                # Lunch rush (constant peak)
            (13, 15, self._linear_ramp(60, 20)),   # Post-lunch decline
            (15, 17, lambda h: 15),                # Late afternoon (steady)
        ]
        
        # Default rate for hours outside schedule
        self.default_rate = 5
        
    def _linear_ramp(self, start_rate: float, end_rate: float):
        """Create a linear interpolation function."""
        def ramp(hour_fraction: float) -> float:
            return start_rate + (end_rate - start_rate) * hour_fraction
        return ramp
    
    def get_rate(self, hour: float) -> float:
        """
        Get arrival rate λ(t) for given hour.
        
        Args:
            hour: Hour of day as float (e.g., 10.5 = 10:30 AM)
            
        Returns:
            Arrival rate (customers per hour)
        """
        for start, end, rate_func in self.schedule:
            if start <= hour < end:
                # Fraction within this period for interpolation
                fraction = (hour - start) / (end - start)
                return rate_func(fraction)
        
        return self.default_rate
    
    def is_lunch_rush(self, hour: float) -> bool:
        """Check if current hour is during lunch rush."""
        return 11 <= hour < 13


class PsychologicalAttributeGenerator:
    """
    Generates psychologically plausible customer attributes.
    
    Each customer differs in:
    - Patience: How long they'll wait before leaving (reneging)
    - Complexity: How difficult their transaction is
    - Contagion: How susceptible they are to emotional spread
    """
    
    def __init__(self, seed: Optional[int] = None):
        # Note: We intentionally use the global numpy RNG to share state
        # with CustomerEventGenerator. For independent streams, use
        # np.random.default_rng(seed) instead.
        if seed is not None:
            np.random.seed(seed)
            
    def generate_patience(self, is_lunch: bool = False) -> float:
        """
        Generate patience limit (minutes until reneging).
        
        P ~ Exponential(β)
        - Base: β = 15 minutes (increased from 10)
        - Lunch: β = 10 minutes (increased from 6)
        
        Returns:
            Patience in minutes
        """
        beta = 10.0 if is_lunch else 15.0
        return float(np.random.exponential(beta))
    
    def generate_complexity(self) -> float:
        """
        Generate task complexity multiplier.
        
        C ~ Normal(1.0, 0.2)
        - < 1.0: Simple transaction (quick deposit)
        - > 1.0: Complex transaction (loan application)
        
        Returns:
            Complexity multiplier (clipped to [0.3, 2.5])
        """
        complexity = np.random.normal(1.0, 0.2)
        return float(np.clip(complexity, 0.3, 2.5))
    
    def generate_contagion_factor(self) -> float:
        """
        Generate emotional contagion susceptibility.
        
        ~ Uniform(0, 1)
        - 0: Immune to crowd emotions
        - 1: Highly susceptible to emotional spread
        
        Returns:
            Contagion factor [0, 1]
        """
        return float(np.random.uniform(0, 1))


class CustomerEventGenerator:
    """
    Main generator combining NHPP arrivals with psychological attributes.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        start_time: Optional[datetime] = None
    ):
        """
        Initialize the customer event generator.
        
        Args:
            seed: Random seed for reproducibility (experiment run ID)
            start_time: Simulation start time (default: today 9:00 AM)
        """
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed} for reproducibility")
        
        # Note: PsychologicalAttributeGenerator shares the global RNG
        # This is intentional for correlated reproducibility across attributes
        self.rate_schedule = ArrivalRateSchedule()
        self.psych_gen = PsychologicalAttributeGenerator()  # No separate seed
        
        # Set start time
        if start_time is None:
            today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
            self.current_time = today
        else:
            self.current_time = start_time
            
        self.customer_count = 0
        
    def _get_current_hour(self) -> float:
        """Get current hour as float (e.g., 10:30 = 10.5)."""
        return self.current_time.hour + self.current_time.minute / 60.0
    
    def _sample_interarrival_time(self) -> float:
        """
        Sample time until next customer using NHPP.
        
        For Poisson process: t_next = -ln(U) / λ(t)
        where U ~ Uniform(0,1)
        
        Note: We approximate NHPP by assuming λ(t) is locally constant
        over short inter-arrival intervals. This is valid given slow
        hour-scale rate variation vs minute-scale inter-arrivals.
        
        Returns:
            Inter-arrival time in minutes
        """
        current_hour = self._get_current_hour()
        rate = self.rate_schedule.get_rate(current_hour)  # customers per hour
        
        # Convert rate from per-hour to per-minute
        rate_per_minute = rate / 60.0  # customers per minute
        
        # Sample exponential inter-arrival time
        u = np.random.uniform(0, 1)
        interarrival = -np.log(u) / rate_per_minute  # minutes
        
        return interarrival
    
    def generate_customer(self) -> Dict:
        """
        Generate a single customer event with all attributes.
        
        Returns:
            Customer event dict following output contract
        """
        # Sample inter-arrival time and advance clock
        interarrival = self._sample_interarrival_time()
        self.current_time += timedelta(minutes=interarrival)
        
        # Get time metadata
        hour = self.current_time.hour
        is_lunch = self.rate_schedule.is_lunch_rush(hour)
        
        # Generate psychological attributes
        patience = self.psych_gen.generate_patience(is_lunch)
        complexity = self.psych_gen.generate_complexity()
        contagion = self.psych_gen.generate_contagion_factor()
        
        self.customer_count += 1
        
        # Construct event following output contract
        # Note: Short ID (8 chars) for logging readability
        # Not guaranteed unique across very long runs (>1M customers)
        event = {
            "customer_id": str(uuid.uuid4())[:8],
            "arrival_time": self.current_time.isoformat(),
            "hour": hour,
            "day_of_week": self.current_time.weekday(),
            "patience_limit": round(patience, 2),
            "task_complexity": round(complexity, 3),
            "contagion_factor": round(contagion, 3),
            "is_lunch_rush": is_lunch
        }
        
        return event
    
    def generate_stream(
        self,
        duration_hours: float = 8.0
    ) -> Generator[Dict, None, None]:
        """
        Generate continuous stream of customer events.
        
        Args:
            duration_hours: Total simulation duration
            
        Yields:
            Customer event dicts
        """
        end_time = self.current_time + timedelta(hours=duration_hours)
        
        while self.current_time < end_time:
            yield self.generate_customer()


class KafkaCustomerProducer:
    """
    Produces customer events to Kafka topic 'bank_arrivals'.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "bank_arrivals"
    ):
        self.topic = topic
        self.producer = None
        self.bootstrap_servers = bootstrap_servers
        
    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """
        Connect to Kafka broker with retries.
        
        Returns:
            True if connected, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    api_version=(2, 5, 0)
                )
                logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
                return True
            except NoBrokersAvailable:
                logger.warning(f"Kafka not available, retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                
        logger.error("Failed to connect to Kafka")
        return False
    
    def publish(self, event: Dict) -> None:
        """Publish customer event to Kafka."""
        if self.producer:
            self.producer.send(self.topic, event)
            self.producer.flush()
            
    def close(self) -> None:
        """Close Kafka producer."""
        if self.producer:
            self.producer.close()


def run_producer(
    seed: int = 42,
    duration_hours: float = 8.0,
    real_time: bool = False,
    kafka_enabled: bool = True
) -> None:
    """
    Main function to run the customer generator.
    
    Args:
        seed: Random seed for reproducibility
        duration_hours: Simulation duration
        real_time: If True, sleep between events (for live demo).
                   Note: Real-time mode is for visualization only and
                   does not preserve true temporal scaling.
        kafka_enabled: If True, publish to Kafka
    """
    logger.info("=" * 60)
    logger.info("Module 2: Emo-Synthetic Generator Starting")
    logger.info(f"Seed: {seed} (reproducible)")
    logger.info("=" * 60)
    
    # Initialize generator
    generator = CustomerEventGenerator(seed=seed)
    
    # Initialize Kafka producer if enabled
    kafka_producer = None
    if kafka_enabled:
        kafka_producer = KafkaCustomerProducer()
        if not kafka_producer.connect():
            logger.warning("Running without Kafka (events will only be logged)")
            kafka_producer = None
    
    # Generate and publish events
    try:
        for event in generator.generate_stream(duration_hours):
            # Log event
            logger.info(f"Customer {event['customer_id']}: "
                       f"patience={event['patience_limit']:.1f}min, "
                       f"complexity={event['task_complexity']:.2f}, "
                       f"contagion={event['contagion_factor']:.2f}"
                       f"{' [LUNCH RUSH]' if event['is_lunch_rush'] else ''}")
            
            # Publish to Kafka
            if kafka_producer:
                kafka_producer.publish(event)
                
            # Real-time pacing (optional)
            if real_time:
                time.sleep(0.1)  # Compressed time for demo
                
    except KeyboardInterrupt:
        logger.info("Producer stopped by user")
    finally:
        if kafka_producer:
            kafka_producer.close()
            
    logger.info(f"\nTotal customers generated: {generator.customer_count}")


if __name__ == "__main__":
    # Demo run with seed for reproducibility
    run_producer(seed=42, duration_hours=1.0, kafka_enabled=False)

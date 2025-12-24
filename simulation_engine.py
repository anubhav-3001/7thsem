"""
Module 3: Affective Digital Twin (simulation_engine.py)
========================================================

The heart of realism: A simulation engine where emotions and fatigue
directly change system behavior over time.

Queue Model:
"The system is modeled as a single shared queue with multiple fatigue-aware servers."

Key Concepts:

1. LobbyAnger (Collective Frustration Index):
   LobbyAnger = min(10, mean(wait_i) / W_ref)
   where W_ref = 5 minutes (acceptable wait threshold)
   
   NOTE: LobbyAnger represents a normalized collective frustration INDEX,
   not an emotional state of any single agent. This is a system-level metric.

2. FatigueTeller (Server with Human Limits):
   - Efficiency decays with fatigue: η = 1 - 0.6 × fatigue
   - Service time: base × complexity / η
   - Fatigue accumulates: +0.01 × service_time
   - Recovery during breaks: fatigue × e^(-0.1 × minutes)

3. Emotional Contagion:
   P_eff = P_base × e^(-0.5 × LobbyAnger)
   Customers tolerate less when everyone is angry.

Output Events (Kafka 'bank_simulation'):
{
    "event_type": "SERVE" | "RENEGE" | "FATIGUE_UPDATE" | "ANGER_UPDATE",
    "timestamp": ISO8601,
    "details": {...}
}

Time Semantics:
Simulation time (env.now) is decoupled from wall-clock timestamps
(datetime.now()) used for logging and Kafka event correlation.
"""

import simpy
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Reference wait time for anger normalization (5 minutes)
W_REF = 5.0

# Base service time in minutes
BASE_SERVICE_TIME = 3.0

# Fatigue parameters
FATIGUE_ACCUMULATION_RATE = 0.01  # Per minute of service
MAX_FATIGUE = 1.0
FATIGUE_EFFICIENCY_IMPACT = 0.6  # Max efficiency loss at fatigue=1

# Recovery parameters
BREAK_RECOVERY_RATE = 0.1  # Exponential decay rate

# Contagion parameters  
# Note: Sensitivity of 0.15 prevents unrealistic queue collapse at high anger
CONTAGION_SENSITIVITY = 0.15  # Reduced from 0.3 for more stability


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Customer:
    """Customer agent with psychological attributes."""
    customer_id: str
    arrival_time: float  # Simulation time
    patience_limit: float  # Base patience in minutes
    task_complexity: float  # Multiplier for service time
    contagion_factor: float  # Susceptibility to crowd emotions
    
    # Dynamic state
    wait_start: float = 0.0
    effective_patience: float = 0.0
    served: bool = False
    reneged: bool = False
    
    def __post_init__(self):
        self.wait_start = self.arrival_time
        self.effective_patience = self.patience_limit


@dataclass 
class SimulationMetrics:
    """Real-time metrics collection."""
    total_arrivals: int = 0
    total_served: int = 0
    total_reneged: int = 0
    total_wait_time: float = 0.0
    current_queue_length: int = 0
    current_lobby_anger: float = 0.0
    
    # Per-interval tracking
    interval_arrivals: int = 0
    interval_wait_times: List[float] = field(default_factory=list)
    
    # Track reneged wait for accurate pain metrics
    total_reneged_wait_time: float = 0.0


# =============================================================================
# FATIGUE-AWARE TELLER
# =============================================================================

class FatigueTeller:
    """
    Server model with human-realistic fatigue dynamics.
    
    Fatigue affects service efficiency:
    - Fresh teller (fatigue=0): η = 1.0 (100% efficiency)
    - Exhausted teller (fatigue=1): η = 0.4 (40% efficiency)
    
    Service time formula:
    T_service = BaseTime × complexity / η
    """
    
    def __init__(
        self,
        teller_id: int,
        env: simpy.Environment,
        initial_fatigue: float = 0.0
    ):
        self.teller_id = teller_id
        self.env = env
        self.fatigue = initial_fatigue
        self.customers_served = 0
        self.is_on_break = False
        self.total_service_time = 0.0
        
    @property
    def efficiency(self) -> float:
        """
        Calculate current service efficiency.
        η = 1 - (0.6 × fatigue), clamped to minimum 0.3
        
        Lower bound prevents divide-by-small-number explosions.
        """
        raw_efficiency = 1.0 - (FATIGUE_EFFICIENCY_IMPACT * self.fatigue)
        return max(0.3, raw_efficiency)  # Safety clamp
    
    def calculate_service_time(self, complexity: float) -> float:
        """
        Calculate actual service time given task complexity.
        T_service = BaseTime × complexity / η
        """
        return BASE_SERVICE_TIME * complexity / self.efficiency
    
    def serve_customer(self, customer: Customer) -> float:
        """
        Serve a customer and update fatigue.
        
        Returns:
            Actual service time in minutes
        """
        service_time = self.calculate_service_time(customer.task_complexity)
        
        # Accumulate fatigue with diminishing returns
        # fatigue_new = fatigue_old + (1 - fatigue) × rate × T_service
        # This creates fast early fatigue, slower burnout near exhaustion
        # More defensible in human factors research
        fatigue_increment = (1 - self.fatigue) * FATIGUE_ACCUMULATION_RATE * service_time
        self.fatigue = min(MAX_FATIGUE, self.fatigue + fatigue_increment)
        
        self.customers_served += 1
        self.total_service_time += service_time
        
        return service_time
    
    def take_break(self, duration_minutes: float) -> None:
        """
        Recovery during break.
        fatigue_new = fatigue_old × e^(-0.1 × minutes)
        """
        self.is_on_break = True
        recovery_factor = np.exp(-BREAK_RECOVERY_RATE * duration_minutes)
        self.fatigue = self.fatigue * recovery_factor
        self.is_on_break = False
        
        logger.info(f"Teller {self.teller_id} break complete. "
                   f"Fatigue: {self.fatigue:.2f}")
    
    def get_status(self) -> Dict:
        """Get current teller status."""
        return {
            "teller_id": self.teller_id,
            "fatigue": round(self.fatigue, 3),
            "efficiency": round(self.efficiency, 3),
            "customers_served": self.customers_served,
            "is_on_break": self.is_on_break,
            "is_burnt_out": self.fatigue > 0.8
        }


# =============================================================================
# LOBBY ANGER INDEX
# =============================================================================

class LobbyAngerTracker:
    """
    Tracks collective frustration index (LobbyAnger).
    
    Formula: LobbyAnger = min(10, mean(wait_i) / W_ref)
    
    NOTE: This represents a normalized collective frustration INDEX,
    not an emotional state of any single agent. It's a system-level
    metric that influences individual patience through contagion.
    """
    
    def __init__(self, w_ref: float = W_REF):
        self.w_ref = w_ref  # Acceptable wait threshold (5 min)
        self.current_waits: List[float] = []
        self.anger_history: List[Dict] = []
        self.current_anger: float = 0.0
        
    def update_waits(self, wait_times: List[float]) -> float:
        """
        Update LobbyAnger based on current wait times.
        
        Args:
            wait_times: List of current wait times for customers in queue
            
        Returns:
            Updated LobbyAnger value [0, 10]
        """
        self.current_waits = wait_times
        
        if not wait_times:
            new_anger = 0.0
        else:
            # Use median for robustness against outliers
            # This prevents one stuck customer → global anger spike
            median_wait = np.median(wait_times)
            # Normalized: min(10, median_wait / W_ref)
            new_anger = min(10.0, median_wait / self.w_ref)
        
        # Apply anger inertia for emotional memory / slower recovery
        # This models realistic slow recovery after congestion
        self.current_anger = 0.7 * self.current_anger + 0.3 * new_anger
            
        return self.current_anger
    
    def record(self, timestamp: str) -> None:
        """Record anger for history/dashboard."""
        self.anger_history.append({
            "timestamp": timestamp,
            "anger": self.current_anger,
            "queue_size": len(self.current_waits)
        })
        
    def get_effective_patience(
        self,
        base_patience: float,
        contagion_factor: float
    ) -> float:
        """
        Calculate effective patience under emotional contagion.
        
        P_eff = P_base × e^(-0.5 × LobbyAnger × contagion_factor)
        
        Key insight: People tolerate less when everyone is angry,
        modulated by their individual contagion susceptibility.
        """
        # Cap effective anger to prevent unrealistic queue collapse
        # At max anger (10) and high contagion, patience drops to ~5% baseline
        effective_anger = min(self.current_anger, 6.0)  # Cap at danger threshold
        
        contagion_effect = np.exp(-CONTAGION_SENSITIVITY * 
                                  effective_anger * 
                                  contagion_factor)
        return base_patience * contagion_effect


# =============================================================================
# MAIN SIMULATION ENGINE
# =============================================================================

class AffectiveSimulationEngine:
    """
    SimPy-based discrete event simulation with emotional dynamics.
    
    Architecture:
    - Single shared queue (FIFO)
    - Multiple FatigueTeller servers
    - Real-time LobbyAnger tracking
    - Customer contagion and reneging
    """
    
    def __init__(
        self,
        num_tellers: int = 3,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)
            
        self.env = simpy.Environment()
        self.queue: simpy.Store = simpy.Store(self.env)
        
        # Initialize tellers
        # Note: Service capacity is managed manually via teller availability,
        # not SimPy resources. This allows fatigue-aware scheduling.
        self.tellers: List[FatigueTeller] = [
            FatigueTeller(i, self.env) for i in range(num_tellers)
        ]
        
        # State trackers
        self.anger_tracker = LobbyAngerTracker()
        self.metrics = SimulationMetrics()
        
        # Customer tracking
        self.waiting_customers: Dict[str, Customer] = {}
        
        # Kafka integration
        self.kafka_producer: Optional[KafkaProducer] = None
        self.kafka_consumer: Optional[KafkaConsumer] = None
        
        # Event callbacks
        self.on_serve: Optional[Callable] = None
        self.on_renege: Optional[Callable] = None
        
        # Control flags
        self.running = False
        
    def connect_kafka(
        self,
        bootstrap_servers: str = "localhost:9092"
    ) -> bool:
        """Connect to Kafka for event streaming."""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 5, 0)
            )
            
            self.kafka_consumer = KafkaConsumer(
                'bank_arrivals',
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                api_version=(2, 5, 0)
            )
            
            logger.info("Kafka connected for simulation")
            return True
            
        except NoBrokersAvailable:
            logger.warning("Kafka not available, running standalone")
            return False
            
    def publish_event(self, event: Dict) -> None:
        """Publish simulation event to Kafka."""
        if self.kafka_producer:
            self.kafka_producer.send('bank_simulation', event)
            self.kafka_producer.flush()
            
    def add_customer(self, customer: Customer) -> None:
        """Add customer to queue."""
        customer.wait_start = self.env.now
        self.waiting_customers[customer.customer_id] = customer
        self.metrics.total_arrivals += 1
        self.metrics.interval_arrivals += 1
        
        # Start customer patience process
        self.env.process(self._customer_patience_process(customer))
        
    def _customer_patience_process(self, customer: Customer) -> simpy.events.Process:
        """
        Process that monitors customer patience and triggers reneging.
        """
        while customer.customer_id in self.waiting_customers:
            # Update effective patience based on current anger
            customer.effective_patience = self.anger_tracker.get_effective_patience(
                customer.patience_limit,
                customer.contagion_factor
            )
            
            # Check if patience exceeded
            current_wait = self.env.now - customer.wait_start
            
            if current_wait >= customer.effective_patience:
                # Customer reneges
                customer.reneged = True
                self.waiting_customers.pop(customer.customer_id, None)
                self.metrics.total_reneged += 1
                
                # Track reneged wait time for accurate pain metrics
                self.metrics.total_reneged_wait_time += current_wait
                
                renege_event = {
                    "event_type": "RENEGE",
                    "timestamp": datetime.now().isoformat(),
                    "customer_id": customer.customer_id,
                    "wait_time": round(current_wait, 2),
                    "effective_patience": round(customer.effective_patience, 2),
                    "lobby_anger": round(self.anger_tracker.current_anger, 2),
                    "reason": "Contagion-induced patience exhaustion"
                }
                
                self.publish_event(renege_event)
                logger.info(f"RENEGE: Customer {customer.customer_id} left "
                           f"after {current_wait:.1f}min (anger={self.anger_tracker.current_anger:.1f})")
                
                if self.on_renege:
                    self.on_renege(renege_event)
                    
                return
                
            # Adaptive check interval: less frequent for patient customers
            # This reduces overhead in large queues
            check_interval = min(1.0, customer.effective_patience / 10)
            yield self.env.timeout(max(0.25, check_interval))
            
    def _get_available_teller(self) -> Optional[FatigueTeller]:
        """Get least fatigued available teller."""
        available = [t for t in self.tellers if not t.is_on_break]
        if not available:
            return None
        return min(available, key=lambda t: t.fatigue)
        
    def _service_process(self) -> simpy.events.Process:
        """Main service loop - picks customers and assigns to tellers."""
        while self.running:
            if self.waiting_customers:
                # Get next customer (FIFO)
                customer_id = next(iter(self.waiting_customers))
                customer = self.waiting_customers.pop(customer_id)
                
                # Get teller
                teller = self._get_available_teller()
                if teller is None:
                    # Put customer back if no teller available
                    self.waiting_customers[customer_id] = customer
                    yield self.env.timeout(0.1)
                    continue
                
                # Defensive check: ensure teller is not on break
                assert not teller.is_on_break, "Teller assigned while on break"
                    
                # Calculate wait time
                wait_time = self.env.now - customer.wait_start
                self.metrics.total_wait_time += wait_time
                self.metrics.interval_wait_times.append(wait_time)
                
                # Service
                service_time = teller.serve_customer(customer)
                yield self.env.timeout(service_time)
                
                # Mark served
                customer.served = True
                self.metrics.total_served += 1
                
                serve_event = {
                    "event_type": "SERVE",
                    "timestamp": datetime.now().isoformat(),
                    "customer_id": customer.customer_id,
                    "teller_id": teller.teller_id,
                    "wait_time": round(wait_time, 2),
                    "service_time": round(service_time, 2),
                    "teller_fatigue": round(teller.fatigue, 3),
                    "complexity": round(customer.task_complexity, 3)
                }
                
                self.publish_event(serve_event)
                
                if self.on_serve:
                    self.on_serve(serve_event)
                    
            else:
                yield self.env.timeout(0.1)
                
    def _anger_update_process(self) -> simpy.events.Process:
        """Update LobbyAnger every simulation minute."""
        while self.running:
            yield self.env.timeout(1.0)  # Every minute
            
            # Calculate current wait times
            current_waits = [
                self.env.now - c.wait_start 
                for c in self.waiting_customers.values()
            ]
            
            # Update anger
            anger = self.anger_tracker.update_waits(current_waits)
            self.anger_tracker.record(datetime.now().isoformat())
            self.metrics.current_lobby_anger = anger
            self.metrics.current_queue_length = len(self.waiting_customers)
            
            anger_event = {
                "event_type": "ANGER_UPDATE",
                "timestamp": datetime.now().isoformat(),
                "lobby_anger": round(anger, 2),
                "queue_length": len(current_waits),
                "mean_wait": round(np.mean(current_waits), 2) if current_waits else 0,
                "teller_fatigue": [
                    {"teller_id": t.teller_id, "fatigue": round(t.fatigue, 3)}
                    for t in self.tellers
                ],
                # Include metrics for dashboard
                "metrics": {
                    "total_arrivals": self.metrics.total_arrivals,
                    "total_served": self.metrics.total_served,
                    "total_reneged": self.metrics.total_reneged,
                    "avg_wait": round(
                        self.metrics.total_wait_time / max(1, self.metrics.total_served), 2
                    ) if self.metrics.total_served > 0 else 0.0,
                    "renege_rate": round(
                        100 * self.metrics.total_reneged / max(1, self.metrics.total_arrivals), 1
                    ) if self.metrics.total_arrivals > 0 else 0.0
                }
            }
            
            self.publish_event(anger_event)
            
    def give_teller_break(self, teller_id: int, duration: float = 10.0) -> None:
        """Give a teller a break for recovery."""
        if 0 <= teller_id < len(self.tellers):
            teller = self.tellers[teller_id]
            self.env.process(self._break_process(teller, duration))
            
    def _break_process(
        self,
        teller: FatigueTeller,
        duration: float
    ) -> simpy.events.Process:
        """Process for teller break."""
        teller.is_on_break = True
        yield self.env.timeout(duration)
        teller.take_break(duration)
        teller.is_on_break = False
        
    def add_teller(self) -> int:
        """Add a new teller. Returns new teller ID."""
        new_id = len(self.tellers)
        new_teller = FatigueTeller(new_id, self.env)
        self.tellers.append(new_teller)
        logger.info(f"Added teller {new_id}")
        return new_id
        
    def remove_teller(self) -> bool:
        """Remove least busy teller. Returns success."""
        if len(self.tellers) <= 1:
            return False
            
        # Remove least fatigued (most rested)
        available = [t for t in self.tellers if not t.is_on_break]
        if not available:
            return False
            
        to_remove = min(available, key=lambda t: t.fatigue)
        self.tellers.remove(to_remove)
        logger.info(f"Removed teller {to_remove.teller_id}")
        return True
        
    def get_state(self) -> Dict:
        """Get complete simulation state for dashboard/optimizer."""
        return {
            "timestamp": datetime.now().isoformat(),
            "simulation_time": round(self.env.now, 2),
            "lobby_anger": round(self.anger_tracker.current_anger, 2),
            "queue_length": len(self.waiting_customers),
            "tellers": [t.get_status() for t in self.tellers],
            "metrics": {
                "total_arrivals": self.metrics.total_arrivals,
                "total_served": self.metrics.total_served,
                "total_reneged": self.metrics.total_reneged,
                # Include reneged wait time for accurate pain measurement
                "avg_wait": round(
                    (self.metrics.total_wait_time + self.metrics.total_reneged_wait_time) 
                    / max(1, self.metrics.total_arrivals), 2
                ),
                "renege_rate": round(
                    self.metrics.total_reneged / max(1, self.metrics.total_arrivals) * 100, 1
                )
            }
        }
        
    def get_fatigue_summary(self) -> Dict:
        """Get fatigue summary for optimizer."""
        return {
            "num_tellers": len(self.tellers),
            "avg_fatigue": round(np.mean([t.fatigue for t in self.tellers]), 3),
            "max_fatigue": round(max(t.fatigue for t in self.tellers), 3),
            "burnt_out_count": sum(1 for t in self.tellers if t.fatigue > 0.8),
            "teller_fatigue": {t.teller_id: t.fatigue for t in self.tellers}
        }
        
    def reset_interval_metrics(self) -> Dict:
        """Reset interval metrics and return summary."""
        summary = {
            "interval_arrivals": self.metrics.interval_arrivals,
            "interval_avg_wait": round(
                np.mean(self.metrics.interval_wait_times) 
                if self.metrics.interval_wait_times else 0, 2
            ),
            "interval_anger": round(self.anger_tracker.current_anger, 2)
        }
        
        self.metrics.interval_arrivals = 0
        self.metrics.interval_wait_times = []
        
        return summary
        
    def start(self) -> None:
        """Start simulation processes."""
        self.running = True
        self.env.process(self._service_process())
        self.env.process(self._anger_update_process())
        
    def step(self, duration: float = 1.0) -> None:
        """Advance simulation by duration minutes."""
        self.env.run(until=self.env.now + duration)
        
    def stop(self) -> None:
        """Stop simulation."""
        self.running = False
        if self.kafka_producer:
            self.kafka_producer.close()


# =============================================================================
# DEMO
# =============================================================================

def run_demo():
    """Demo: Run standalone simulation with synthetic customers."""
    logger.info("=" * 60)
    logger.info("Module 3: Affective Digital Twin Demo")
    logger.info("Queue Model: Single shared queue, multiple fatigue-aware servers")
    logger.info("=" * 60)
    
    # Create engine
    engine = AffectiveSimulationEngine(num_tellers=3, seed=42)
    engine.start()
    
    # Simulate 30 minutes with customer arrivals
    np.random.seed(42)
    
    for minute in range(30):
        # Random arrivals (Poisson with λ=4)
        num_arrivals = np.random.poisson(4)
        
        for _ in range(num_arrivals):
            customer = Customer(
                customer_id=f"C{engine.metrics.total_arrivals:04d}",
                arrival_time=engine.env.now,
                patience_limit=np.random.exponential(8),
                task_complexity=np.clip(np.random.normal(1.0, 0.2), 0.3, 2.5),
                contagion_factor=np.random.uniform(0, 1)
            )
            engine.add_customer(customer)
            
        # Step simulation
        engine.step(1.0)
        
        # Log every 5 minutes
        if minute % 5 == 4:
            state = engine.get_state()
            logger.info(f"\n--- Minute {minute+1} ---")
            logger.info(f"LobbyAnger: {state['lobby_anger']:.1f}/10")
            logger.info(f"Queue: {state['queue_length']} waiting")
            logger.info(f"Served: {state['metrics']['total_served']}, "
                       f"Reneged: {state['metrics']['total_reneged']}")
            for t in state['tellers']:
                status = "🔴 BURNT" if t['is_burnt_out'] else "🟢 OK"
                logger.info(f"  Teller {t['teller_id']}: "
                           f"fatigue={t['fatigue']:.2f} {status}")
                           
    engine.stop()
    
    # Final summary
    final_state = engine.get_state()
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info(f"Total Arrivals: {final_state['metrics']['total_arrivals']}")
    logger.info(f"Total Served: {final_state['metrics']['total_served']}")
    logger.info(f"Total Reneged: {final_state['metrics']['total_reneged']} "
               f"({final_state['metrics']['renege_rate']:.1f}%)")
    logger.info(f"Avg Wait Time: {final_state['metrics']['avg_wait']:.1f} min")


if __name__ == "__main__":
    run_demo()

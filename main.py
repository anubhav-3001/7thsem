"""
Main Orchestrator (main.py)
============================

Orchestrates execution of all modules in the socio-technical system.

Design Principle:
"All control logic remains decentralized; main.py only orchestrates execution."

This file does NOT contain business logic. It only:
1. Starts processes
2. Wires Kafka topics  
3. Handles graceful shutdown

Each module remains independently runnable for testing and debugging.
"""

import threading
import signal
import sys
import time
import logging
from datetime import datetime
from typing import Optional
import subprocess
import os

# Import modules
from forecaster import BayesianForecaster, BayesianLSTM
from producer import CustomerEventGenerator, KafkaCustomerProducer
from simulation_engine import (
    AffectiveSimulationEngine, 
    Customer, 
    SimulationMetrics
)
from optimization_agent import OptimizationAgent, SystemState, Action

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('orchestrator')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration."""
    KAFKA_BOOTSTRAP = "localhost:9092"
    INITIAL_TELLERS = 3  # Start low to see ADD_TELLER decisions
    SIMULATION_SPEED = 1.0  # 1.0 = real-time, 0.1 = fast
    DECISION_INTERVAL = 2.0  # Reduced from 5.0 for faster response
    FORECASTER_SEQUENCE_LEN = 10  # Reduced from 15 for faster predictions
    RANDOM_SEED = 42


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class SystemOrchestrator:
    """
    Coordinates all modules without implementing control logic.
    
    Responsibilities:
    - Start/stop module threads
    - Wire data flow between modules
    - Handle graceful shutdown
    """
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.running = False
        self.threads = []
        
        # Module instances
        self.forecaster: Optional[BayesianForecaster] = None
        self.producer: Optional[CustomerEventGenerator] = None
        self.simulation: Optional[AffectiveSimulationEngine] = None
        self.optimizer: Optional[OptimizationAgent] = None
        
        # Kafka integration
        self.kafka_enabled = False
        
        # Shutdown event
        self.shutdown_event = threading.Event()
        
    def _init_modules(self) -> None:
        """Initialize all modules."""
        logger.info("Initializing modules...")
        
        # Module 1: Bayesian Forecaster
        self.forecaster = BayesianForecaster(
            sequence_length=self.config.FORECASTER_SEQUENCE_LEN
        )
        # Load pre-trained weights if available
        import os
        weights_path = os.path.join(os.path.dirname(__file__), "forecaster_weights.pth")
        if os.path.exists(weights_path):
            self.forecaster.load_model(weights_path)
            logger.info("✓ Forecaster initialized (pre-trained weights loaded)")
        else:
            logger.info("✓ Forecaster initialized (no pre-trained weights found)")
        
        # Module 2: Customer Producer
        self.producer = CustomerEventGenerator(
            seed=self.config.RANDOM_SEED
        )
        logger.info("✓ Producer initialized")
        
        # Module 3: Simulation Engine
        self.simulation = AffectiveSimulationEngine(
            num_tellers=self.config.INITIAL_TELLERS,
            seed=self.config.RANDOM_SEED
        )
        logger.info("✓ Simulation engine initialized")
        
        # Module 4: Optimization Agent
        self.optimizer = OptimizationAgent()
        logger.info("✓ Optimizer initialized")
        
    def _connect_kafka(self) -> bool:
        """Attempt to connect all modules to Kafka."""
        logger.info("Connecting to Kafka...")
        
        try:
            # Test connection
            self.kafka_enabled = self.simulation.connect_kafka(
                self.config.KAFKA_BOOTSTRAP
            )
            
            if self.kafka_enabled:
                self.optimizer.connect_kafka(self.config.KAFKA_BOOTSTRAP)
                logger.info("✓ Kafka connected")
            else:
                logger.warning("⚠ Running without Kafka (standalone mode)")
                
            return self.kafka_enabled
            
        except Exception as e:
            logger.warning(f"⚠ Kafka connection failed: {e}")
            return False
            
    def _producer_loop(self) -> None:
        """Thread: Generate customers and feed to simulation."""
        logger.info("Producer thread started")
        
        while not self.shutdown_event.is_set():
            # Generate next customer (this samples inter-arrival time from NHPP)
            event = self.producer.generate_customer()
            
            # Create Customer object for simulation
            customer = Customer(
                customer_id=event["customer_id"],
                arrival_time=self.simulation.env.now,
                patience_limit=event["patience_limit"],
                task_complexity=event["task_complexity"],
                contagion_factor=event["contagion_factor"]
            )
            
            # Add to simulation
            self.simulation.add_customer(customer)
            
            # Update forecaster with arrival data
            self.forecaster.update_history({
                "arrivals": 1,
                "hour": event["hour"],
                "day": event["day_of_week"],
                "avg_anger": self.simulation.anger_tracker.current_anger
            })
            
            # Pace arrivals using NHPP inter-arrival time (converted to real seconds)
            # At speed 0.1, 1 simulated minute = 6 real seconds
            # Typical inter-arrival at λ=30/hr is ~2 min = 0.2 real seconds at 10x speed
            interarrival_real_seconds = 0.5 * self.config.SIMULATION_SPEED  # Simplified pacing
            time.sleep(interarrival_real_seconds)
            
    def _simulation_loop(self) -> None:
        """Thread: Run simulation engine."""
        logger.info("Simulation thread started")
        
        self.simulation.start()
        
        while not self.shutdown_event.is_set():
            # Step simulation
            self.simulation.step(duration=1.0)
            
            # Pace simulation
            time.sleep(0.1 * self.config.SIMULATION_SPEED)
            
    def _optimizer_loop(self) -> None:
        """Thread: Run optimization decisions."""
        logger.info("Optimizer thread started")
        
        decision_interval_seconds = self.config.DECISION_INTERVAL * 60 * self.config.SIMULATION_SPEED * 0.1
        
        while not self.shutdown_event.is_set():
            # Wait for decision interval
            time.sleep(decision_interval_seconds)
            
            if self.shutdown_event.is_set():
                break
            
            # Get simulation state first to update forecaster history
            sim_state = self.simulation.get_state()
            
            # Update forecaster with actual observations (CRITICAL for LSTM)
            from datetime import datetime
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            self.forecaster.update_history({
                "arrivals": self.simulation.metrics.interval_arrivals,
                "hour": current_hour,
                "day": current_day,
                "avg_anger": self.simulation.anger_tracker.current_anger
            })
            
            # Get forecaster prediction (now uses trained model if enough history)
            prediction = self.forecaster.predict_with_uncertainty(num_samples=50)
            
            if prediction is None:
                continue
                
            # Get fatigue summary
            fatigue_summary = self.simulation.get_fatigue_summary()
            
            # Build optimizer state
            state = SystemState(
                num_tellers=fatigue_summary["num_tellers"],
                current_queue=len(self.simulation.waiting_customers),
                avg_fatigue=fatigue_summary["avg_fatigue"],
                max_fatigue=fatigue_summary["max_fatigue"],
                burnt_out_count=fatigue_summary["burnt_out_count"],
                teller_fatigue=fatigue_summary["teller_fatigue"],
                lobby_anger=self.simulation.anger_tracker.current_anger,
                predicted_arrivals_mean=prediction["mean"],
                predicted_arrivals_ucb=prediction["ucb"],
                prediction_uncertainty=prediction["std"],
                current_wait=sim_state["metrics"]["avg_wait"]
            )
            
            # Get decision
            action, command = self.optimizer.decide(state)
            
            # Execute action
            self._execute_action(action, command)
            
            # Log decision
            logger.info(f"🎯 Decision: {action.value} | "
                       f"Queue: {state.current_queue} | "
                       f"UCB: {state.predicted_arrivals_ucb:.1f}")
            
            # Publish prediction to dashboard
            if self.optimizer.kafka_producer:
                prediction_event = {
                    "event_type": "PREDICTION_UPDATE",
                    "timestamp": prediction.get("timestamp", datetime.now().isoformat()),
                    "actual_arrivals": self.simulation.metrics.interval_arrivals,
                    "mean": prediction["mean"],
                    "std": prediction["std"],
                    "ucb": prediction["ucb"]
                }
                self.optimizer.kafka_producer.send('bank_simulation', prediction_event)
                self.optimizer.kafka_producer.flush()
                # Reset interval counter after publishing
                self.simulation.metrics.interval_arrivals = 0
                       
    def _execute_action(self, action: Action, command: dict) -> None:
        """Execute optimization decision on simulation."""
        if action == Action.ADD_TELLER:
            self.simulation.add_teller()
            
        elif action == Action.REMOVE_TELLER:
            self.simulation.remove_teller()
            
        elif action == Action.GIVE_BREAK:
            teller_id = command.get("teller_id")
            if teller_id is not None:
                self.simulation.give_teller_break(teller_id, duration=10.0)
                
        # DO_NOTHING and DELAY_DECISION require no action
        
    def _status_loop(self) -> None:
        """Thread: Print periodic status updates."""
        while not self.shutdown_event.is_set():
            time.sleep(5)
            
            if self.shutdown_event.is_set():
                break
                
            state = self.simulation.get_state()
            
            logger.info(
                f"📊 Status | "
                f"Queue: {state['queue_length']} | "
                f"Anger: {state['lobby_anger']:.1f} | "
                f"Served: {state['metrics']['total_served']} | "
                f"Reneged: {state['metrics']['total_reneged']}"
            )
            
    def start(self) -> None:
        """Start all modules."""
        logger.info("=" * 60)
        logger.info("SOCIO-TECHNICAL SERVICE SYSTEM")
        logger.info("Closed-loop operation starting...")
        logger.info("=" * 60)
        
        # Initialize
        self._init_modules()
        self._connect_kafka()
        
        self.running = True
        
        # Start threads
        threads_config = [
            ("Producer", self._producer_loop),
            ("Simulation", self._simulation_loop),
            ("Optimizer", self._optimizer_loop),
            ("Status", self._status_loop)
        ]
        
        for name, target in threads_config:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)
            
        logger.info(f"Started {len(self.threads)} threads")
        
    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Initiating shutdown...")
        
        self.shutdown_event.set()
        self.running = False
        
        # Wait for threads
        for t in self.threads:
            t.join(timeout=2.0)
            
        # Stop simulation
        if self.simulation:
            self.simulation.stop()
            
        logger.info("Shutdown complete")
        
        # Print final summary
        self._print_summary()
        
    def _print_summary(self) -> None:
        """Print final system summary."""
        if not self.simulation:
            return
            
        state = self.simulation.get_state()
        metrics = state["metrics"]
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Arrivals: {metrics['total_arrivals']}")
        logger.info(f"Total Served: {metrics['total_served']}")
        logger.info(f"Total Reneged: {metrics['total_reneged']} ({metrics['renege_rate']:.1f}%)")
        logger.info(f"Avg Wait Time: {metrics['avg_wait']:.1f} min")
        
        # Decision summary
        if self.optimizer:
            decisions = self.optimizer.get_decision_trace()
            logger.info(f"Total Decisions: {len(decisions)}")
            
    def wait(self) -> None:
        """Wait for shutdown signal."""
        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def run_dashboard():
    """Start dashboard in separate process."""
    logger.info("Starting dashboard...")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Socio-Technical Service Operations System"
    )
    parser.add_argument(
        "--dashboard", 
        action="store_true",
        help="Also start the dashboard"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Simulation speed (0.1=fast, 1.0=realtime)"
    )
    
    args = parser.parse_args()
    
    # Configure
    Config.SIMULATION_SPEED = args.speed
    
    # Start dashboard if requested
    if args.dashboard:
        run_dashboard()
        time.sleep(3)  # Let dashboard start
        
    # Create and run orchestrator
    orchestrator = SystemOrchestrator()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        orchestrator.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start system
    orchestrator.start()
    orchestrator.wait()
    orchestrator.stop()


if __name__ == "__main__":
    main()

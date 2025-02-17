import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta

# Simulation Configuration
@dataclass
class SimConfig:
    NUM_WORKSTATIONS: int = 6
    NUM_SUPPLIERS: int = 3
    BIN_CAPACITY: int = 25
    TOTAL_SIM_TIME: int = 5000
    NUM_RUNS: int = 100
    
    # Workstation-specific parameters
    FAILURE_PROBS: List[float] = None
    WORK_TIME_MEAN: float = 4
    WORK_TIME_STD: float = 1
    FIXING_TIME_MEAN: float = 3
    
    # Quality and accident parameters
    QUALITY_ISSUE_PROB: float = 0.05
    FACILITY_ACCIDENT_PROB: float = 0.0001
    
    # Supply chain parameters
    RESTOCK_DELAY_MEAN: float = 2
    RESTOCK_DELAY_STD: float = 0.5
    
    # Cost parameters (monetary units)
    OPERATING_COST_PER_HOUR: float = 100
    MAINTENANCE_COST_PER_HOUR: float = 200
    PRODUCT_VALUE: float = 50
    QUALITY_FAILURE_COST: float = 25
    
    def __post_init__(self):
        if self.FAILURE_PROBS is None:
            self.FAILURE_PROBS = [0.02, 0.01, 0.05, 0.15, 0.07, 0.06]
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        if self.NUM_WORKSTATIONS <= 0 or self.NUM_SUPPLIERS <= 0:
            raise ValueError("Invalid number of workstations or suppliers")
        if any(p < 0 or p > 1 for p in self.FAILURE_PROBS):
            raise ValueError("Invalid failure probabilities")
        if len(self.FAILURE_PROBS) != self.NUM_WORKSTATIONS:
            raise ValueError("Failure probabilities must match number of workstations")

class MaintenanceSchedule:
    def __init__(self, env, facility):
        self.env = env
        self.facility = facility
        self.maintenance_log = []
        self.schedule = self._generate_schedule()
    
    def _generate_schedule(self):
        """Generate preventive maintenance schedule for each workstation"""
        schedule = {}
        for i in range(self.facility.config.NUM_WORKSTATIONS):
            # Schedule maintenance every 500 time units with some randomness
            interval = random.normalvariate(500, 50)
            schedule[i] = interval
        return schedule
    
    def run(self):
        """Execute maintenance schedule"""
        while True:
            for station_id, interval in self.schedule.items():
                if self.env.now % interval < 1:  # Allow for some flexibility
                    maintenance_time = random.expovariate(1/4)  # Mean 4 time units
                    self.maintenance_log.append({
                        'station_id': station_id,
                        'time': self.env.now,
                        'duration': maintenance_time
                    })
                    self.facility.total_downtime[station_id] += maintenance_time
                    yield self.env.timeout(maintenance_time)
            yield self.env.timeout(1)

class SupplyChain:
    def __init__(self, env, facility):
        self.env = env
        self.facility = facility
        self.stock_levels = [facility.config.BIN_CAPACITY] * facility.config.NUM_WORKSTATIONS
        self.restock_events = []
        
    def monitor_stock(self):
        """Monitor and log stock levels"""
        while True:
            for i in range(self.facility.config.NUM_WORKSTATIONS):
                if self.stock_levels[i] < self.facility.config.BIN_CAPACITY * 0.2:  # 20% threshold
                    self.restock_events.append({
                        'station_id': i,
                        'time': self.env.now,
                        'current_level': self.stock_levels[i]
                    })
            yield self.env.timeout(1)

class ManufacturingFacility:
    def __init__(self, env, config: SimConfig):
        self.env = env
        self.config = config
        
        # Resources
        self.workstations = [simpy.Resource(env, capacity=1) 
                            for _ in range(config.NUM_WORKSTATIONS)]
        self.supplier = simpy.Resource(env, capacity=config.NUM_SUPPLIERS)
        
        # State tracking
        self.bins = [config.BIN_CAPACITY for _ in range(config.NUM_WORKSTATIONS)]
        self.production_count = 0
        self.faulty_count = 0
        self.total_downtime = [0] * config.NUM_WORKSTATIONS
        self.supplier_occupancy = 0
        self.processed_items = [0] * config.NUM_WORKSTATIONS
        
        # Performance metrics
        self.total_fix_time = 0
        self.total_bottleneck_delay = 0
        self.workstation_wait_time = [0] * config.NUM_WORKSTATIONS
        self.cycle_times = [[] for _ in range(config.NUM_WORKSTATIONS)]
        self.quality_issues = []
        self.maintenance_costs = 0
        self.operating_costs = 0
        self.revenue = 0
        
        # Initialize subsystems
        self.maintenance = MaintenanceSchedule(env, self)
        self.supply_chain = SupplyChain(env, self)

    def workstation_process(self, station_id):
        """Simulate workstation operations with enhanced monitoring"""
        while True:
            start_time = self.env.now
            
            if self.bins[station_id] > 0:
                # Process time calculation
                process_time = max(0, random.gauss(
                    self.config.WORK_TIME_MEAN,
                    self.config.WORK_TIME_STD
                ))
                
                yield self.env.timeout(process_time)
                self.bins[station_id] -= 1
                self.processed_items[station_id] += 1
                
                # Equipment failure handling
                if random.random() < self.config.FAILURE_PROBS[station_id]:
                    repair_time = random.expovariate(1 / self.config.FIXING_TIME_MEAN)
                    self.total_fix_time += repair_time
                    self.total_downtime[station_id] += repair_time
                    self.maintenance_costs += repair_time * self.config.MAINTENANCE_COST_PER_HOUR
                    yield self.env.timeout(repair_time)
                
                # Quality control for final station
                if station_id == self.config.NUM_WORKSTATIONS - 1:
                    if random.random() < self.config.QUALITY_ISSUE_PROB:
                        self.faulty_count += 1
                        self.quality_issues.append({
                            'time': self.env.now,
                            'station_id': station_id
                        })
                        self.operating_costs += self.config.QUALITY_FAILURE_COST
                    else:
                        self.production_count += 1
                        self.revenue += self.config.PRODUCT_VALUE
                
                # Record cycle time
                self.cycle_times[station_id].append(self.env.now - start_time)
                
            else:
                yield self.env.timeout(1)
            
            # Update operating costs
            self.operating_costs += self.config.OPERATING_COST_PER_HOUR

    def restocking_process(self):
        """Handle material restocking with enhanced supply chain logic"""
        while True:
            for i in range(self.config.NUM_WORKSTATIONS):
                if self.bins[i] < self.config.BIN_CAPACITY * 0.2:  # 20% threshold
                    with self.supplier.request() as req:
                        yield req
                        
                        delay = abs(random.gauss(
                            self.config.RESTOCK_DELAY_MEAN,
                            self.config.RESTOCK_DELAY_STD
                        ))
                        
                        self.supplier_occupancy += delay
                        yield self.env.timeout(delay)
                        
                        # Partial restock if supply chain issues
                        restock_amount = self.config.BIN_CAPACITY
                        if random.random() < 0.1:  # 10% chance of supply issue
                            restock_amount = int(self.config.BIN_CAPACITY * random.uniform(0.5, 0.8))
                        
                        self.bins[i] = min(self.config.BIN_CAPACITY, 
                                         self.bins[i] + restock_amount)
            
            yield self.env.timeout(1)

    def accident_check(self):
        """Simulate and handle facility-wide accidents"""
        while True:
            if random.random() < self.config.FACILITY_ACCIDENT_PROB:
                downtime = random.randint(5, 50)
                accident_cost = downtime * self.config.MAINTENANCE_COST_PER_HOUR * 2  # Double cost for accidents
                
                for i in range(self.config.NUM_WORKSTATIONS):
                    self.total_downtime[i] += downtime
                
                self.maintenance_costs += accident_cost
                yield self.env.timeout(downtime)
            else:
                yield self.env.timeout(1)

    def analyze_bottlenecks(self):
        """Identify system bottlenecks"""
        utilization = [sum(times)/self.config.TOTAL_SIM_TIME 
                      for times in self.cycle_times]
        bottleneck_station = np.argmax(utilization)
        return {
            'bottleneck_station': bottleneck_station,
            'utilization': utilization,
            'average_cycle_times': [np.mean(times) if times else 0 
                                  for times in self.cycle_times]
        }

def run_simulation(config: SimConfig = None):
    """Run complete manufacturing simulation with multiple iterations"""
    if config is None:
        config = SimConfig()
    
    results = []
    for run in range(config.NUM_RUNS):
        env = simpy.Environment()
        factory = ManufacturingFacility(env, config)
        
        # Start all processes
        for i in range(config.NUM_WORKSTATIONS):
            env.process(factory.workstation_process(i))
        env.process(factory.restocking_process())
        env.process(factory.accident_check())
        env.process(factory.maintenance.run())
        env.process(factory.supply_chain.monitor_stock())
        
        # Run simulation
        env.run(until=config.TOTAL_SIM_TIME)
        
        # Analyze bottlenecks
        bottleneck_analysis = factory.analyze_bottlenecks()
        
        # Collect results
        results.append({
            'run_id': run,
            'final_production': factory.production_count,
            'faulty_products': factory.faulty_count,
            'downtime_per_ws': factory.total_downtime,
            'supplier_occupancy': factory.supplier_occupancy,
            'average_fix_time': factory.total_fix_time / config.NUM_WORKSTATIONS,
            'bottleneck_delay': sum(factory.workstation_wait_time) / config.NUM_WORKSTATIONS,
            'workstation_delays': factory.workstation_wait_time,
            'bottleneck_station': bottleneck_analysis['bottleneck_station'],
            'station_utilization': bottleneck_analysis['utilization'],
            'maintenance_costs': factory.maintenance_costs,
            'operating_costs': factory.operating_costs,
            'revenue': factory.revenue,
            'profit': factory.revenue - (factory.maintenance_costs + factory.operating_costs)
        })
    
    return results

def visualize_results(results):
    """Create comprehensive visualization of simulation results"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # Production metrics
    plt.subplot(2, 3, 1)
    production_data = [r['final_production'] for r in results]
    plt.hist(production_data, bins=20, color='blue', alpha=0.7)
    plt.title("Production Distribution")
    plt.xlabel("Units Produced")
    plt.ylabel("Frequency")
    
    # Financial metrics
    plt.subplot(2, 3, 2)
    profit_data = [r['profit'] for r in results]
    plt.hist(profit_data, bins=20, color='green', alpha=0.7)
    plt.title("Profit Distribution")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    
    # Workstation utilization
    plt.subplot(2, 3, 3)
    utilization_data = np.array([r['station_utilization'] for r in results])
    plt.boxplot(utilization_data)
    plt.title("Workstation Utilization")
    plt.xlabel("Workstation")
    plt.ylabel("Utilization Rate")
    
    # Downtime analysis
    plt.subplot(2, 3, 4)
    downtime_data = np.array([r['downtime_per_ws'] for r in results])
    plt.boxplot(downtime_data)
    plt.title("Downtime per Workstation")
    plt.xlabel("Workstation")
    plt.ylabel("Downtime")
    
    # Quality metrics
    plt.subplot(2, 3, 5)
    quality_rate = [(r['final_production']/(r['final_production'] + r['faulty_products']))*100 
                   for r in results]
    plt.hist(quality_rate, bins=20, color='red', alpha=0.7)
    plt.title("Quality Rate Distribution")
    plt.xlabel("Quality Rate (%)")
    plt.ylabel("Frequency")
    
    # Bottleneck analysis
    plt.subplot(2, 3, 6)
    bottleneck_counts = np.bincount([r['bottleneck_station'] for r in results], 
                                  minlength=SimConfig.NUM_WORKSTATIONS)
    plt.bar(range(SimConfig.NUM_WORKSTATIONS), bottleneck_counts)
    plt.title("Bottleneck Frequency")
    plt.xlabel("Workstation")
    plt.ylabel("Times Identified as Bottleneck")
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    # Run simulation with default configuration
    results = run_simulation()
    
    # Create visualizations
    fig = visualize_results(results)
    
    # Calculate and display summary statistics
    summary_stats = {
        'avg_production': np.mean([r['final_production'] for r in results]),
        'avg_profit': np.mean([r['profit'] for r in results]),
        'avg_quality_rate': np.mean([(r['final_production']/(r['final_production'] + r['faulty_products']))*100 
                                    for r in results]),
        'most_common_bottleneck': np.argmax([r['bottleneck_station'] for r in results]),
        'avg_maintenance_cost': np.mean([r['maintenance_costs'] for r in results])
    }
    
    print("\nSimulation Summary:")
    print(f"Average Production: {summary_stats['avg_production']:.2f} units")
    print(f"Average Profit: ${summary_stats['avg_profit']:.2f}")
    print(f"Average Quality Rate: {summary_stats['avg_quality_rate']:.2f}%")
    print(f"Most Common Bottleneck: Workstation {summary_stats['most_common_bottleneck']}")
    print(f"Average Maintenance Cost: ${summary_stats['avg_maintenance_cost']:.2f}")
    
    plt.show()

if __name__ == "__main__":
    main()
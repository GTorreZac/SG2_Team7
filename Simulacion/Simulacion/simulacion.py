import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

NUM_WORKSTATIONS = 6
NUM_SUPPLIERS = 3
BIN_CAPACITY = 25
TOTAL_SIM_TIME = 5000
NUM_RUNS = 100

FAILURE_PROBS = [0.02, 0.01, 0.05, 0.15, 0.07, 0.06]
FIXING_TIME_MEAN = 3  
WORK_TIME_MEAN = 4  
QUALITY_ISSUE_PROB = 0.05
FACILITY_ACCIDENT_PROB = 0.0001  

RESTOCK_DELAY_MEAN = 2  

class ManufacturingFacility:
    def __init__(self, env):
        self.env = env
        self.workstations = [simpy.Resource(env, capacity=1) for _ in range(NUM_WORKSTATIONS)]
        self.supplier = simpy.Resource(env, capacity=NUM_SUPPLIERS)
        self.bins = [BIN_CAPACITY for _ in range(NUM_WORKSTATIONS)]
        self.production_count = 0
        self.faulty_count = 0
        self.total_downtime = [0] * NUM_WORKSTATIONS
        self.supplier_occupancy = 0
        self.processed_items = [0] * NUM_WORKSTATIONS
        self.total_fix_time = 0
        self.total_bottleneck_delay = 0
        self.workstation_wait_time = [0] * NUM_WORKSTATIONS  # Track bottleneck delays

    def workstation_process(self, station_id):
        while True:
            start_wait = self.env.now  # Track when waiting starts
            if self.bins[station_id] > 0:
                self.workstation_wait_time[station_id] += self.env.now - start_wait  # Log waiting time

                yield self.env.timeout(max(0, random.gauss(WORK_TIME_MEAN, 1)))
                self.bins[station_id] -= 1
                self.processed_items[station_id] += 1

                if random.random() < FAILURE_PROBS[station_id]:
                    repair_time = random.expovariate(1 / FIXING_TIME_MEAN)
                    self.total_fix_time += repair_time
                    self.total_downtime[station_id] += repair_time
                    yield self.env.timeout(repair_time)

                if station_id == NUM_WORKSTATIONS - 1:
                    if random.random() < QUALITY_ISSUE_PROB:
                        self.faulty_count += 1
                    else:
                        self.production_count += 1
            else:
                yield self.env.timeout(1)  

    def restocking_process(self):
        while True:
            for i in range(NUM_WORKSTATIONS):
                if self.bins[i] == 0:
                    with self.supplier.request() as req:
                        yield req
                        delay = abs(random.gauss(RESTOCK_DELAY_MEAN, 0.5))
                        self.supplier_occupancy += delay
                        yield self.env.timeout(delay)
                        self.bins[i] = BIN_CAPACITY
            yield self.env.timeout(1)

    def accident_check(self):
        while True:
            if random.random() < FACILITY_ACCIDENT_PROB:
                downtime = random.randint(5, 50)
                for i in range(NUM_WORKSTATIONS):
                    self.total_downtime[i] += downtime
                yield self.env.timeout(downtime)
            else:
                yield self.env.timeout(1)


def run_simulation():
    results = []
    for _ in range(NUM_RUNS):
        env = simpy.Environment()
        factory = ManufacturingFacility(env)
        
        for i in range(NUM_WORKSTATIONS):
            env.process(factory.workstation_process(i))
        env.process(factory.restocking_process())
        env.process(factory.accident_check())
        
        env.run(until=TOTAL_SIM_TIME)
        
        results.append({
            'Final Production': factory.production_count,
            'Faulty Products': factory.faulty_count,
            'Downtime per WS': factory.total_downtime,
            'Supplier Occupancy': factory.supplier_occupancy,
            'Average Fix Time': factory.total_fix_time / NUM_WORKSTATIONS,
            'Bottleneck Delay': sum(factory.workstation_wait_time) / NUM_WORKSTATIONS,
            'Workstation Delays': factory.workstation_wait_time
        })
    return results


# Run and analyze the results
simulation_results = run_simulation()

# Extract relevant data
final_production = [r['Final Production'] for r in simulation_results]
faulty_products = [r['Faulty Products'] for r in simulation_results]
downtime = np.array([r['Downtime per WS'] for r in simulation_results])
supplier_occupancy = [r['Supplier Occupancy'] for r in simulation_results]
bottleneck_delay = [r['Bottleneck Delay'] for r in simulation_results]

# Visualization
plt.figure(figsize=(12, 6))

# Production histogram
plt.subplot(2, 3, 1)
plt.hist(final_production, bins=20, color='blue', alpha=0.7)
plt.title("Final Production Count")

# Faulty products histogram
plt.subplot(2, 3, 2)
plt.hist(faulty_products, bins=20, color='red', alpha=0.7)
plt.title("Faulty Products per Run")

# Downtime per workstation
plt.subplot(2, 3, 3)
plt.boxplot(downtime.T)
plt.title("Downtime per Workstation")

# Supplier occupancy
plt.subplot(2, 3, 4)
plt.hist(supplier_occupancy, bins=20, color='green', alpha=0.7)
plt.title("Supplier Occupancy per Run")

# Bottleneck delay
plt.subplot(2, 3, 5)
plt.hist(bottleneck_delay, bins=20, color='purple', alpha=0.7)
plt.title("Bottleneck Delays per Run")

plt.tight_layout()
plt.show()

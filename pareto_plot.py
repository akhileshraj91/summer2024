import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_time(log_path):
    """Parse execution time from the log file."""
    with open(log_path, 'r') as f:
        content = f.read()
        match = re.search(r'real\s+(\d+)m(\d+\.\d+)s', content)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        else:
            return None

def get_energy(energy_path):
    """Get total energy consumed from energy.csv."""
    df = pd.read_csv(energy_path)
    # Assuming scope 1 is the total energy scope
    scope1_values = df[df['scope'] == 1]['value']
    if not scope1_values.empty:
        last_value = scope1_values.iloc[-1]
        first_value = scope1_values.iloc[0]
        return last_value - first_value
    return None

def get_pcap_mean(pcap_path):
    """Get the mean PCAP value from PCAP_file.csv."""
    df = pd.read_csv(pcap_path)
    return df['value'].mean()

def pareto_front(points):
    """Compute the Pareto front for minimization of both objectives."""
    # Sort points by execution time ascending
    points.sort(key=lambda x: x[0])
    front = []
    min_energy = float('inf')
    for t, e in points:
        if e < min_energy:
            min_energy = e
            front.append((t, e))
    return front

# Directory containing the data
data_dir = '/Users/akhileshraj/Documents/summer2024/main_codes/experiment_data/plotting_data/ones-stream-full'

# Collect all points
raw_points = []
for item in os.listdir(data_dir):
    if item.startswith('compressed_iteration_') and os.path.isdir(os.path.join(data_dir, item)):
        log_path = os.path.join(data_dir, item, 'ones-stream-full_output.log')
        energy_path = os.path.join(data_dir, item, 'energy.csv')
        pcap_path = os.path.join(data_dir, item, 'PCAP_file.csv')
        if os.path.exists(log_path) and os.path.exists(energy_path) and os.path.exists(pcap_path):
            time_s = parse_time(log_path)
            energy_j = get_energy(energy_path)
            pcap_mean = get_pcap_mean(pcap_path)
            if time_s is not None and energy_j is not None and pcap_mean is not None:
                raw_points.append((time_s, energy_j, pcap_mean))

# Group by PCAP mean and average
from collections import defaultdict
import numpy as np
pcap_groups = defaultdict(list)
for t, e, p in raw_points:
    pcap_groups[p].append((t, e))

# Average within each PCAP group and compute std
points = []
errors = []
for p, group in pcap_groups.items():
    ts = [t for t, e in group]
    es = [e for t, e in group]
    avg_t = np.mean(ts)
    avg_e = np.mean(es) / 1000  # Energy in kJ
    std_e = np.std(es) / 1000   # Std in kJ
    points.append((avg_t, avg_e))
    errors.append(std_e)

# Sort points by execution time
points.sort(key=lambda x: x[0])

# Compute Pareto front on the averaged points
pareto = pareto_front(points.copy())

# Plot
plt.figure(figsize=(10, 6))
ts = [p[0] for p in points]
es = [p[1] for p in points]
plt.errorbar(ts, es, yerr=errors, fmt='o-', capsize=5, label='Mean with Std Dev')
if pareto:
    plt.plot([p[0] for p in pareto], [p[1] for p in pareto], 'r--', label='Pareto front')
plt.xlabel('Execution Time (seconds)')
plt.ylabel('Energy Consumption (kJ)')
plt.title('Pareto Front: Energy vs Execution Time for ones-stream-full')
plt.legend()
plt.grid(True)
plt.savefig('pareto_front_ones_stream_full.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented out to avoid hanging in terminal
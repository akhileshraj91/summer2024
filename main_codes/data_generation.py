import time
import numpy as np
import sys
import pandas as pd
import nrm
import csv
import subprocess
import os
import tarfile
import random
from datetime import datetime
import psutil
import argparse

# Set environment variables for OpenMP
os.environ['OMP_PLACES'] = 'threads'
os.environ['OMP_PROC_BIND'] = 'true'
os.environ['OMP_NUM_THREADS'] = str(psutil.cpu_count() - 1)


ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
# ACTIONS = [95.0]



# APPLICATIONS = ['ones-npb-ep', 'ones-npb-cg', 'ones-npb-is', 'ones-npb-bt', 'ones-npb-mg', 'ones-npb-ft', 'hpccg']
APPLICATIONS = []
# APPLICATIONS = ['ones-stream-full', 'ones-stream-add', 'ones-stream-copy', 'ones-stream-triad', 'ones-stream-scale', 'ones-npb-ep', 'ones-npb-cg', 'ones-npb-is', 'ones-npb-bt', 'ones-npb-mg', 'ones-npb-ft']

parser = argparse.ArgumentParser(description="Add new applications to the list")
parser.add_argument('-a', '--application', nargs='+', help='List of applications to append')
parser.add_argument('-e', '--experiment', default='random', help='Choice of experiment - values random and static')

args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help()
    
if args.application:
    APPLICATIONS.extend(args.application)

print(APPLICATIONS)


client = nrm.Client()
actuators = client.list_actuators()


def compress_files(iteration):
    tar_file = EXP_DIR+f'/compressed_iteration_{iteration}.tar'
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, dirs, files in os.walk(EXP_DIR):
            for file in files:
                if file.endswith('.csv') or file.endswith('.yaml') or file.endswith('.log'):
                    file_path = os.path.join(root, file)  # <-- use root here!
                    tarf.add(file_path, arcname=os.path.relpath(file_path, EXP_DIR))
                    os.remove(file_path)

    print(f'Compressed files into {tar_file}')

def get_pid(application):
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    
    pids = []
    # Iterate over each line of the result
    for line in result.stdout.strip().split('\n'):
        if application in line and 'grep' not in line:
            parts = line.split()
            pid = parts[1]  # PID is usually the second element
            pids.append(pid)
    
    return pids


def get_random_walk_weights(current_action, actions, sigma=2.0, center_bias=0.0):
    """Generate weights for random walk - higher probability for nearby actions
    
    Args:
        current_action: Current PCAP value
        actions: List of possible actions
        sigma: Controls random walk tightness (lower = tighter)
        center_bias: Controls preference for middle actions (0 = no bias, higher = stronger bias toward center)
    """
    current_idx = actions.index(current_action)
    middle_idx = len(actions) // 2
    weights = []
    
    for i, action in enumerate(actions):
        # Random walk component: favor nearby actions
        distance = abs(i - current_idx)
        walk_weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        # Center bias component: favor middle actions
        if center_bias > 0:
            distance_from_center = abs(i - middle_idx)
            center_weight = np.exp(-(distance_from_center ** 2) / (2 * center_bias ** 2))
            weight = walk_weight * center_weight
        else:
            weight = walk_weight
            
        weights.append(weight)
    
    # Normalize weights
    total = sum(weights)
    return [w / total for w in weights]

def experiment_for(APPLICATION, EXP_DIR, ACTION=None):
    if "stream" in APPLICATION:
        PROBLEM_SIZE = 33554432
        ITERATIONS = 10000
    elif "npb" in APPLICATION:
        PROBLEM_SIZE = 26
        ITERATIONS = 10000
    with open(f'{EXP_DIR}/{APPLICATION}_output.log','w') as log_file, open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, open(f'{EXP_DIR}/PCAP_file.csv', mode='w', newline='') as PCAP_file, open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:
        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        PCAP_writer = csv.writer(PCAP_file)
        power_writer.writerow(['time', 'scope', 'value'])
        progress_writer.writerow(['time', 'value'])
        energy_writer.writerow(['time', 'scope', 'value'])
        PCAP_writer.writerow(['time', 'actuator', 'value'])
        papi_writer.writerow(['time', 'scope', 'value'])

        def cb(*args):
            print(args)
            (sensor, time, scope, value) = args
            scope = scope.get_uuid()
            sensor = sensor.decode("UTF-8")
            timestamp = time/1e9
            if sensor == "nrm.benchmarks.progress":
                progress_writer.writerow([timestamp, value])
            elif sensor == "nrm.geopm.CPU_POWER":
                power_writer.writerow([timestamp, scope[-1], value])
            elif sensor == "nrm.geopm.CPU_ENERGY":
                energy_writer.writerow([timestamp, scope[-1], value])
            elif "PAPI" in sensor:
                papi_writer.writerow([timestamp, sensor, value])


        client.set_event_listener(cb)
        client.start_event_listener("") 
        if "solvers" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} poor 0 {ITERATIONS}'],
                stdout=log_file,
                stderr=log_file
            )
        elif "phases" in APPLICATION:    
            print(f"Starting Execution of phases {APPLICATION, PROBLEM_SIZE, ITERATIONS}")
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} 5 500'],
                stdout=log_file,
                stderr=log_file
            )
        elif "ones-npb-ft" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 500'],
                stdout=log_file,
                stderr=log_file
            )
        elif "ones-npb-mg" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                stdout=log_file,
                stderr=log_file
            )
        elif "ones-npb-bt" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                stdout=log_file,
                stderr=log_file
            )
        elif "ones-npb-cg" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'],
                stdout=log_file,
                stderr=log_file
            )
        elif "ones-npb-is" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 26 1000'],
                stdout=log_file,
                stderr=log_file
            )
        elif "hpccg" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time OMP_NUM_THREADS=96 OMP_PLACES=threads nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- /home/cc/dependencies/HPCCG-tasking/hpccg-omp-task-clang++ 432 432 432 100 4 100'],
                stdout=log_file,
                stderr=log_file
            )
        else:
            process = subprocess.Popen(
                ['bash', '-c', 'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {} {} {}'.format(APPLICATION, PROBLEM_SIZE, ITERATIONS)],
                stdout=log_file,
                stderr=log_file
            )
        time.sleep(0.5)
   
        
        last_pcap_change = 0
        # current_pcap = random.choice(ACTIONS)  # Initialize with random action
        const_PCAP = 89.0  # Start from edge for testing
        while True:
            current_time = time.time()
            if current_time - last_pcap_change >= 8:
                if not ACTION: 
                    # Random walk: choose next action with weights favoring nearby values
                    # Increase center_bias (e.g., 3.0) to prefer middle actions more
                    weights = get_random_walk_weights(const_PCAP, ACTIONS, sigma=2, center_bias=0)
                    PCAP = random.choices(ACTIONS, weights=weights)[0]
                    # current_pcap = PCAP  # Update current action
                else:
                    PCAP = ACTION
                print(PCAP)
                client.actuate(actuators[0], PCAP)
                PCAP_time = time.time()
                PCAP_writer.writerow([PCAP_time, actuators[0], PCAP])
                last_pcap_change = current_time
            
            time.sleep(0.1)  # Short sleep to prevent busy-waiting
            if process.poll() is not None:  
                print("Process has completed.")
                break
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    compress_files(current_time)
    print("----------------------------------")




   

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    repeat = 5
    ACTION = None
    if args.experiment == 'random':
        for REPEAT in range(repeat):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>{REPEAT}")
            for APPLICATION in APPLICATIONS:
                experiment = 'training_data'
                EXP_DIR = f'{current_dir}/experiment_data/{experiment}/{APPLICATION}'
                if os.path.exists(EXP_DIR):
                    print(f"Directories {EXP_DIR} exist")
                else:
                    os.makedirs(EXP_DIR)
                    print(f"Directory {EXP_DIR} created") 
                experiment_for(APPLICATION, EXP_DIR, ACTION=ACTION)
                time.sleep(1)
    elif args.experiment == 'static':
        for ACTION in ACTIONS:
            for REPEAT in range(repeat):
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>{REPEAT}")
                for APPLICATION in APPLICATIONS:
                    experiment = 'plotting_data'
                    EXP_DIR = f'{current_dir}/experiment_data/{experiment}/{APPLICATION}'
                    if os.path.exists(EXP_DIR):
                        print(f"Directories {EXP_DIR} exist")
                    else:
                        os.makedirs(EXP_DIR)
                        print(f"Directory {EXP_DIR} created") 
                    experiment_for(APPLICATION, EXP_DIR, ACTION=ACTION)
                    time.sleep(1)



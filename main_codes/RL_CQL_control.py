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
import torch

class FCNetwork(torch.nn.Module):
  def __init__(self, layers=[20,20]):
    super(FCNetwork, self).__init__()
    # self.all_observations = torch.tensor(stack_observations(env), dtype=torch.float32)
    dim_input = 5
    dim_output = 16
    net_layers = []

    dim = dim_input
    for i, layer_size in enumerate(layers):
      net_layers.append(torch.nn.Linear(dim, layer_size))
      net_layers.append(torch.nn.ReLU())
      dim = layer_size
    net_layers.append(torch.nn.Linear(dim, dim_output))
    self.layers = net_layers
    self.network = torch.nn.Sequential(*net_layers)

  def forward(self, states):
    # observations = torch.index_select(self.all_observations, 0, states)
    states_tensor = torch.tensor(states, dtype=torch.float32)  # Ensure the correct dtype
    return self.network(states_tensor)

  def print_weights(self):
    for name, param in self.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data.numpy()}")
            
model = FCNetwork(layers=[20, 20])

i = 0
APPLICATIONS = ['ones-stream-full', 'ones-stream-triad', 'ones-stream-add', 'ones-stream-copy', 'ones-stream-scale']
policy_folder = '/home/cc/summer2024/main_codes/results/'  # Default policy file
# policy_file = os.path.join(policy_folder,'BCQ_SYS_0_20240929_183736.pt')
while i < len(sys.argv):
    if sys.argv[i] == '--application':
        APPLICATION = sys.argv[i+1]
        i += 1
    elif sys.argv[i] == '--policy':
        policy_name = sys.argv[i+1]  # Update policy file from argument
        policy_file = os.path.join(policy_folder, policy_name)
        i += 1
    i +=1



client = nrm.Client()
actuators = client.list_actuators()
ACTIONS = actuators[0].list_choices()
# policy = torch.load(policy_file)  # Load policy from the specified file
model.load_state_dict(torch.load(policy_file))
model.eval()
# max_values = {'PAPI_L3_TCA': 2519860487.0, 'PAPI_TOT_INS': 108949748763.0, 'PAPI_TOT_CYC': 297655869900.0, 'PAPI_RES_STL': 259228596139.0, 'PAPI_L3_TCM': 2370673711.0}

def compress_files(iteration):
    tar_file = EXP_DIR+f'/compressed_iteration_{iteration}.tar'
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, dirs, files in os.walk(EXP_DIR):
            for file in files:
                if file.endswith('.csv') or file.endswith('.yaml'):
                    file_path = os.path.join(EXP_DIR, file)
                    # rel_path = os.path.relpath(file_path, EXP_DIR)
                    tarf.add(file_path, arcname=os.path.basename(file_path))
                    # tarf.add(os.path.join(root, file), os.path.relpath(os.path.join(root, file), EXP_DIR))
                    os.remove(file_path)

    print(f'Compressed files into {tar_file}')
    
def calculate_power_with_wraparound(current, previous, time_diff, wraparound_value=262143.328850):
    diff = current - previous
    if diff < 0:  # Wraparound detected
        diff = (wraparound_value - previous) + current
    return diff / time_diff

def compute_power(E0,E1):
    power = {}

    power['geopm_power_0'] =[
            calculate_power_with_wraparound(
                E0[i][1],
                E0[i-1][1],
                E0[i][0] - E0[i-1][0]
            ) for i in range(1, len(E0))
        ]

    power['geopm_power_1'] = [
            calculate_power_with_wraparound(
                E1[i][1],
                E1[i-1][1],
                E1[i][0] - E1[i-1][0]
            ) for i in range(1, len(E1))
        ]

    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]
    # print(geopm_power_0,geopm_power_1)
    average_power = [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0, geopm_power_1)]
    return np.mean(average_power)

def measure_progress(progress_data):
    # first_sensor_point = progress_data[0][0]
    frequency_values = [
        progress_data[k][1] / (progress_data[k][0] - progress_data[k-1][0]) for k in range(1, len(progress_data))
    ]
    frequency_values = [0] + frequency_values  # Prepend a 0 for the first index
    progress_data = np.nanmedian(frequency_values)
    return progress_data

def collect_papi(PAPI_data):
    PAPI = {}
    # print(PAPI_data)
    for scope in PAPI_data.keys():
        print(scope)
        if "PAPI_L3_TCA" in scope:
            L3_TCA = np.mean([(PAPI_data[scope][k+1][1] - PAPI_data[scope][k][1]) for k, cum in enumerate(PAPI_data[scope][:-1])])  # Extracted value
        if "PAPI_TOT_INS" in scope:
            TOT_INS = np.mean([(PAPI_data[scope][k+1][1] - PAPI_data[scope][k][1]) for k, cum in enumerate(PAPI_data[scope][:-1])])  # Extracted value
        if 'PAPI_TOT_CYC' in scope:
            TOT_CYC = np.mean([(PAPI_data[scope][k+1][1] - PAPI_data[scope][k][1]) for k, cum in enumerate(PAPI_data[scope][:-1])])  # Extracted value
        if 'PAPI_RES_STL' in scope:
            RES_STL = np.mean([(PAPI_data[scope][k+1][1] - PAPI_data[scope][k][1]) for k, cum in enumerate(PAPI_data[scope][:-1])])  # Extracted value
        if 'PAPI_L3_TCM' in scope:
            L3_TCM = np.mean([(PAPI_data[scope][k+1][1] - PAPI_data[scope][k][1]) for k, cum in enumerate(PAPI_data[scope][:-1])])  # Extracted value
    TOT_INS_PER_CYC = TOT_INS/TOT_CYC
    L3_TCM_PER_TCA = L3_TCM/L3_TCA
    TOT_STL_PER_CYC = RES_STL/TOT_CYC
    return [TOT_INS_PER_CYC, L3_TCM_PER_TCA, TOT_STL_PER_CYC]
    
def process_callback(states):
    progress = measure_progress(states['progress'])
    measured_power = compute_power(states['energy_0'],states['energy_1'])
    PAPI = collect_papi(states)
    # Concatenate the progress, measured_power, and PAPI lists
    combined_data = [progress, measured_power] + PAPI
    return combined_data

def initialize_state_dict():
    global state_dict
    state_dict = {}
    state_dict['progress'] = []
    state_dict['energy_0'] = []
    state_dict['energy_1'] = []
    state_dict['PAPI_L3_TCA'] = []
    state_dict['PAPI_TOT_INS'] = []
    state_dict['PAPI_TOT_CYC'] = []
    state_dict['PAPI_RES_STL'] = []
    state_dict['PAPI_L3_TCM'] = []
    return state_dict

reference_lib = {}
reference_lib['progress'] = []
reference_lib['energy_0'] = []
reference_lib['energy_1'] = []
reference_lib['PAPI_L3_TCA'] = []
reference_lib['PAPI_TOT_INS'] = []
reference_lib['PAPI_TOT_CYC'] = []
reference_lib['PAPI_RES_STL'] = []
reference_lib['PAPI_L3_TCM'] = []
    
def experiment_for(APPLICATION, EXP_DIR):
    global state_dict
    state_dict = initialize_state_dict() 
    if "stream" in APPLICATION: 
        PROBLEM_SIZE = 33554432
        ITERATIONS = 100000
    elif "solvers" in APPLICATION:
        PROBLEM_SIZE = 10000
        ITERATIONS = 10000
    elif "ep" in APPLICATION:
        PROBLEM_SIZE = 22
        ITERATIONS = 10000
    with open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, open(f'{EXP_DIR}/PCAP_file.csv', mode='w', newline='') as PCAP_file, open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:
        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        PCAP_writer = csv.writer(PCAP_file)
        # Write headers if files are empty
        power_writer.writerow(['time', 'scope', 'value'])
        progress_writer.writerow(['time', 'value'])
        energy_writer.writerow(['time', 'scope', 'value'])
        PCAP_writer.writerow(['time', 'actuator', 'value'])
        papi_writer.writerow(['time', 'scope', 'value'])

        def cb(*args):
            global state_dict
            # print(args)
            (sensor, time, scope, value) = args
            scope = scope.get_uuid()
            sensor = sensor.decode("UTF-8")
            timestamp = time/1e9
            # print(f"----------{state_dict}---------")
            if sensor == "nrm.benchmarks.progress":
                progress_writer.writerow([timestamp, value])
                state_dict["progress"].append([timestamp,value])
                # print("1")
            elif sensor == "nrm.geopm.CPU_POWER":
                power_writer.writerow([timestamp, scope[-1], value])
                # print("2")
            elif sensor == "nrm.geopm.CPU_ENERGY":
                energy_writer.writerow([timestamp, scope[-1], value])
                state_dict[f"energy_{scope[-1]}"].append((timestamp,value))
                # print("---3-----")
            elif "PAPI" in sensor:
                scope_parts = sensor.split('.')
                papi_writer.writerow([timestamp, sensor, value])
                state_dict[scope_parts[3]].append((timestamp,value))
            # print(f"------4-----------")

        client.set_event_listener(cb)
        client.start_event_listener("") 
        if "solvers" in APPLICATION:
            process = subprocess.Popen(['nrm-papiwrapper', '-i', '-e', 'PAPI_L3_TCA', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'{APPLICATION}', f'{PROBLEM_SIZE}', 'poor', '0', f'{ITERATIONS}'])
        else:    
            process = subprocess.Popen(['nrm-papiwrapper', '-i', '-e', 'PAPI_L3_TCA', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'{APPLICATION}', f'{PROBLEM_SIZE}', f'{ITERATIONS}'])


        last_pcap_change = 0
        while True:
            current_time = time.time()
            if current_time - last_pcap_change >= 2:
                # PCAP = random.choice(ACTIONS)
                # print(state_dict)
                if 'state_dict' in globals() and state_dict and state_dict != reference_lib:                    
                    state = process_callback(state_dict)
                    print(state)
                    state = np.array(state)
                    OUT = model(np.array(state))
                    argmax = np.argmax(OUT.detach().numpy(), axis=-1)
                    PCAP = ACTIONS[argmax]
                    # PCAP = min(ACTIONS, key=lambda x: abs(x-PCAP))
                else: 
                    PCAP = 78.0
                print(PCAP)
                client.actuate(actuators[0], PCAP)
                PCAP_time = time.time()
                PCAP_writer.writerow([PCAP_time, actuators[0], PCAP])
                last_pcap_change = current_time
                state_dict = initialize_state_dict()
            
            time.sleep(0.1)  # Short sleep to prevent busy-waiting
            if process.poll() is not None:  
                print("Process has completed.")
                break
            if process.poll() is not None:  
                print("Process has completed.")
                break
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    compress_files(current_time)
    print("----------------------------------")




   

if __name__ == "__main__":
    # Get the current file path
    current_file_path = os.path.abspath(__file__)

    # Get the directory containing the current file
    current_dir = os.path.dirname(current_file_path)


    for APPLICATION in APPLICATIONS:
        experiment = 'Control'
        EXP_DIR = f'{current_dir}/experiment_data/{experiment}/{APPLICATION}'
        if os.path.exists(EXP_DIR):
            print(f"Directories {EXP_DIR} exist")
        else:
            os.makedirs(EXP_DIR)
            print(f"Directory {EXP_DIR} created") 
        experiment_for(APPLICATION, EXP_DIR)




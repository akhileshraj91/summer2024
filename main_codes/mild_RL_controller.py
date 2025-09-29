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
import argparse
my_env = os.environ.copy()
my_env["OMP_NUM_THREADS"] = "94"

# ----------------------------
# Model definition (lean forward)
# ----------------------------
class FCNetwork(torch.nn.Module):
    def __init__(self, layers=[10, 10]):
        super().__init__()
        dim_input, dim_output = 7, 32
        layers_ = []
        d = dim_input
        for h in layers:
            layers_.append(torch.nn.Linear(d, h))
            layers_.append(torch.nn.ReLU())
            d = h
        layers_.append(torch.nn.Linear(d, dim_output))
        self.network = torch.nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x as a torch tensor of shape (B, 7), dtype float32, on same device as model
        return self.network(x)

    def print_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.cpu().numpy()}")


# ----------------------------
# Argparse
# ----------------------------
parser = argparse.ArgumentParser(description="Evaluate applications with trained policy")
parser.add_argument('-a', '--application', nargs='+', help='List of applications to evaluate')
parser.add_argument('-p', '--policy', required=True,
                    help='Relative path of the trained model under the policy folder')
def parse_array(s):
    """Parse string like '[0.5,0.5]' into a list of floats"""
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]  # Remove brackets
    return [float(x.strip()) for x in s.split(',')]

parser.add_argument('-r', '--preference', default=[0.5,0.5], type=parse_array, help='Preference needed for the execution (format: [0.5,0.5])')
args = parser.parse_args()

APPLICATIONS = []
if args.application:
    APPLICATIONS.extend(args.application)
if len(sys.argv) == 1:
    parser.print_help()

# ----------------------------
# Device & runtime setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keep CPU footprint polite; prevent oversubscription
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ----------------------------
# Load model politely (once)
# ----------------------------
policy_folder = '/home/cc/summer2024/main_codes/'
policy_file = os.path.join(policy_folder, args.policy)

model = FCNetwork(layers=[20, 20]).to(device)
# PyTorch < 2.0 does not support weights_only; try it, then fall back
try:
    state = torch.load(policy_file, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(policy_file, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

# Optional: TorchScript for faster inference/load (pre-export offline if you like)
# scripted = torch.jit.script(model)
# torch.jit.save(scripted, os.path.join(policy_folder, "scripted_model.pt"))
# model = torch.jit.load(os.path.join(policy_folder, "scripted_model.pt"), map_location=device).eval()

# Global inference preference vector (tweak as needed)
# pref_np = np.array([0.17, 0.83], dtype=np.float32)    # shape (2,)
pref_np = np.array(args.preference, dtype=np.float32)
pref_t = torch.from_numpy(pref_np).to(device)         # tensor (2,)

# ----------------------------
# NRM / actions
# ----------------------------
client = nrm.Client()
actuators = client.list_actuators()
ACTIONS = actuators[0].list_choices()   # expected 16 values


# ----------------------------
# Utilities
# ----------------------------
def compress_files(iteration, preference):
    tar_file = EXP_DIR + f'/compressed_iteration_{iteration}_{preference}.tar'
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, dirs, files in os.walk(EXP_DIR):
            if root == EXP_DIR:
                for file in files:
                    if file.endswith('.csv') or file.endswith('.yaml') or file.endswith('.log'):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            tarf.add(file_path, arcname=os.path.basename(file_path))
                            os.remove(file_path)
                        else:
                            print(f"File {file_path} does not exist, skipping...")
    print(f'Compressed files into {tar_file}')


def calculate_power_with_wraparound(current, previous, time_diff, wraparound_value=262143.328850):
    diff = current - previous
    if diff < 0:  # Wraparound detected
        diff = (wraparound_value - previous) + current
    return diff / time_diff


def compute_power(E0, E1):
    power = {}
    power['geopm_power_0'] = [
        calculate_power_with_wraparound(
            E0[i][1],
            E0[i - 1][1],
            E0[i][0] - E0[i - 1][0]
        ) for i in range(1, len(E0))
    ]
    power['geopm_power_1'] = [
        calculate_power_with_wraparound(
            E1[i][1],
            E1[i - 1][1],
            E1[i][0] - E1[i - 1][0]
        ) for i in range(1, len(E1))
    ]

    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]
    average_power = [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0, geopm_power_1)]
    return np.mean(average_power)


def measure_progress(progress_data):
    frequency_values = [
        progress_data[k][1] / (progress_data[k][0] - progress_data[k - 1][0]) for k in range(1, len(progress_data))
    ]
    frequency_values = [0] + frequency_values  # Prepend a 0 for the first index
    return np.nanmedian(frequency_values)


def collect_papi(PAPI_data):
    PAPI = {}
    for scope in PAPI_data.keys():
        if "PAPI_L3_TCA" in scope:
            L3_TCA = np.mean([(PAPI_data[scope][k + 1][1] - PAPI_data[scope][k][1]) for k, _ in enumerate(PAPI_data[scope][:-1])])
        if "PAPI_TOT_INS" in scope:
            TOT_INS = np.mean([(PAPI_data[scope][k + 1][1] - PAPI_data[scope][k][1]) for k, _ in enumerate(PAPI_data[scope][:-1])])
        if 'PAPI_TOT_CYC' in scope:
            TOT_CYC = np.mean([(PAPI_data[scope][k + 1][1] - PAPI_data[scope][k][1]) for k, _ in enumerate(PAPI_data[scope][:-1])])
        if 'PAPI_RES_STL' in scope:
            RES_STL = np.mean([(PAPI_data[scope][k + 1][1] - PAPI_data[scope][k][1]) for k, _ in enumerate(PAPI_data[scope][:-1])])
        if 'PAPI_L3_TCM' in scope:
            L3_TCM = np.mean([(PAPI_data[scope][k + 1][1] - PAPI_data[scope][k][1]) for k, _ in enumerate(PAPI_data[scope][:-1])])
    TOT_INS_PER_CYC = TOT_INS / TOT_CYC
    L3_TCM_PER_TCA = L3_TCM / L3_TCA
    TOT_STL_PER_CYC = RES_STL / TOT_CYC
    return [TOT_INS_PER_CYC, L3_TCM_PER_TCA, TOT_STL_PER_CYC]


def measure_power(P0, P1):
    power = {}
    power['geopm_power_0'] = [P0[i][1] for i in range(1, len(P0))]
    power['geopm_power_1'] = [P1[i][1] for i in range(1, len(P1))]
    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]
    average_power = [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0, geopm_power_1)]
    return np.mean(average_power)


def process_callback(states):
    progress = measure_progress(states['progress'])
    measured_power = measure_power(states['measured_power_0'], states['measured_power_1'])
    PAPI = collect_papi(states)
    return [progress, measured_power] + PAPI  # 5 floats


def initialize_state_dict():
    sd = {}
    sd['progress'] = []
    sd['energy_0'] = []
    sd['energy_1'] = []
    sd['PAPI_L3_TCA'] = []
    sd['PAPI_TOT_INS'] = []
    sd['PAPI_TOT_CYC'] = []
    sd['PAPI_RES_STL'] = []
    sd['PAPI_L3_TCM'] = []
    sd['measured_power_0'] = []
    sd['measured_power_1'] = []
    return sd


# reference (empty) structure for quick-compare
reference_lib = initialize_state_dict()


# ----------------------------
# Action selection helper (fast path)
# ----------------------------
@torch.inference_mode()
def pick_action_from_state5(state5_list):
    """
    state5_list: iterable of 5 floats (progress, measured_power, 3×PAPI-derived)
    Uses global model, device, pref_t, pref_np, ACTIONS.
    """
    s_np = np.asarray(state5_list, dtype=np.float32).reshape(-1)      # (5,)
    inp = np.concatenate([s_np, pref_np], axis=0)                     # (7,)
    x = torch.from_numpy(inp).unsqueeze(0).to(device)                 # (1,7)

    out = model(x)                                                    # (1,32)
    act_vec = out.view(1, 16, 2).squeeze(0)                           # (16,2)
    scores = (act_vec * pref_t).sum(dim=1)                            # (16,)
    a_idx = int(torch.argmax(scores).item())
    return float(ACTIONS[a_idx])


# ----------------------------
# Main experiment runner
# ----------------------------
def experiment_for(APPLICATION, EXP_DIR):
    global EXP_DIR_GLOBAL
    EXP_DIR_GLOBAL = EXP_DIR

    state_dict = initialize_state_dict()

    # Problem size & iterations
    if "stream" in APPLICATION:
        PROBLEM_SIZE, ITERATIONS = 33554432, 10000
    elif "npb" in APPLICATION:
        PROBLEM_SIZE, ITERATIONS = 26, 1000
    else:
        PROBLEM_SIZE, ITERATIONS = 33554432, 10000

    with open(f'{EXP_DIR}/{APPLICATION}_output.log', 'w') as log_file, \
         open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, \
         open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, \
         open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, \
         open(f'{EXP_DIR}/PCAP_file.csv', mode='w', newline='') as PCAP_file, \
         open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:

        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        PCAP_writer = csv.writer(PCAP_file)

        # headers
        power_writer.writerow(['time', 'scope', 'value'])
        progress_writer.writerow(['time', 'value'])
        energy_writer.writerow(['time', 'scope', 'value'])
        PCAP_writer.writerow(['time', 'actuator', 'value'])
        papi_writer.writerow(['time', 'scope', 'value'])

        def cb(*args):
            (sensor, time_ns, scope, value) = args
            scope_uuid = scope.get_uuid()
            sensor = sensor.decode("UTF-8")
            timestamp = time_ns / 1e9

            if sensor == "nrm.benchmarks.progress":
                progress_writer.writerow([timestamp, value])
                state_dict["progress"].append([timestamp, value])

            elif sensor == "nrm.geopm.CPU_POWER":
                power_writer.writerow([timestamp, scope_uuid[-1], value])
                state_dict[f'measured_power_{scope_uuid[-1]}'].append([timestamp, value])

            elif sensor == "nrm.geopm.CPU_ENERGY":
                energy_writer.writerow([timestamp, scope_uuid[-1], value])
                state_dict[f"energy_{scope_uuid[-1]}"].append((timestamp, value))

            elif "PAPI" in sensor:
                papi_writer.writerow([timestamp, sensor, value])
                parts = sensor.split('.')
                state_dict[parts[3]].append((timestamp, value))

        client.set_event_listener(cb)
        client.start_event_listener("")

        # App launch
        if "solvers" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} poor 0 {ITERATIONS}'
        elif "phases" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} 5 1000'
        elif "ones-npb-ft" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 500'
        elif "ones-npb-mg" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'
        elif "ones-npb-bt" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'
        elif "ones-npb-cg" in APPLICATION:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} 1000'
        else:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} {ITERATIONS}'

        process = subprocess.Popen(['bash', '-c', cmd], stdout=log_file, stderr=log_file, env=my_env)

        last_pcap_change = 0.0

        while True:
            now = time.time()
            if now - last_pcap_change >= 2.0:
                if state_dict and state_dict != reference_lib and len(state_dict['progress']) >= 2:
                    try:
                        state5 = process_callback(state_dict)   # 5 floats
                        PCAP = pick_action_from_state5(state5)
                        # PCAP = 165.0
                    except Exception as e:
                        print(f"[WARN] Falling back to default PCAP due to error: {e}")
                        PCAP = 165.0
                else:
                    PCAP = 165.0  # default

                client.actuate(actuators[0], PCAP)
                PCAP_writer.writerow([time.time(), actuators[0], PCAP])
                last_pcap_change = now
                state_dict = initialize_state_dict()

            # time.sleep(0.1)
            if process.poll() is not None:
                print("Process has completed.")
                break

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    time.sleep(1)
    compress_files(current_time, tuple(pref_np.tolist()))
    print("----------------------------------")


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    for STEP in range(1):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>{STEP}")
        for APPLICATION in APPLICATIONS:
            experiment = 'Control_evaluation'
            EXP_DIR = f'{current_dir}/experiment_data/{experiment}/{APPLICATION}'
            if os.path.exists(EXP_DIR):
                print(f"Directory {EXP_DIR} exists")
            else:
                os.makedirs(EXP_DIR, exist_ok=True)
                print(f"Directory {EXP_DIR} created")
            experiment_for(APPLICATION, EXP_DIR)

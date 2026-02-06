#!/usr/bin/env python3
"""
Noise-reduced data generation for NRM + PAPI wrapper runs.

Key changes vs your original:
- Uses a "step" concept: each PCAP action is held for DWELL_SEC.
- Adds SETTLE_SEC at the beginning of each step; samples during settle are labeled stable=0.
- Every sensor row includes step_id and stable flag so you can filter/aggregate cleanly later.
- Uses a random-walk (sticky) action policy instead of uniform random jumps to reduce transients.
- Writes both NRM sensor timestamp (nrm_time) and wall clock time (wall_time) for alignment/debug.

Output CSV schemas:
- measured_power.csv: wall_time, nrm_time, step_id, stable, scope, value
- energy.csv:         wall_time, nrm_time, step_id, stable, scope, value
- progress.csv:       wall_time, nrm_time, step_id, stable, value
- papi.csv:           wall_time, nrm_time, step_id, stable, scope, value
- PCAP_file.csv:      wall_time, step_id, actuator, value

You can later:
- drop stable==0 rows
- aggregate per (step_id) with median/trimmed mean
- compute energy-diff power per step (recommended)
"""

import time
import numpy as np
import sys
import csv
import subprocess
import os
import tarfile
import random
from datetime import datetime
import argparse
import nrm


ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]

APPLICATIONS = [
]

parser = argparse.ArgumentParser(description="Run NRM experiments and log sensors with reduced noise.")
parser.add_argument('-a', '--application', nargs='+', help='List of applications to append')
parser.add_argument('-e', '--experiment', default='random', choices=['random', 'static'],
                    help='Choice of experiment - values random and static')
parser.add_argument('--repeat', type=int, default=2, help='Number of repeats per application')
parser.add_argument('--dwell', type=float, default=8.0, help='Seconds to hold each PCAP action')
parser.add_argument('--settle', type=float, default=2.0, help='Seconds to label as transient after PCAP change')
parser.add_argument('--sleep', type=float, default=0.05, help='Loop sleep (seconds) to avoid busy-waiting')
parser.add_argument('--seed', type=int, default=0, help='Random seed (0 uses time-based seed)')
args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help()

if args.application:
    APPLICATIONS.extend(args.application)

print("Applications:", APPLICATIONS)

# Seed for reproducibility (optional)
if args.seed == 0:
    random.seed()
    np.random.seed()
else:
    random.seed(args.seed)
    np.random.seed(args.seed)

client = nrm.Client()
actuators = client.list_actuators()


def compress_files(exp_dir: str, iteration_tag: str):
    tar_file = os.path.join(exp_dir, f'compressed_iteration_{iteration_tag}.tar')
    with tarfile.open(tar_file, 'w:gz') as tarf:
        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file.endswith(('.csv', '.yaml', '.log')):
                    file_path = os.path.join(root, file)
                    tarf.add(file_path, arcname=os.path.relpath(file_path, exp_dir))
                    os.remove(file_path)
    print(f'Compressed files into {tar_file}')


def make_action_scheduler(n_actions: int, seed=None):
    """
    Returns a function next_idx() that cycles through a shuffled permutation
    of [0..n_actions-1]. This guarantees full action-space coverage each cycle.
    """
    rng = random.Random(seed)
    perm = list(range(n_actions))
    rng.shuffle(perm)
    i = 0

    def next_idx():
        nonlocal perm, i
        if i >= n_actions:
            rng.shuffle(perm)  # new random order each cycle
            i = 0
        idx = perm[i]
        i += 1
        return idx

    return next_idx


def experiment_for(application: str, exp_dir: str, action_fixed: float = None,
                   dwell_sec: float = 8.0, settle_sec: float = 2.0, loop_sleep: float = 0.05):
    if "stream" in application:
        problem_size = 33554432
        iterations = 10000
    elif "npb" in application:
        problem_size = 26
        iterations = 10000
    else:
        problem_size = 26
        iterations = 10000

    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, f"{application}_output.log")
    power_path = os.path.join(exp_dir, "measured_power.csv")
    progress_path = os.path.join(exp_dir, "progress.csv")
    energy_path = os.path.join(exp_dir, "energy.csv")
    pcap_path = os.path.join(exp_dir, "PCAP_file.csv")
    papi_path = os.path.join(exp_dir, "papi.csv")

    with open(log_path, 'w') as log_file, \
         open(power_path, 'w', newline='') as power_file, \
         open(progress_path, 'w', newline='') as progress_file, \
         open(energy_path, 'w', newline='') as energy_file, \
         open(pcap_path, 'w', newline='') as pcap_file, \
         open(papi_path, 'w', newline='') as papi_file:

        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        pcap_writer = csv.writer(pcap_file)

        # Headers include step_id + stable so you can filter transient periods later
        power_writer.writerow(['wall_time', 'nrm_time', 'step_id', 'stable', 'scope', 'value'])
        energy_writer.writerow(['wall_time', 'nrm_time', 'step_id', 'stable', 'scope', 'value'])
        progress_writer.writerow(['wall_time', 'nrm_time', 'step_id', 'stable', 'value'])
        papi_writer.writerow(['wall_time', 'nrm_time', 'step_id', 'stable', 'scope', 'value'])
        pcap_writer.writerow(['wall_time', 'step_id', 'actuator', 'value'])

        # Step state (shared with callback via nonlocal)
        step_id = 0
        step_start_wall = None  # wall clock at actuation
        current_pcap = None
        action_idx = random.randrange(len(ACTIONS))

        # Extra: allow a “warmup” phase to fill buffers (optional)
        warmup_sec = 0.5

        def cb(*cb_args):
            nonlocal step_id, step_start_wall
            (sensor, nrm_time_ns, scope_obj, value) = cb_args

            sensor = sensor.decode("UTF-8")
            nrm_time = nrm_time_ns / 1e9  # NRM event time (seconds)
            wall_time = time.time()       # local wall time at callback reception

            # Guard: if we haven't actuated yet, mark stable=0
            if step_start_wall is None:
                stable = 0
                sid = 0
                age = None
            else:
                age = wall_time - step_start_wall
                stable = 1 if age >= settle_sec else 0
                sid = step_id

            scope_uuid = scope_obj.get_uuid()
            scope = scope_uuid[-1] if scope_uuid else ""

            if sensor == "nrm.benchmarks.progress":
                progress_writer.writerow([wall_time, nrm_time, sid, stable, value])
            elif sensor == "nrm.geopm.CPU_POWER":
                power_writer.writerow([wall_time, nrm_time, sid, stable, scope, value])
            elif sensor == "nrm.geopm.CPU_ENERGY":
                energy_writer.writerow([wall_time, nrm_time, sid, stable, scope, value])
            elif "PAPI" in sensor:
                papi_writer.writerow([wall_time, nrm_time, sid, stable, sensor, value])

        client.set_event_listener(cb)
        client.start_event_listener("")

        # Launch the workload (same commands as you had, kept intact)
        if "solvers" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} {problem_size} poor 0 {iterations}'
        elif "phases" in application:
            print(f"Starting Execution of phases: {(application, problem_size, iterations)}")
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} {problem_size} 5 200'
        elif "ones-npb-ft" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} 500'
        elif "ones-npb-mg" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} 1000'
        elif "ones-npb-bt" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} 1000'
        elif "ones-npb-cg" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} 1000'
        elif "ones-npb-is" in application:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} 26 1000'
        else:
            cmd = f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {application} {problem_size} {iterations}'

        process = subprocess.Popen(['bash', '-c', cmd], stdout=log_file, stderr=log_file)

        # Warmup before first actuation (lets listener start receiving)
        time.sleep(warmup_sec)

        last_change_wall = 0.0
        # Create scheduler once per run (NOT inside the loop)
        if action_fixed is None:
            next_action_idx = make_action_scheduler(
                len(ACTIONS),
                seed=(args.seed if args.seed != 0 else None)
            )
        else:
            next_action_idx = None

        # Main loop
        while True:
            now = time.time()

            if (now - last_change_wall) >= dwell_sec:
                if action_fixed is not None:
                    current_pcap = action_fixed
                else:
                    idx = next_action_idx()          # <-- just call it
                    current_pcap = ACTIONS[idx]

                step_id += 1
                step_start_wall = now

                client.actuate(actuators[0], current_pcap)
                pcap_writer.writerow([step_start_wall, step_id, actuators[0], current_pcap])

                last_change_wall = now
                print(f"[{application}] step={step_id} pcap={current_pcap}")

            time.sleep(loop_sleep)


            # Stop once the process completes
            if process.poll() is not None:
                print(f"[{application}] process completed.")
                break

    # Compress after run
    iteration_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    compress_files(exp_dir, iteration_tag)
    print("----------------------------------")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    repeat = args.repeat
    dwell_sec = args.dwell
    settle_sec = args.settle
    loop_sleep = args.sleep

    if args.experiment == 'random':
        for rep in range(repeat):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>{rep}")
            for app in APPLICATIONS:
                experiment = 'training_data'
                exp_dir = os.path.join(current_dir, "experiment_data", experiment, app)
                experiment_for(app, exp_dir, action_fixed=None,
                               dwell_sec=dwell_sec, settle_sec=settle_sec, loop_sleep=loop_sleep)
                time.sleep(1.0)

    elif args.experiment == 'static':
        for fixed_action in ACTIONS:
            for rep in range(repeat):
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>{rep} ACTION={fixed_action}")
                for app in APPLICATIONS:
                    experiment = 'plotting_data'
                    exp_dir = os.path.join(current_dir, "experiment_data", experiment, app)
                    experiment_for(app, exp_dir, action_fixed=fixed_action,
                                   dwell_sec=dwell_sec, settle_sec=settle_sec, loop_sleep=loop_sleep)
                    time.sleep(1.0)

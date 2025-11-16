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





ACTIONS = [
    1000000000.0, 1100000000.0, 1200000000.0, 1300000000.0, 1400000000.0,
    1500000000.0, 1600000000.0, 1700000000.0, 1800000000.0, 1900000000.0,
    2000000000.0, 2100000000.0, 2200000000.0, 2300000000.0, 2400000000.0,
    2500000000.0, 2600000000.0, 2700000000.0, 2800000000.0, 2900000000.0,
    3000000000.0, 3100000000.0, 3200000000.0, 3300000000.0, 3400000000.0,
    3500000000.0, 3600000000.0, 3700000000.0, 3800000000.0, 4000000000.0,
]


# APPLICATIONS = ['ones-stream-scale', 'ones-stream-triad', 'ones-npb-ep', 'ones-stream-copy', 'ones-stream-add', 'ones-npb-is', 'ones-npb-cg', 'ones-npb-bt', 'ones-npb-ft']

APPLICATIONS = []
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


def experiment_for(APPLICATION, EXP_DIR, ACTION=None):
    if "stream" in APPLICATION:
        PROBLEM_SIZE = 33554432
        ITERATIONS = 10000
    elif "npb" in APPLICATION:
        PROBLEM_SIZE = 26
        ITERATIONS = 1000
    # create a separate CSV to record the frequency actions (FREQ_file.csv)
    with open(f'{EXP_DIR}/{APPLICATION}_output.log','w') as log_file, open(f'{EXP_DIR}/frequency.csv', mode='w', newline='') as frequency_file, open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, open(f'{EXP_DIR}/FREQ_file.csv', mode='w', newline='') as FREQ_file, open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:
        frequency_writer = csv.writer(frequency_file)
        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        FREQ_writer = csv.writer(FREQ_file)
        power_writer.writerow(['time', 'scope', 'value'])
        progress_writer.writerow(['time', 'value'])
        energy_writer.writerow(['time', 'scope', 'value'])
        # FREQ action logfile: time, actuator_uuid, value(Hz)
        FREQ_writer.writerow(['time', 'actuator', 'value'])
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
            elif "FREQUENCY" in sensor:
                frequency_writer.writerow([timestamp, scope[-1], value])



        client.set_event_listener(cb)
        client.start_event_listener("") 
        print(f"Starting execution of {APPLICATION} with problem size {PROBLEM_SIZE} for {ITERATIONS} iterations")
        if "solvers" in APPLICATION:
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} poor 0 {ITERATIONS}'],
                stdout=log_file,
                stderr=log_file
            )
        elif "phases" in APPLICATION:    
            print(f"Starting Execution of phases {APPLICATION, PROBLEM_SIZE, ITERATIONS}")
            process = subprocess.Popen(
                ['bash', '-c', f'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {APPLICATION} {PROBLEM_SIZE} 5 200'],
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
        else:
            process = subprocess.Popen(
                ['bash', '-c', 'time nrm-papiwrapper -i -e PAPI_L3_TCA -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e PAPI_L3_TCM -- {} {} {}'.format(APPLICATION, PROBLEM_SIZE, ITERATIONS)],
                stdout=log_file,
                stderr=log_file
            )
        time.sleep(0.5)
   
        
        last_pcap_change = 0
        while True:
            current_time = time.time()
            if current_time - last_pcap_change >= 2:
                if not ACTION:
                    FREQ = random.choice(ACTIONS)
                else:
                    FREQ = ACTION
                print(FREQ)
                # Actuate CPU frequency actuator (assumed to be actuators[1])
                try:
                    client.actuate(actuators[1], FREQ)
                except Exception as e:
                    # fallback: try to actuate first actuator if second missing
                    try:
                        client.actuate(actuators[0], FREQ)
                    except Exception:
                        print(f"[WARN] failed to actuate frequency: {e}")
                FREQ_time = time.time()
                FREQ_writer.writerow([FREQ_time, actuators[1] if len(actuators) > 1 else actuators[0], FREQ])
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




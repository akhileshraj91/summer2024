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





# ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
# ACTIONS = [78.0]

# argument parser for the application

i = 0
APPLICATIONS = ['ones-stream-full', 'ones-stream-triad', 'ones-stream-add', 'ones-stream-copy', 'ones-stream-scale', 'ones-npb-ep']
# APPLICATIONS = ['ones-stream-full', 'ones-stream-triad', 'ones-stream-add', 'ones-stream-copy', 'ones-stream-scale', 'ones-npb-ep', 'ones-solvers-cg', 'ones-solvers-bicgstab']
while i < len(sys.argv):
    if sys.argv[i] == '--application':
        APPLICATION = sys.argv[i+1]
        i += 1
    i +=1

# define the problem size and iterations based on the applications




# initialize the nrm clients

client = nrm.Client()
actuators = client.list_actuators()
ACTIONS = actuators[0].list_choices()


# For post processing
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


def experiment_for(APPLICATION, EXP_DIR):
    if "stream" in APPLICATION:
        PROBLEM_SIZE = 33554432
        ITERATIONS = 100000
    elif "solvers" in APPLICATION:
        PROBLEM_SIZE = 10000
        ITERATIONS = 100000
    elif "ep" in APPLICATION:
        PROBLEM_SIZE = 22
        ITERATIONS = 100000
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
            # print(args)
            (sensor, time, scope, value) = args
            scope = scope.get_uuid()
            sensor = sensor.decode("UTF-8")
            timestamp = time/1e9
            if sensor == "nrm.benchmarks.progress":
                progress_writer.writerow([timestamp, value])
            elif sensor == "nrm.geopm.CPU_POWER":
                # print(scope[-1])
                power_writer.writerow([timestamp, scope[-1], value])
            elif sensor == "nrm.geopm.CPU_ENERGY":
                # print(scope[-1])
                energy_writer.writerow([timestamp, scope[-1], value])
            elif "PAPI" in sensor:
                # print(args)
                papi_writer.writerow([timestamp, sensor, value])


        client.set_event_listener(cb)
        client.start_event_listener("") 
        if "solvers" in APPLICATION:
            process = subprocess.Popen(['nrm-papiwrapper', '-i', '-e', 'PAPI_L3_TCA', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'{APPLICATION}', f'{PROBLEM_SIZE}', 'poor', '0', f'{ITERATIONS}'])
        else:    
            process = subprocess.Popen(['nrm-papiwrapper', '-i', '-e', 'PAPI_L3_TCA', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'{APPLICATION}', f'{PROBLEM_SIZE}', f'{ITERATIONS}'])


        last_pcap_change = 0
        while True:
            current_time = time.time()
            if current_time - last_pcap_change >= 5:
                PCAP = random.choice(ACTIONS)
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
    # Get the current file path
    current_file_path = os.path.abspath(__file__)

    # Get the directory containing the current file
    current_dir = os.path.dirname(current_file_path)


    for APPLICATION in APPLICATIONS:
        experiment = 'data_generation'
        EXP_DIR = f'{current_dir}/experiment_data/{experiment}/{APPLICATION}'
        if os.path.exists(EXP_DIR):
            print(f"Directories {EXP_DIR} exist")
        else:
            os.makedirs(EXP_DIR)
            print(f"Directory {EXP_DIR} created") 
        experiment_for(APPLICATION, EXP_DIR)



# compress the experiment details for post processing

import time
import numpy as np
import sys
import pandas as pd
import nrm
import csv
import subprocess
import os
import tarfile




ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
# ACTIONS = [78.0]

# argument parser for the application

i = 0
# APPLICATION = 'ones-stream-full'
APPLICATIONS = ['ones-stream-full', 'ones-stream-triad', 'ones-stream-add', 'ones-stream-copy', 'ones-stream-scale']
while i < len(sys.argv):
    if sys.argv[i] == '--application':
        APPLICATION = sys.argv[i+1]
        i += 1
    i +=1

# define the problem size and iterations based on the applications

# if APPLICATION == 'ones-stream-full':
PROBLEM_SIZE = 33554432
ITERATIONS = 1000

# initialize the nrm clients

client = nrm.Client()
actuators = client.list_actuators()


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


def experiment_for(PCAP, APPLICATION, EXP_DIR):
    with open(f'{EXP_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{EXP_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{EXP_DIR}/energy.csv', mode='w', newline='') as energy_file, open(f'{EXP_DIR}/parameters.yaml', mode='w', newline='') as parameter_file, open(f'{EXP_DIR}/papi.csv', mode='w', newline='') as papi_file:
        power_writer = csv.writer(power_file)
        progress_writer = csv.writer(progress_file)
        energy_writer = csv.writer(energy_file)
        papi_writer = csv.writer(papi_file)
        # Write headers if files are empty
        power_writer.writerow(['time', 'scope', 'value'])
        progress_writer.writerow(['time', 'value'])
        energy_writer.writerow(['time', 'scope', 'value'])
        parameter_file.write(f"PCAP: {PCAP}\n")
        papi_writer.writerow(['time','scope','value']) 

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
                print(args)
                papi_writer.writerow([timestamp, sensor, value])


        client.set_event_listener(cb)
        client.start_event_listener("") 
        process = subprocess.Popen(['nrm-papiwrapper', '-i' , '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'{APPLICATION}', '33554432', '1000'])
        #process = subprocess.Popen(['sudo','nrm-papiwrapper', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCR', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_L3_TCM', '--', 'ones-stream-full', '33554432', '1000'])
        # -e PAPI_TOT_INS -e PAPI_TOT_CYC -e PAPI_RES_STL -e ###PAPI_VEC_INS### -e PAPI_L3_TCR -e PAPI_L3_TCM
        client.actuate(actuators[0],PCAP)
        while True:
            time.sleep(1)
            if process.poll() is not None:  
                print("Process has completed.")
                break
    compress_files(PCAP)
    print("----------------------------------")


# start the experiment 



   

if __name__ == "__main__":
    for APPLICATION in APPLICATIONS:
        experiment = 'identification'
        EXP_DIR = f'./experiment_data/{experiment}/{APPLICATION}'
        if os.path.exists(EXP_DIR):
            print(f"Directories {EXP_DIR} exist")
        else:
            os.makedirs(EXP_DIR)
            print(f"Directory {EXP_DIR} created") 
        for PCAP in ACTIONS:
            experiment_for(PCAP, APPLICATION, EXP_DIR)



# compress the experiment details for post processing

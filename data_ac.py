import nrm
import subprocess
import time
import csv
import os
import shutil

client = nrm.Client()
DATA_DIR = "./test_results"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

with open(f'{DATA_DIR}/measured_power.csv', mode='w', newline='') as power_file, open(f'{DATA_DIR}/progress.csv', mode='w', newline='') as progress_file, open(f'{DATA_DIR}/energy.csv', mode='w', newline='') as energy_file:
    power_writer = csv.writer(power_file)
    progress_writer = csv.writer(progress_file)
    energy_writer = csv.writer(energy_file)

    if power_file.tell() == 0:
        power_writer.writerow(['time', 'sensor', 'scope', 'value'])

    if progress_file.tell() == 0:
        progress_writer.writerow(['time', 'sensor', 'scope', 'value'])
    
    if energy_file.tell() == 0:
        energy_writer.writerow(['time', 'sensor', 'scope', 'value'])


    def cb(*args):
        (sensor, time, scope, value) = args
        scope = scope.get_uuid()
        sensor = sensor.decode("UTF-8")
        timestamp = time/1e9
        if sensor == "nrm.benchmarks.progress":
            progress_writer.writerow([timestamp, sensor, scope, value])
        elif sensor == "nrm.geopm.CPU_POWER":
            print(value)
            power_writer.writerow([timestamp, sensor, scope[-1], value])
        elif sensor == "nrm.geopm.CPU_ENERGY":
            energy_writer.writerow([timestamp, sensor, scope[-1], value])

    client.set_event_listener(cb)
    client.start_event_listener("") 
    process = subprocess.Popen(['phases-stream-full', '33554432', '5', '250'])


    switch = True
    while True:
        actuators = client.list_actuators()
        if switch:
            client.actuate(actuators[0], 78.0)
            switch = False
        else:
            client.actuate(actuators[0], 165.0)
            switch = True
        print(actuators)
        time.sleep(2)
        if process.poll() is not None:  
            print("Process has completed.")
            break

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        print(f"Deleted DATA_DIR: {DATA_DIR}")


import nrm
import subprocess
import time
import csv

# Initialize the client
client = nrm.Client()

# Open CSV files for appending data
with open('measured_power.csv', mode='w', newline='') as power_file, open('progress.csv', mode='w', newline='') as progress_file, open('energy.csv', mode='w', newline='') as energy_file:
    power_writer = csv.writer(power_file)
    progress_writer = csv.writer(progress_file)
    energy_writer = csv.writer(energy_file)

    # Write headers if files are empty
    if power_file.tell() == 0:
        power_writer.writerow(['time', 'sensor', 'scope', 'value'])

    if progress_file.tell() == 0:
        progress_writer.writerow(['time', 'sensor', 'scope', 'value'])
    
    if energy_file.tell() == 0:
        energy_writer.writerow(['time', 'sensor', 'scope', 'value'])


    def cb(*args):
        # print(args)
        (sensor, time, scope, value) = args
        # print(args)
        scope = scope.get_uuid()
        sensor = sensor.decode("UTF-8")
        timestamp = time/1e9
        if sensor == "nrm.benchmarks.progress":
            # print(value)
            progress_writer.writerow([timestamp, sensor, scope, value])
        elif sensor == "nrm.geopm.CPU_POWER":
            print(value)
            power_writer.writerow([timestamp, sensor, scope[-1], value])
        elif sensor == "nrm.geopm.CPU_ENERGY":
            # print(sensor)
            energy_writer.writerow([timestamp, sensor, scope[-1], value])

    client.set_event_listener(cb)
    client.start_event_listener("") 
    # actuators = client.list_actuators()
    # print("--------", actuators)
    # Run the terminal command in the background
    # process = subprocess.Popen(['ones-npb-ep', '22', '1000'])
    # process = subprocess.Popen(['nrm-papiwrapper', '-i', '-e', 'PAPI_L3_TCA', '-e', 'PAPI_TOT_INS', '-e', 'PAPI_TOT_CYC', '-e', 'PAPI_RES_STL', '-e', 'PAPI_L3_TCM', '--', f'ones-solvers-cg', f'3000', 'poor', '0', f'1000'])

    process = subprocess.Popen(['phases-stream-full', '33554432', '10', '500'])


    # Keep the main script running
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
        time.sleep(3)
        if process.poll() is not None:  # Process has completed
            print("Process has completed.")
            break


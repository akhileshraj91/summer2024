import os
import pandas as pd
import matplotlib.pyplot as plt
# import ruamel.yaml
import math
import scipy.optimize as opt
import numpy as np
import tarfile
from matplotlib import cm
# import seaborn as sns
import warnings
# import yaml
os.chdir(os.path.dirname(__file__))

warnings.filterwarnings('ignore')

experiment = "static_analysis"
# experiment_dir = "./"
experiment_dir = "/home/cc/summer2024/tests/experiment_data/identification/ones-stream-full"

exp_files = next(os.walk(experiment_dir))[2]
print(exp_files)

data = {}

if "progress.csv" and "energy.csv" in exp_files:
    folder_path = experiment_dir
else:
    print ("Wrong directory")


pubEnergy = pd.read_csv(folder_path+"/energy.csv")
pubProgress = pd.read_csv(folder_path+"/progress.csv")

geopm_sensor0 = geopm_sensor1 = pd.DataFrame({'timestamp':[],'value':[]})
for i,row in pubEnergy.iterrows():
    if i%2 == 0:
        geopm_sensor0 = pd.concat([geopm_sensor0, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)
    else:
        geopm_sensor1 = pd.concat([geopm_sensor1, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)

data['geopm_power_0'] = [(geopm_sensor0['value'][i] - geopm_sensor0['value'][i-1])/(geopm_sensor0['timestamp'][i] - geopm_sensor0['timestamp'][i-1]) for i in range(1,len(geopm_sensor0))]
data['geopm_power_1'] = [(geopm_sensor1['value'][i] - geopm_sensor1['value'][i-1])/(geopm_sensor1['timestamp'][i] - geopm_sensor1['timestamp'][i-1]) for i in range(1,len(geopm_sensor1))]
data['timestamp'] = (geopm_sensor0['timestamp'] + geopm_sensor1['timestamp']) / 2

min_length = min(len(data['geopm_power_0']), len(data['geopm_power_1']))
geopm_power_0 = data['geopm_power_0'][:min_length]
geopm_power_1 = data['geopm_power_1'][:min_length]

average_power = [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0, geopm_power_1)]

data['average_power'] = average_power

print(average_power)


# plt.savefig("./test.pdf")
# geopm_power_0.plot(ax=axs[1])


def measure_progress(progress_data, energy_data):
    progress_sensor = pd.DataFrame(progress_data)
    first_sensor_point = min(energy_data['time'][0], progress_sensor['time'][0])
    progress_sensor['elapsed_time'] = progress_sensor['time'] - first_sensor_point  # New column for elapsed time
    # progress_sensor = progress_sensor.set_index('elapsed_time')
    performance_elapsed_time = progress_sensor.elapsed_time
    # Add performance_frequency as a new column in progress_sensor
    frequency_values = [
        progress_data['value'].iloc[t] / (performance_elapsed_time[t] - performance_elapsed_time[t-1]) for t in range(1, len(performance_elapsed_time))
    ]
    
    # Ensure the frequency_values length matches the index length
    frequency_values = [0] + frequency_values  # Prepend a 0 for the first index
    progress_sensor['frequency'] = frequency_values
    return progress_sensor


progress = measure_progress(pubProgress,pubEnergy)
data['progress'] = progress

data['upsampled_timestamps'] = data['timestamp']

data['progress_frequency_median'] = pd.DataFrame({'median': np.nanmedian(
            progress['frequency'].where(
            progress['time'] <=
                data['upsampled_timestamps'][0], 0)), 'timestamp':data['upsampled_timestamps'][0]}, index=[0])

for t in range(len(data['upsampled_timestamps'])):
    # Change append to concat for better performance
    data['progress_frequency_median'] = pd.concat([data['progress_frequency_median'], 
        pd.DataFrame({
            'median': np.nanmedian(
                progress['frequency'].where(
                    progress['time'] <= data['upsampled_timestamps'][t], 0)
            ),
            'timestamp': data['upsampled_timestamps'][t]
        }, index=[0])], ignore_index=True)

data['progress_frequency_median']['elapsed_time'] = data['progress_frequency_median'].timestamp-data['progress_frequency_median'].timestamp[0]

fig,axs = plt.subplots(nrows=2,ncols=1)
axs[0].scatter(range(len(average_power)),average_power)
axs[1].scatter(data['progress_frequency_median']['elapsed_time'],data['progress_frequency_median']['median'])
plt.show()
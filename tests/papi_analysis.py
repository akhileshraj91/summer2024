import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import tarfile
from matplotlib import cm
import warnings
import yaml
import csv
import re
warnings.filterwarnings('ignore')

current_working_directory = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_working_directory)

exp_type = 'identification' 
experiment_dir = f'{current_working_directory}/experiment_data/{exp_type}/'

root, apps, files = next(os.walk(experiment_dir))
num_subs = len(apps)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
fig_pow, axs_pow = plt.subplots(1,1)
# Create a colormap
colormap = plt.cm.get_cmap('tab10', len(apps))


def compute_power(pubEnergy):
    power = {}
    geopm_sensor0 = geopm_sensor1 = pd.DataFrame({'timestamp':[],'value':[]})
    for i,row in pubEnergy.iterrows():
        if i%2 == 0:
            geopm_sensor0 = pd.concat([geopm_sensor0, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)
        else:
            geopm_sensor1 = pd.concat([geopm_sensor1, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)

    power['geopm_power_0'] = [(geopm_sensor0['value'][i] - geopm_sensor0['value'][i-1])/(geopm_sensor0['timestamp'][i] - geopm_sensor0['timestamp'][i-1]) for i in range(1,len(geopm_sensor0))]
    power['geopm_power_1'] = [(geopm_sensor1['value'][i] - geopm_sensor1['value'][i-1])/(geopm_sensor1['timestamp'][i] - geopm_sensor1['timestamp'][i-1]) for i in range(1,len(geopm_sensor1))]
    power['timestamp'] = (geopm_sensor0['timestamp'] + geopm_sensor1['timestamp']) / 2

    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]

    average_power = [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0, geopm_power_1)]

    power['average_power'] = average_power
    return power

legend_handles = []
legend_labels = []
traces = {}
traces_tmp = {}
for app_index, app in enumerate(apps):
    cwd = root + '/' + app
    # traces[app] = pd.DataFrame()
    if next(os.walk(cwd))[1] == []:
        files = os.listdir(cwd)
        for fname in files:
            if fname.endswith("tar"):
                print(fname)
                tar = tarfile.open(cwd + '/' + fname, "r")
                tar.extractall(path=cwd + '/' + fname[:-4])
                tar.close()
    traces[app] = next(os.walk(cwd))[1]
    data = {}
    for trace in traces[app]:
        data[trace] = {}
        pwd = f'{cwd}/{trace}'
        papi_file = pd.read_csv(f'{pwd}/papi.csv')
        energy_file = pd.read_csv(f'{pwd}/energy.csv')
        data[trace]['power_calculated'] = compute_power(energy_file)
        with open(f'{pwd}/parameters.yaml', 'r') as yaml_file:
            parameters = yaml.safe_load(yaml_file)
        data[trace]['PCAP'] = parameters['PCAP']
        for row in papi_file.iterrows():
            timestamp = row[1]['time']
            scope_info = row[1]['scope']
            value = row[1]['value']
            
            # Extract the scope name using a regular expression
            match = re.search(r'\bPAPI_[A-Z_1-9]+\b', scope_info)
            if match:
                scope_name = match.group(0)
            else:
                continue
            
            if scope_name not in data[trace].keys():
                data[trace][scope_name] = {}
                data[trace][scope_name]['timestamp'] = []
                data[trace][scope_name]['value'] = []
                
            data[trace][scope_name]['timestamp'].append(timestamp)
            data[trace][scope_name]['value'].append(value)
        count = 0


        for scope in data[trace].keys():
            if scope not in ['PCAP', 'power_calculated']:
                data[trace][scope]['instantaneous_values'] = [data[trace][scope]['value'][0]]
                data[trace][scope]['instantaneous_values'].extend([data[trace][scope]['value'][t] - data[trace][scope]['value'][t-1] for t in range(1,len(data[trace][scope]['value']))])
                data[trace][scope]['average'] = sum(data[trace][scope]['instantaneous_values']) / len(data[trace][scope]['instantaneous_values'])   
                ax = axs[round((count+1)/4), count%2]
                scatter = ax.scatter(data[trace]['PCAP'], data[trace][scope]['average'], label=f'{app}', color=colormap(app_index))
                ax.set_xlabel('PCAP', fontsize=10)  # Adjust font size
                ax.set_ylabel(f'{scope} - Average', fontsize=10)  # Adjust font size
                count += 1

                # Collect handles and labels for the combined legend
                print(app,legend_labels)
                if app not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append(f'{app}')
        axs_pow.scatter(data[trace]['PCAP'],np.mean(data[trace]['power_calculated']['average_power']), label=f'{app}', color=colormap(app_index))
        if np.mean(data[trace]['power_calculated']['average_power']) < 0:
            print("pause")
fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=8)  # Adjust legend position and font size
fig.savefig(f'{current_working_directory}/average_plot.pdf')
# fig_pow.legend(legend_handles, legend_labels, loc='upper right', ncols=4, fontsize=8)
fig_pow.savefig(f'{current_working_directory}/power_relation.pdf')



plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
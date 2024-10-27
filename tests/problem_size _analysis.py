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

plt.rcParams.update({'font.size': 12, "font.weight": "bold", 'axes.labelsize': 'x-large'})
plt.tight_layout()  # Add this line to adjust the layout
warnings.filterwarnings('ignore')

current_working_directory = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_working_directory)

exp_type = 'problem_size_identification' 
# fig, axs = plt.subplots(2, 1, figsize=(12, 10))
# fig_avg, axs_avg = plt.subplots(2, 1, figsize=(12, 10))
# fig_PAPI, axs_PAPI = plt.subplots(5, 1, figsize=(12, 16))

def calculate_power_with_wraparound(current, previous, time_diff, wraparound_value=262143.328850):
    diff = current - previous
    if diff < 0:  # Wraparound detected
        diff = (wraparound_value - previous) + current
    return diff / time_diff

def compute_power(pubEnergy):
    power = {}
    geopm_sensor0 = geopm_sensor1 = pd.DataFrame({'timestamp':[],'value':[]})
    for i,row in pubEnergy.iterrows():
        if i%2 == 0:
            geopm_sensor0 = pd.concat([geopm_sensor0, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)
        else:
            geopm_sensor1 = pd.concat([geopm_sensor1, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)


    power['geopm_power_0'] = pd.DataFrame({
        'timestamp': geopm_sensor0['timestamp'][1:],  # Add timestamps
        'power': [
            calculate_power_with_wraparound(
                geopm_sensor0['value'][i],
                geopm_sensor0['value'][i-1],
                geopm_sensor0['timestamp'][i] - geopm_sensor0['timestamp'][i-1]
            ) for i in range(1, len(geopm_sensor0))
        ]
    })

    # Apply the same logic to geopm_power_1
    power['geopm_power_1'] = pd.DataFrame({
        'timestamp': geopm_sensor1['timestamp'][1:],  # Add timestamps
        'power': [
            calculate_power_with_wraparound(
                geopm_sensor1['value'][i],
                geopm_sensor1['value'][i-1],
                geopm_sensor1['timestamp'][i] - geopm_sensor1['timestamp'][i-1]
            ) for i in range(1, len(geopm_sensor1))
        ]
    })

    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]

    average_power = pd.DataFrame({
        'timestamp': geopm_power_0['timestamp'],  # Use the timestamp from geopm_power_0
        'average_power': [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0['power'], geopm_power_1['power'])]
    })
    average_power['elapsed_time'] = average_power['timestamp'] - average_power['timestamp'].iloc[0]
    power['average_power'] = average_power
    return power

def measure_progress(progress_data, energy_data):
    Progress_DATA = {} 
    progress_sensor = pd.DataFrame(progress_data)
    first_sensor_point = min(energy_data['average_power']['timestamp'].iloc[0], progress_sensor['time'][0])
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
    upsampled_timestamps= energy_data['average_power']['timestamp']
    
    # true_count = (progress_sensor['time'] <= upsampled_timestamps.iloc[0]).sum()

    progress_frequency_median = pd.DataFrame({'median': np.nanmedian(progress_sensor['frequency'].where(progress_sensor['time'] <= upsampled_timestamps.iloc[0])), 'timestamp': upsampled_timestamps.iloc[0]}, index=[0])
    for t in range(1, len(upsampled_timestamps)):
        progress_frequency_median = pd.concat([progress_frequency_median, pd.DataFrame({'median': [np.nanmedian(progress_sensor['frequency'].where((progress_sensor['time'] >= upsampled_timestamps.iloc[t-1]) & (progress_sensor['time'] <= upsampled_timestamps.iloc[t])))],
        'timestamp': [upsampled_timestamps.iloc[t]]})], ignore_index=True)
    progress_frequency_median['elapsed_time'] = progress_frequency_median['timestamp'] - progress_frequency_median['timestamp'].iloc[0]
    # Assign progress_frequency_median as a new column
    Progress_DATA['progress_sensor'] = progress_sensor
    Progress_DATA['progress_frequency_median'] = progress_frequency_median
    return Progress_DATA

def collect_papi(PAPI_data):
    PAPI = {}
    for scope in PAPI_data['scope'].unique():
        # Extract the string between the 3rd and 4th dots
        scope_parts = scope.split('.')
        if len(scope_parts) > 4:  # Ensure there are enough parts
            extracted_scope = scope_parts[3]
            # Aggregate the data for the extracted scope using pd.concat
            PAPI[extracted_scope] = PAPI_data[PAPI_data['scope'] == scope]
            instantaneous_values = [0] + [PAPI[extracted_scope]['value'].iloc[k] - PAPI[extracted_scope]['value'].iloc[k-1] for k in range(1,len(PAPI[extracted_scope]))]
            # Normalize the instantaneous values between 0 and 10
            min_val = min(instantaneous_values)
            max_val = max(instantaneous_values)
            # PAPI[extracted_scope]['instantaneous_value'] = [(value - min_val) / (max_val - min_val) * 10 for value in instantaneous_values]
            PAPI[extracted_scope]['instantaneous_value'] = instantaneous_values
            PAPI[extracted_scope]['elapsed_time'] = PAPI[extracted_scope]['time'] - PAPI[extracted_scope]['time'].iloc[0]
            # PAPI[extracted_scope]['max_min'] = {}
    return PAPI

def normalize(traces):
    normalized_PAPI = {}
    max_value = {'PAPI_L3_TCA': float('-inf'), 'PAPI_TOT_INS': float('-inf'), 'PAPI_TOT_CYC': float('-inf'), 'PAPI_RES_STL': float('-inf'), 'PAPI_L3_TCM': float('-inf')}
    min_value = {'PAPI_L3_TCA': float('inf'), 'PAPI_TOT_INS': float('inf'), 'PAPI_TOT_CYC': float('inf'), 'PAPI_RES_STL': float('inf'), 'PAPI_L3_TCM': float('inf')}
    for app in traces.keys():
        for trace in traces[app]['data'].keys():
            for scope in traces[app]['data'][trace]['papi'].keys():
                max_value[scope] = max(max_value[scope],max(traces[app]['data'][trace]['papi'][scope]['instantaneous_value']))
                min_value[scope] = min(min_value[scope],min(traces[app]['data'][trace]['papi'][scope]['instantaneous_value']))
    for app in traces.keys():
        for trace in traces[app]['data'].keys():
            for scope in traces[app]['data'][trace]['papi'].keys():
                traces[app]['data'][trace]['papi'][scope]['normalized_value'] = [(value - min_value[scope]) / (max_value[scope] - min_value[scope]) * 10 for value in traces[app]['data'][trace]['papi'][scope]['instantaneous_value']]
    return traces

def main(folder):  # Define the main function
    experiment_dir = f'{current_working_directory}/experiment_data/{folder}/{exp_type}/'

    print(experiment_dir)
    root, apps, files = next(os.walk(experiment_dir))
    num_subs = len(apps)

    colormap = plt.cm.get_cmap('tab10', len(apps))
    
    traces = {}

    # Check if RESULTS directory exists, if not, create it
    results_dir = f'{current_working_directory}/experiment_data/RESULTS'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Create the directory if it doesn't exist

    for app_index, app in enumerate(apps):
        cwd = root + '/' + app
        if next(os.walk(cwd))[1] == []:
            files = os.listdir(cwd)
            for fname in files:
                if fname.endswith("tar"):
                    tar = tarfile.open(cwd + '/' + fname, "r")
                    tar.extractall(path=cwd + '/' + fname[:-4])
                    tar.close()
        traces[app] = {'directories': next(os.walk(cwd))[1]}
        traces[app]['data'] = {}
        for trace in traces[app]['directories']:
            traces[app]['data'][trace] = {}
            pwd = f'{cwd}/{trace}'
            pubProgress = pd.read_csv(f'{pwd}/progress.csv')
            pubEnergy = pd.read_csv(f'{pwd}/energy.csv')
            pubPAPI = pd.read_csv(f'{pwd}/papi.csv')
            parameters = yaml.safe_load(open(f'{pwd}/parameters.yaml'))  # Load parameters from YAML file

            traces[app]['data'][trace]['power'] = compute_power(pubEnergy)
            traces[app]['data'][trace]['progress'] = measure_progress(pubProgress, traces[app]['data'][trace]['power'])
            traces[app]['data'][trace]['papi'] = collect_papi(pubPAPI)
            traces[app]['data'][trace]['PARAMS'] = parameters
    return traces

    # traces = normalize(traces)




if __name__ == "__main__":  # Ensure the main function runs when the script is executed
    root, exps, files = next(os.walk(os.path.join(current_working_directory, 'experiment_data')))
    PLOT_DATA = {}
    for exp in exps:
        if "exp" in exp:
            PLOT_DATA[exp] = main(exp)
    print(PLOT_DATA)
    indieces = PLOT_DATA['experiment_data_3']['ones-stream-full']['data'].keys()     
    scopes = PLOT_DATA['experiment_data_3']['ones-stream-full']['data']['compressed_iteration_100000']['papi'].keys()
    plot_dict = {i:{j:[] for j in scopes}for i  in indieces}
    for traces in PLOT_DATA.keys():
        for app in PLOT_DATA[traces].keys():
            for trace in PLOT_DATA[traces][app]['data'].keys():
                for scope in PLOT_DATA[traces][app]['data'][trace]['papi'].keys():
                    plot_dict[trace][scope].append(np.mean(PLOT_DATA[traces][app]['data'][trace]['papi'][scope]['instantaneous_value']))
                
    # print(plot_dict)
    high_contrast_colors = plt.cm.tab10.colors[:4]
    fig, axs = plt.subplots(5,1,figsize=(12,10))
    for x in plot_dict.keys():
        count = 0
        for y in plot_dict[x].keys():
            ps = int(re.search(r'\d+', x).group())
            mean = np.mean(plot_dict[x][y])
            std = np.std(plot_dict[x][y])
            
            axs[count].errorbar(ps, mean, std, label=x, elinewidth=2,fmt='o', color = high_contrast_colors[0])
            axs[count].set_title(y)  # Add title for each subplot
            axs[count].grid(True)
            
            axs[count].tick_params(labelbottom=False)  # Keep y-axis ticks visible
            if count == 4:
                axs[count].tick_params(labelbottom=True)  # Keep x-axis ticks visible for the 5th subplot
            else:
                axs[count].tick_params(labelbottom=False)  # Hide x-axis ticks for other subplots
            count += 1

    plt.savefig('output_problem_size_PAPI.pdf')  # Save the figure as a PNG file
    plt.show()
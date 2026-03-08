import os
import pandas as pd
import matplotlib.pyplot as plt
import ruamel.yaml
import math
import numpy as np
import tarfile
from matplotlib import cm
from matplotlib import pyplot as plt
import warnings
import yaml
yaml_format = ruamel.yaml.YAML()
warnings.filterwarnings('ignore')
from matplotlib.ticker import FuncFormatter  # Add this import at the top
import pickle  # Add this import at the top
import matplotlib.ticker as mticker


import matplotlib.ticker as ticker  # Add this import at the top
OUTPUT_DIR = './figures'

app_name_mapping = {
    'ones-npb-ft': 'NPB-FT',
    'ones-npb-is': 'NPB-IS',
    'ones-stream-triad': 'Stream-Triad',
    'ones-stream-scale': 'Stream-Scale',
    'ones-stream-copy': 'Stream-Copy',
    'ones-stream-add': 'Stream-Add',
    'ones-stream-full': 'Stream-Full',
    'ones-npb-ep': 'NPB-EP',
    'phases-stream-full': 'Stream-Phase',
    'ones-npb-mg': 'NPB-MG',
    'ones-npb-cg': 'NPB-CG',
    'ones-npb-bt': 'NPB-BT'
}

def get_data_dir(subfolder):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    print(current_dir)
    return os.path.join(current_dir, "experiment_data", subfolder)

def derived_papi(PAPI_data):
    DERIVED = {}
    min_length = min(len(PAPI_data['PAPI_TOT_INS']['instantaneous_value']), len(PAPI_data['PAPI_TOT_CYC']['instantaneous_value']))
    DERIVED['TOT_INS_PER_CYC'] = pd.DataFrame({
        'value': np.array(PAPI_data['PAPI_TOT_INS']['instantaneous_value'][:min_length]) / np.array(PAPI_data['PAPI_TOT_CYC']['instantaneous_value'].iloc[:min_length]),
        'timestamp': np.array(PAPI_data['PAPI_TOT_INS'].time)
    })
    DERIVED['TOT_CYC_PER_INS'] = pd.DataFrame({
        'value': np.array(PAPI_data['PAPI_TOT_CYC']['instantaneous_value']) / np.array(PAPI_data['PAPI_TOT_INS']['instantaneous_value']),
        'timestamp': np.array(PAPI_data['PAPI_TOT_CYC'].time)
    })
    DERIVED['L3_TCM_PER_TCA'] = pd.DataFrame({
        'value': np.array(PAPI_data['PAPI_L3_TCM']['instantaneous_value']) / np.array(PAPI_data['PAPI_L3_TCA']['instantaneous_value']),
        'timestamp': np.array(PAPI_data['PAPI_L3_TCM'].time)
    })  
    DERIVED['TOT_STL_PER_CYC'] = pd.DataFrame({
        'value': np.array(PAPI_data['PAPI_RES_STL']['instantaneous_value']) / np.array(PAPI_data['PAPI_TOT_CYC']['instantaneous_value']),
        'timestamp': np.array(PAPI_data['PAPI_RES_STL'].time)
    })
    
    # Create a mask for non-NaN values across all keys
    mask = ~DERIVED['TOT_INS_PER_CYC']['value'].isna()  # Start with one key to create the mask
    for key in DERIVED:
        mask &= ~DERIVED[key]['value'].isna()  # Combine masks for all keys

    for key in DERIVED:
        DERIVED[key] = DERIVED[key][mask]  # Filter each DataFrame using the combined mask

    return DERIVED

def calculate_power_with_wraparound(current, previous, time_diff, wraparound_value=262143.328850):
    diff = current - previous
    if diff < 0:  # Wraparound detected
        diff = (wraparound_value - previous) + current
    return diff / time_diff

def compute_power(pubEnergy,power_data=None):
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
    # fig,axs = plt.subplots(3,1)
    # axs[0].plot(geopm_power_0['timestamp'], geopm_power_0['power'], label='Node 0')
    # axs[1].plot(geopm_power_1['timestamp'], geopm_power_1['power'], label='Node 1')

    average_power = pd.DataFrame({
        'timestamp': geopm_power_0['timestamp'],  # Use the timestamp from geopm_power_0
        'average_power': [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0['power'], geopm_power_1['power'])]
    })
    average_power['elapsed_time'] = average_power['timestamp'] - average_power['timestamp'].iloc[0]
    # axs[2].plot(average_power['timestamp'], average_power['average_power'], label='Average Power', color='green')
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
            # min_val = min(instantaneous_values)
            # max_val = max(instantaneous_values)
            PAPI[extracted_scope]['instantaneous_value'] = instantaneous_values
            PAPI[extracted_scope]['elapsed_time'] = PAPI[extracted_scope]['time'] - PAPI[extracted_scope]['time'].iloc[0]
    return PAPI

def generate_PCAP(PCAP_data):
    for row in PCAP_data.iterrows():
        if row[1]['time'] == 0:
            PCAP_data = PCAP_data.drop(row[0])


    PCAP_data['elapsed_time'] = PCAP_data['time'] - PCAP_data['time'].iloc[0]
    return PCAP_data

def compute_energy_consump(test_data_frame, start_time=None, end_time=None):
    total_energy = 0
    for i in range(len(test_data_frame['average_power']['elapsed_time']) - 1):
        current_timestamp = test_data_frame['average_power']['timestamp'].iloc[i]
        if start_time is not None and end_time is not None:
            if start_time <= current_timestamp <= end_time:
                total_energy += (
                    test_data_frame['average_power']['average_power'].iloc[i]
                    * (
                        test_data_frame['average_power']['elapsed_time'].iloc[i + 1]
                        - test_data_frame['average_power']['elapsed_time'].iloc[i]
                    )
                )
        else:
            total_energy += (
                test_data_frame['average_power']['average_power'].iloc[i]
                * (
                    test_data_frame['average_power']['elapsed_time'].iloc[i + 1]
                    - test_data_frame['average_power']['elapsed_time'].iloc[i]
                )
            )
    return total_energy

# Generate test_results
DATA_DIR = get_data_dir('plotting_data_ISORC')
root,folders,files = next(os.walk(DATA_DIR))
test_results = {}
for APP in folders:
    if APP != 'ones-stream-full':
        continue
    APP_DIR = os.path.join(DATA_DIR, APP)
    test_results[APP] = {}
    for tar_file in next(os.walk(APP_DIR))[2]:
        test_results[APP][tar_file] = {}
        if tar_file.endswith('.tar'):
            tar_path = os.path.join(APP_DIR, tar_file)
            extract_dir = os.path.join(APP_DIR, tar_file[:-4])  
            
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
        
        pubProgress = pd.read_csv(f'{extract_dir}/progress.csv')
        pubEnergy = pd.read_csv(f'{extract_dir}/energy.csv')
        pubPAPI = pd.read_csv(f'{extract_dir}/papi.csv')
        pubPCAP = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
        pubPower = pd.read_csv(f'{extract_dir}/measured_power.csv')

        test_results[APP][tar_file]['energy'] = pubEnergy
        test_results[APP][tar_file]['power'] = compute_power(pubEnergy)
        test_results[APP][tar_file]['progress'] = measure_progress(pubProgress,test_results[APP][tar_file]['power'])
        test_results[APP][tar_file]['papi'] = collect_papi(pubPAPI)
        test_results[APP][tar_file]['PCAP'] = generate_PCAP(pubPCAP)
        test_results[APP][tar_file]['derived_papi'] = derived_papi(test_results[APP][tar_file]['papi'])   
        test_results[APP][tar_file]['pubPower'] = pubPower

# # Generate test_results_control
# DATA_DIR = get_data_dir('Control_evaluation_ISORC')
# root,folders,files = next(os.walk(DATA_DIR))
# test_results_control = {}
# for APP in folders:
#     APP_DIR = os.path.join(DATA_DIR, APP)
#     test_results_control[APP] = {}
#     for file in next(os.walk(APP_DIR))[2]:
#         print("application:",APP,"file:",file)
#         test_results_control[APP][file] = {}
#         if file.endswith('.tar'):
#             tar_path = os.path.join(APP_DIR, file)
#             extract_dir = os.path.join(APP_DIR, file[:-4])  
            
#             if not os.path.exists(extract_dir):
#                 os.makedirs(extract_dir)
            
#             with tarfile.open(tar_path, 'r') as tar:
#                 tar.extractall(path=extract_dir)
            
#         pubProgress = pd.read_csv(f'{extract_dir}/progress.csv')
#         pubEnergy = pd.read_csv(f'{extract_dir}/energy.csv')
#         pubPAPI = pd.read_csv(f'{extract_dir}/papi.csv')
#         pubPCAP = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
#         pubPower = pd.read_csv(f'{extract_dir}/measured_power.csv')
#         test_results_control[APP][file]['energy'] = pubEnergy
#         test_results_control[APP][file]['power'] = compute_power(pubEnergy,power_data=pubPower)
#         test_results_control[APP][file]['progress'] = measure_progress(pubProgress,test_results_control[APP][file]['power'])
#         test_results_control[APP][file]['papi'] = collect_papi(pubPAPI)
#         test_results_control[APP][file]['PCAP'] = generate_PCAP(pubPCAP)
#         test_results_control[APP][file]['derived_papi'] = derived_papi(test_results_control[APP][file]['papi'])   
#         test_results_control[APP][file]['pubPower'] = pubPower

# Now the plotting code
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

marker_size = 40
high_contrast_colors = plt.cm.YlOrRd(np.linspace(0, 1, 16))
sorted_keys = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
norm = plt.Normalize(vmin=min(sorted_keys), vmax=max(sorted_keys))
smappable = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
smappable.set_array([])

for app in test_results.keys():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 10, "font.weight": "bold", 'axes.labelsize': 'small'})

    grouped_traces = defaultdict(list)
    for trace in test_results[app].keys():
        grouped_traces[test_results[app][trace]['PCAP'].value.mean()].append(trace)

    points = []
    for i, PCAP_traces in enumerate(sorted(grouped_traces.keys())):
        energy_list = []
        exec_time_list = []
        for PCAP_trace in grouped_traces[PCAP_traces]:
            start_time = test_results[app][PCAP_trace]['progress']['progress_sensor'].time.iloc[0]
            end_time = test_results[app][PCAP_trace]['progress']['progress_sensor'].time.iloc[-1]
            energy = compute_energy_consump(test_results[app][PCAP_trace]['power'], start_time=start_time, end_time=end_time) / 1000
            exec_time = test_results[app][PCAP_trace]['progress']['progress_sensor']['elapsed_time'].iloc[-1]
            energy_list.append(energy)
            exec_time_list.append(exec_time)

        # Calculate mean for this group
        energy_mean = np.mean(energy_list)
        exec_time_mean = np.mean(exec_time_list)
        points.append((exec_time_mean, energy_mean, PCAP_traces))

    # Sort points by PCAP value
    points.sort(key=lambda x: x[2])

    # Extract sorted values
    exec_times = [p[0] for p in points]
    energies = [p[1] for p in points]
    pcaps = [p[2] for p in points]

    # Plot the connecting line
    ax.plot(exec_times, energies, color='blue', linewidth=2, label='Trend Line')

    # Plot the points with colors based on PCAP
    colors = [high_contrast_colors[int((p - min(pcaps)) / (max(pcaps) - min(pcaps)) * (len(high_contrast_colors) - 1))] for p in pcaps]
    ax.scatter(exec_times, energies, c=colors, s=marker_size, edgecolors='black', zorder=5)

    ax.set_title(app_name_mapping.get(app, app), fontsize=14, loc='center', fontweight='bold')
    ax.grid(True)
    ax.set_xlabel('Execution Time [s]')
    ax.set_ylabel('Consumed Energy [kJ]')

    # Add colorbar for PCAP
    cbar = fig.colorbar(smappable, ax=ax, shrink=1.0, pad=0.1)
    cbar.set_label('Power Cap [W]', rotation=270, labelpad=20)
    cbar.set_ticks([78.0, 165.0])
    # cbar.set_ticklabels(['78', '165'])

    fig.savefig(f'{OUTPUT_DIR}/{app}_comparison_trend.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
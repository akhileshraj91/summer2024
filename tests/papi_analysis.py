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
plt.rcParams.update({'font.size': 12, "font.weight": "bold", 'axes.labelsize': 'x-large'})

os.chdir(current_working_directory)

exp_type = 'identification' 
experiment_dir = f'{current_working_directory}/experiment_data/{exp_type}/'
# print(experiment_dir)
root, apps, files = next(os.walk(experiment_dir))
num_subs = len(apps)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
fig_pow, axs_pow = plt.subplots(1,1)
# Create a colormap
colormap = plt.cm.get_cmap('tab10', len(apps))

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


    power['geopm_power_0'] = [
        calculate_power_with_wraparound(
            geopm_sensor0['value'][i],
            geopm_sensor0['value'][i-1],
            geopm_sensor0['timestamp'][i] - geopm_sensor0['timestamp'][i-1]
        ) for i in range(1, len(geopm_sensor0))
    ]

    # Apply the same logic to geopm_power_1
    power['geopm_power_1'] = [
        calculate_power_with_wraparound(
            geopm_sensor1['value'][i],
            geopm_sensor1['value'][i-1],
            geopm_sensor1['timestamp'][i] - geopm_sensor1['timestamp'][i-1]
        ) for i in range(1, len(geopm_sensor1))
    ]

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
                # print(fname)
                tar = tarfile.open(cwd + '/' + fname, "r")
                tar.extractall(path=cwd + '/' + fname[:-4])
                tar.close()
    traces[app] = {'directories': next(os.walk(cwd))[1]}
    data = {}
    for trace in traces[app]['directories']:
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
                data[trace][scope]['elapsed_time'] = [t - data[trace][scope]['timestamp'][0] for t in data[trace][scope]['timestamp']]
                data[trace][scope]['average'] = sum(data[trace][scope]['instantaneous_values']) / len(data[trace][scope]['instantaneous_values'])   
                ax = axs[round((count+1)/4), count%2]
                scatter = ax.scatter(data[trace]['PCAP'], data[trace][scope]['average'], label=f'{app}', color=colormap(app_index))
                ax.set_xlabel('PCAP', fontsize=10)  # Adjust font size
                ax.set_ylabel(f'{scope} - Average', fontsize=10)  # Adjust font size
                count += 1
                ax.grid(True)

                # Collect handles and labels for the combined legend
                # print(app,legend_labels)
                if app not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append(f'{app}')
        axs_pow.scatter(data[trace]['PCAP'],np.mean(data[trace]['power_calculated']['average_power']), label=f'{app}', color=colormap(app_index))
        if np.mean(data[trace]['power_calculated']['average_power']) < 0:
            print("pause")
    traces[app]['data'] = data
fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=8)  # Adjust legend position and font size
plt.grid(True)  # Add grid to the main figure
fig.savefig(f'{current_working_directory}/average_plot.pdf')

fig_pow.legend(legend_handles, legend_labels, loc='upper right', ncols=4, fontsize=8)
axs_pow.grid(True)  # Add grid to the power figure
fig_pow.savefig(f'{current_working_directory}/power_relation.pdf')

plt.tight_layout()  # Adjust layout to prevent overlap

# Remove or comment out this line as it's no longer needed
# plt.grid(True)
for scope_name in data[trace].keys():
    if "PAPI" in scope_name:  # Check if "PAPI" is in the scope name
        fig_inst, axs_inst = plt.subplots(2, 3, figsize=(12, 8))
        app_count = 0
        legend_handles = []  # Collect handles for the legend
        legend_labels = []   # Collect labels for the legend
        # Use a colormap with higher color differentiation
        colormap = plt.cm.get_cmap('tab20', len(set(data[trace].keys())))  # Changed to 'tab20'
        color_map = {scope_name: colormap(i) for i, scope_name in enumerate(set(data[trace].keys()))}
        for app in traces.keys():
            for trace in traces[app]['data'].keys():
                # print(app, trace)
                scatter = axs_inst[int(app_count/3), int(app_count%3)].scatter(
                    traces[app]['data'][trace][f"{scope_name}"]['elapsed_time'],
                    traces[app]['data'][trace][scope_name]['instantaneous_values'],
                    label=f"{traces[app]['data'][trace]['PCAP']}"
                )
                if f"{traces[app]['data'][trace]['PCAP']}" not in legend_labels:
                    legend_handles.append(scatter)  # Add scatter handle to legend
                    legend_labels.append(f"{traces[app]['data'][trace]['PCAP']}")  # Add label to legend
            axs_inst[int(app_count/3), int(app_count%3)].set_title(app)
            axs_inst[int(app_count/3), int(app_count%3)].set_xlabel('Elapsed Time', fontsize=10)  # Set x-axis label
            axs_inst[int(app_count/3), int(app_count%3)].set_ylabel('Instantaneous Values', fontsize=10)  # Set y-axis label
            axs_inst[int(app_count/3), int(app_count%3)].grid(True)
            app_count += 1
        
        # Adjust layout to minimize empty space
        fig_inst.suptitle(f'Instantaneous Values for {scope_name}', fontsize=14)  # Add title to the figure
        fig_inst.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.2)  # Adjust margins to leave space for title
        fig_inst.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
        fig_inst.savefig(f'{current_working_directory}/{scope_name}_vs_time.pdf')
        plt.tight_layout()  # Adjust layout to prevent overlap

fig_stat, axs_stat = plt.subplots(2, 3, figsize=(12, 8))
legend_handles = []  # Collect handles for the legend
legend_labels = []   # Collect labels for the legend
for scope_name in data[trace].keys():
    if "PAPI" in scope_name:  # Check if "PAPI" is in the scope name
        app_count = 0
        # Define a color mapping for each scope_name
        color_map = {scope_name: colormap(i) for i, scope_name in enumerate(set(data[trace].keys()))}
        for app in traces.keys():
            for trace in traces[app]['data'].keys():
                # Use the same color for the same scope_name
                color = color_map[scope_name]  # Get color from the mapping
                mean_values = np.mean(traces[app]['data'][trace][scope_name]['instantaneous_values'])
                std_values = np.std(traces[app]['data'][trace][scope_name]['instantaneous_values'])
                fmt = 'o' if "L3_TCA" in scope_name else 'x'  # Set fmt based on scope_name
                scatter = axs_stat[int(app_count/3), int(app_count%3)].errorbar(
                    traces[app]['data'][trace]['PCAP'],
                    mean_values, std_values,
                    label=f"{scope_name}", fmt=fmt, color=color
                )
                if f"{scope_name}" not in legend_labels:
                    legend_handles.append(scatter)  # Add scatter handle to legend
                    legend_labels.append(f"{scope_name}")  # Add label to legend
            axs_stat[int(app_count/3), int(app_count%3)].set_title(app)
            axs_stat[int(app_count/3), int(app_count%3)].set_xlabel('PCAP', fontsize=10)  # Set x-axis label
            axs_stat[int(app_count/3), int(app_count%3)].set_ylabel('Mean and variance of data in one experiment', fontsize=10)  # Set y-axis label
            axs_stat[int(app_count/3), int(app_count%3)].grid(True)
            # axs_stat[int(app_count/3), int(app_count%3)].set_ylim(-100, 100)  # Set y-axis limits for larger range
            app_count += 1
        
fig_stat.suptitle(f'mu and sigma of HC params over each execution with varying PCAPs', fontsize=14)  # Add title to the figure
fig_stat.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.2)  # Adjust margins to leave space for title
fig_stat.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
fig_stat.savefig(f'{current_working_directory}/One_stat_vs_Power.pdf')
plt.tight_layout()  # Adjust layout to prevent overlap


fig_der1, axs_der1 = plt.subplots(2, 3, figsize=(12, 8))
fig_der2, axs_der2 = plt.subplots(2, 3, figsize=(12, 8))
fig_der3, axs_der3 = plt.subplots(2, 3, figsize=(12, 8))
fig_der4, axs_der4 = plt.subplots(2, 3, figsize=(12, 8))


legend_handles = []  # Collect handles for the legend
legend_labels = []   # Collect labels for the legend
# for scope_name in data[trace].keys():
# if "PAPI" in scope_name:  # Check if "PAPI" is in the scope name
app_count = 0
    # Define a color mapping for each scope_name
    # color_map = {scope_name: colormap(i) for i, scope_name in enumerate(set(data[trace].keys()))}
for app in traces.keys():
    for trace in traces[app]['data'].keys():
        # Use the same color for the same scope_name
        # color = color_map[scope_name]  # Get color from the mapping
        # mean_values = np.mean(traces[app]['data'][trace][scope_name]['instantaneous_values'])
        # std_values = np.std(traces[app]['data'][trace][scope_name]['instantaneous_values'])
        # fmt = 'o' if "L3_TCA" in scope_name else 'x'  # Set fmt based on scope_name
        scatter1 = axs_der1[int(app_count/3), int(app_count%3)].scatter(traces[app]['data'][trace]['PAPI_TOT_INS']['elapsed_time'],
            np.array(traces[app]['data'][trace]['PAPI_TOT_INS']['instantaneous_values']) / 
            np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values']),
            label=f"{traces[app]['data'][trace]['PCAP']}"
        )
        scatter2 = axs_der2[int(app_count/3), int(app_count%3)].scatter(traces[app]['data'][trace]['PAPI_TOT_INS']['elapsed_time'],
            np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values']) / 
            np.array(traces[app]['data'][trace]['PAPI_TOT_INS']['instantaneous_values']),
            label=f"{traces[app]['data'][trace]['PCAP']}"
        )
        scatter3 = axs_der3[int(app_count/3), int(app_count%3)].scatter(traces[app]['data'][trace]['PAPI_TOT_INS']['elapsed_time'],
            np.array(traces[app]['data'][trace]['PAPI_L3_TCM']['instantaneous_values']) / 
            np.array(traces[app]['data'][trace]['PAPI_L3_TCA']['instantaneous_values']),
            label=f"{traces[app]['data'][trace]['PCAP']}"
        )
        scatter4 = axs_der4[int(app_count/3), int(app_count%3)].scatter(traces[app]['data'][trace]['PAPI_TOT_INS']['elapsed_time'],
            np.array(traces[app]['data'][trace]['PAPI_RES_STL']['instantaneous_values']) / 
            np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values']),
            label=f"{traces[app]['data'][trace]['PCAP']}"
        )
        if f"{traces[app]['data'][trace]['PCAP']}" not in legend_labels:
            legend_handles.append(scatter1)  # Add scatter handle to legend
            legend_labels.append(f"{traces[app]['data'][trace]['PCAP']}")  # Add label to legend
    axs_der1[int(app_count/3), int(app_count%3)].set_title(app)
    axs_der1[int(app_count/3), int(app_count%3)].set_xlabel('time', fontsize=10)  # Set x-axis label
    axs_der1[int(app_count/3), int(app_count%3)].set_ylabel(r'$\frac{\text{Total Instructions}}{\text{Total Cycles}}$', fontsize=10)  # Set y-axis label
    axs_der1[int(app_count/3), int(app_count%3)].grid(True)
    axs_der2[int(app_count/3), int(app_count%3)].set_title(app)
    axs_der2[int(app_count/3), int(app_count%3)].set_xlabel('time', fontsize=10)  # Set x-axis label
    axs_der2[int(app_count/3), int(app_count%3)].set_ylabel(r'$\frac{\text{Total Cycles}}{\text{Total Instructions}}$', fontsize=10)  # Set y-axis label
    axs_der2[int(app_count/3), int(app_count%3)].grid(True)
    axs_der3[int(app_count/3), int(app_count%3)].set_title(app)
    axs_der3[int(app_count/3), int(app_count%3)].set_xlabel('time', fontsize=10)  # Set x-axis label
    axs_der3[int(app_count/3), int(app_count%3)].set_ylabel(r'$\frac{\text{L3 TCM}}{\text{L3 TCA}}$', fontsize=10)  # Set y-axis label
    axs_der3[int(app_count/3), int(app_count%3)].grid(True)
    axs_der4[int(app_count/3), int(app_count%3)].set_title(app)
    axs_der4[int(app_count/3), int(app_count%3)].set_xlabel('time', fontsize=10)  # Set x-axis label
    axs_der4[int(app_count/3), int(app_count%3)].set_ylabel(r'$\frac{\text{RES STL}}{\text{Total Cycles}}$', fontsize=10)  # Set y-axis label
    axs_der4[int(app_count/3), int(app_count%3)].grid(True)
    # axs_stat[int(app_count/3), int(app_count%3)].set_ylim(-100, 100)  # Set y-axis limits for larger range
    app_count += 1

plt.tight_layout()  # Adjust layout to prevent overlap

# fig_der1.suptitle(f'Total i', fontsize=14)  # Add title to the figure
fig_der1.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.3)  # Adjust margins to leave space for title
fig_der1.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
fig_der1.savefig(f'{current_working_directory}/derived_ins_cyc.pdf')
fig_der2.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.3)  # Adjust margins to leave space for title
fig_der2.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
fig_der2.savefig(f'{current_working_directory}/derived_cyc_ins.pdf')
fig_der3.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.3)  # Adjust margins to leave space for title
fig_der3.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
fig_der3.savefig(f'{current_working_directory}/derived_L3.pdf')
fig_der4.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.3)  # Adjust margins to leave space for title
fig_der4.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
fig_der4.savefig(f'{current_working_directory}/derived_cyc.pdf')



fig_derstat, axs_derstat = plt.subplots(2, 3, figsize=(12, 8))
legend_handles = []  # Collect handles for the legend
legend_labels = []   # Collect labels for the legend
app_count = 0
high_contrast_colors = plt.cm.tab10.colors[:4]

for app in traces.keys():
    for trace in traces[app]['data'].keys():
        # print(legend_labels)
        POI = np.array(traces[app]['data'][trace]['PAPI_TOT_INS']['instantaneous_values']) / np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values'])
        mean = np.mean(POI)
        std = np.std(POI)
        label1=r'$\frac{\text{Total Instructions}}{\text{Total Cycles}}$'
        scatter1 = axs_derstat[int(app_count/3), int(app_count%3)].errorbar(traces[app]['data'][trace]['PCAP'],
            mean, std,
            label=label1, elinewidth=2,fmt='o', color = high_contrast_colors[0]  # Increased error bar size
        )
        POI = np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values']) / np.array(traces[app]['data'][trace]['PAPI_TOT_INS']['instantaneous_values'])
        mean = np.mean(POI)
        std = np.std(POI)
        label2=r'$\frac{\text{Total Cycles}}{\text{Total Instructions}}$'
        scatter2 = axs_derstat[int(app_count/3), int(app_count%3)].errorbar(traces[app]['data'][trace]['PCAP'],
            mean,std,
            label=label2, elinewidth=2,fmt='o', color = high_contrast_colors[1]  # Increased error bar size
        )
        POI = np.array(np.array(traces[app]['data'][trace]['PAPI_L3_TCM']['instantaneous_values']) / np.array(traces[app]['data'][trace]['PAPI_L3_TCA']['instantaneous_values']))
        mean = np.mean(POI)
        std = np.std(POI)
        label3=r'$\frac{\text{L3 TCM}}{\text{L3 TCA}}$'
        scatter3 = axs_derstat[int(app_count/3), int(app_count%3)].errorbar(traces[app]['data'][trace]['PCAP'],
            mean,std,
            label=label3, elinewidth=2,fmt='o', color = high_contrast_colors[2]  # Increased error bar size
        )
        POI = np.array(np.array(traces[app]['data'][trace]['PAPI_RES_STL']['instantaneous_values']) / 
            np.array(traces[app]['data'][trace]['PAPI_TOT_CYC']['instantaneous_values']))
        mean = np.mean(POI)
        std = np.std(POI)
        label4=r'$\frac{\text{RES STL}}{\text{Total Cycles}}$'
        scatter4 = axs_derstat[int(app_count/3), int(app_count%3)].errorbar(traces[app]['data'][trace]['PCAP'],
            mean,std,
            label=label4, elinewidth=2,fmt='o', color = high_contrast_colors[3]  # Increased error bar size
        )
    axs_derstat[int(app_count/3), int(app_count%3)].set_title(app)
    axs_derstat[int(app_count/3), int(app_count%3)].set_xlabel('PCAP', fontsize=10)  # Set x-axis label
    axs_derstat[int(app_count/3), int(app_count%3)].set_ylabel('Mean and variance of derived data for each exp', fontsize=10)  # Set y-axis label
    axs_derstat[int(app_count/3), int(app_count%3)].grid(True)
    app_count += 1
legend_handles = [scatter1,scatter2,scatter3,scatter4]  # Add scatter handle to legend
legend_labels = [label1, label2, label3, label4]  # Add label to legend
# axs_derstat[0,2].legend(loc='upper right', ncol=1, fontsize=8)  # Set legend
 
# Set the legend only once after all error bars are plotted
fig_derstat.suptitle(f'mu and sigma of derived HC params with varying PCAPs', fontsize=14)  # Add title to the figure
fig_derstat.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.06, hspace=0.3, wspace=0.2)  # Adjust margins to leave space for title
fig_derstat.legend(legend_handles, legend_labels, loc='upper right', ncol=1, fontsize=8)
plt.tight_layout()  # Adjust layout to prevent overlap

for app_count in range(2):  # Adjust the range based on the number of rows in axs_derstat
    for ax in axs_derstat[app_count]:
        ax.grid(True)  # Enable grid for each axis

# Generate four high-contrast colors
fig_derstat.savefig(f'{current_working_directory}/derived_One_stat_vs_Power.pdf')

plt.show()
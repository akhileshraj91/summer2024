import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12, "font.weight": "bold", 'axes.labelsize': 'x-large'})
plt.tight_layout()  # Add this line to adjust the layout

# Load the data
present_working_directory = "/Users/akhileshraj/Desktop/summer2024"
filepath = os.path.join(present_working_directory,'SPEC_omp_data')
root, dirs, files = next(os.walk(filepath))

derived_data = ['CPU_ENERGY','DRAM_ENERGY','BOARD_ENERGY','CPU_CYCLES_THREAD','CPU_CYCLES_REFERENCE','CPU_INSTRUCTIONS_RETIRED','GPU_ENERGY','GPU_CORE_ENERGY','MSR::IA32_PMC1:PERFCTR','MSR::IA32_PMC0:PERFCTR']

data = {}

for file in files:
    data[file] = {}
    data[file]['raw'] = pd.read_csv(os.path.join(filepath, file), sep='|', skiprows=5, skipfooter=1, engine='python')

num_columns = len(data[next(iter(data))]['raw'].columns)
# num_rows = (num_columns + 4) // 5  # Calculate number of rows needed
fig, axs = plt.subplots(2, 2, figsize=(30,30))  # Adjust figsize based on number of rows

median_sampling_interval = 10
for key in data.keys():
    sampling_indices = list(range(0, len(data[key]['raw']['TIME']), median_sampling_interval))  # Get indices of sampling times
    sampling_times = [data[key]['raw']['TIME'][k] for k in sampling_indices]  # Get sampling times using indices
    data[key]["smooth"] = pd.DataFrame()
    data[key]["smooth"]["TIME"] = sampling_times
    for column in data[key]['raw'].columns:
        if column in derived_data:
            data[key]['raw'][f"{column}_instantaneous"] = data[key]["raw"][column].diff().fillna(0)

    for column in data[key]['raw'].columns:
        if column not in ['TIME', 'REGION_HASH', 'REGION_TYPE', 'REGION_ID', 'REGION_NAME', 'REGION_HINT']:
            # Calculate the median for each interval defined by sampling_indices
            medians = []
            for i in range(len(sampling_indices) - 1):
                # Get the slice of the data for the current interval
                interval_data = data[key]['raw'][column].iloc[sampling_indices[i]:sampling_indices[i + 1]]
                # Calculate the median and append to the list
                medians.append(interval_data.median())
            # Assign the medians to the smooth DataFrame
            data[key]["smooth"][column] = pd.Series(medians)
        
        

num_columns = len(data[next(iter(data))]['smooth'].columns)
num_rows = num_columns  // 5
fig_smooth, axs_smooth = plt.subplots(2, 2, figsize=(25, 5 * num_rows))  # Adjust figsize based on number of rows


for key in data.keys():
    axs[0,0].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_POWER'], label=key, s=14)
    # axs[0,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_ENERGY'], label=key, s=10)
    # axs[0,2].scatter(data[key]['raw']['TIME'], data[key]['raw']['DRAM_ENERGY'], label=key, s=10)
    axs[0,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['DRAM_POWER'], label=key, s=14)
    # axs[0,4].scatter(data[key]['raw']['TIME'], data[key]['raw']['BOARD_ENERGY'], label=key, s=10)
    # axs[1,0].scatter(data[key]['raw']['TIME'], data[key]['raw']['BOARD_POWER'], label=key, s=10)
    # axs[1,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_FREQUENCY_STATUS'], label=key, s=10)
    # axs[1,2].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_CYCLES_THREAD'], label=key, s=10)
    # axs[1,3].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_CYCLES_REFERENCE'], label=key, s=10)
    # axs[1,4].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_CORE_TEMPERATURE'], label=key, s=10)
    # axs[1,0].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_INSTRUCTIONS_RETIRED'], label=key, s=10)
    # axs[2,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_UNCORE_FREQUENCY_STATUS'], label=key, s=10)
    # axs[2,2].scatter(data[key]['raw']['TIME'], data[key]['raw']['GPU_ENERGY'], label=key, s=10)
    # axs[2,3].scatter(data[key]['raw']['TIME'], data[key]['raw']['GPU_CORE_POWER'], label=key, s=10)
    # axs[2,4].scatter(data[key]['raw']['TIME'], data[key]['raw']['GPU_CORE_ENERGY'], label=key, s=10)
    # axs[1,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['MSR::IA32_PMC1:PERFCTR'], label=key, s=10)
    # axs[1,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['MSR::IA32_PMC0:PERFCTR'], label=key, s=10)
    # axs[3,2].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_ENERGY_instantaneous'], label='CPU Power Instantaneous', s=10)
    # axs[3,2].set_title('CPU Power Instantaneous')
    # axs[3,3].scatter(data[key]['raw']['TIME'], data[key]['raw']['GPU_CORE_ENERGY_instantaneous'], label='GPU Core Power Instantaneous', s=10)
    # axs[3,3].set_title('GPU Core Power Instantaneous')  
    # axs[3,4].scatter(data[key]['raw']['TIME'], data[key]['raw']['DRAM_ENERGY_instantaneous'], label='DRAM Power Instantaneous', s=10)
    # axs[3,4].set_title('DRAM Power Instantaneous')
    # axs[4,0].scatter(data[key]['raw']['TIME'], data[key]['raw']['BOARD_ENERGY_instantaneous'], label='BORAD POWER Instantaneous', s=10)
    # axs[4,0].set_title('BOARD POWER Instantaneous')
    # axs[4,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_CYCLES_THREAD_instantaneous'], label='CPU Cycles Thread Instantaneous', s=10)
    # axs[4,1].set_title('CPU Cycles Thread Instantaneous')
    # axs[4,2].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_CYCLES_REFERENCE_instantaneous'], label='CPU Cycles Reference Instantaneous', s=10)
    # axs[4,2].set_title('CPU Cycles Reference Instantaneous')
    axs[1,0].scatter(data[key]['raw']['TIME'], data[key]['raw']['CPU_INSTRUCTIONS_RETIRED_instantaneous'], label='CPU Instructions Retired Instantaneous', s=14)    
    axs[1,0].set_title('CPU Instructions Retired Instantaneous')
    # axs[4,4].scatter(data[key]['raw']['TIME'], data[key]['raw']['GPU_ENERGY_instantaneous'], label='GPU Energy Instantaneous', s=10)
    # axs[4,4].set_title('GPU Energy Instantaneous')
    axs[1,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['MSR::IA32_PMC1:PERFCTR_instantaneous'], label='L3_IA32_PMC1:PERFCTR Instantaneous', s=14)
    axs[1,1].set_title('L3 Misses Instantaneous')
    # axs[1,1].scatter(data[key]['raw']['TIME'], data[key]['raw']['MSR::IA32_PMC0:PERFCTR_instantaneous'], label='L3_IA32_PMC0:PERFCTR Instantaneous', s=14)
    # axs[1,1].set_title('L3 References')
    # axs[5,2].scatter(data[key]['raw']['TIME'].iloc[0:100], data[key]['raw']['GPU_CORE_POWER'].iloc[0:100], label='GPU_CORE_POWER', s=10)
    # axs[5,2].scatter(data[key]['raw']['TIME'].iloc[0:100], data[key]['raw']['CPU_POWER'].iloc[0:100], label='CPU_POWER', s=10)
    # axs[5,2].scatter(data[key]['raw']['TIME'].iloc[0:100], data[key]['raw']['DRAM_POWER'].iloc[0:100], label='DRAM_POWER', s=10)
    # axs[5,2].legend(loc='lower left', fontsize='small', title='Legend')
    # axs[5,2].set_title('Power Comparison Zoomed (raw)')
    
# Set titles for each subplot
axs[0, 0].set_title('CPU Power')
# axs[0, 1].set_title('CPU Energy')
# axs[0, 2].set_title('DRAM Energy')
axs[0, 1].set_title('DRAM Power')
# axs[0, 4].set_title('Board Energy')
# axs[1, 0].set_title('Board Power')
# axs[1, 1].set_title('CPU Frequency Status')
# axs[1, 2].set_title('CPU Cycles (Thread)')
# axs[1, 3].set_title('CPU Cycles (Reference)')
# axs[1, 4].set_title('CPU Core Temperature')
axs[1, 0].set_title('CPU Instructions Retired')
# axs[2, 1].set_title('CPU Uncore Frequency Status')
# axs[2, 2].set_title('GPU Energy')
# axs[2, 3].set_title('GPU Core Power')
# axs[2, 4].set_title('GPU Core Energy')
axs[1, 1].set_title('L3_Misses')
# axs[3, 1].set_title('L3_References')

# Turn on the grid for all subplots
for ax in axs.flat:
    ax.grid(True)

# Add a single legend for the entire figure, using unique labels
handles, labels = axs[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(1, 0), fontsize='small', title='Legend')

# Add scatter plots for axs_smooth using data[key]['smooth']
for key in data.keys():
    axs_smooth[0,0].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_POWER'], label=key, s=14)
    axs_smooth[0,0].set_title('CPU Power (Smooth)')  # Set title for CPU Power
    # axs_smooth[0,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_ENERGY'], label=key, s=10)
    # axs_smooth[0,1].set_title('CPU Energy (Smooth)')  # Set title for CPU Energy
    # axs_smooth[0,2].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['DRAM_ENERGY'], label=key, s=10)
    # axs_smooth[0,2].set_title('DRAM Energy (Smooth)')  # Set title for DRAM Energy
    axs_smooth[0,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['DRAM_POWER'], label=key, s=14)
    axs_smooth[0,1].set_title('DRAM Power (Smooth)')  # Set title for DRAM Power
    # axs_smooth[0,4].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['BOARD_ENERGY'], label=key, s=10)
    # axs_smooth[0,4].set_title('Board Energy (Smooth)')  # Set title for Board Energy
    # axs_smooth[1,0].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['BOARD_POWER'], label=key, s=10)
    # axs_smooth[1,0].set_title('Board Power (Smooth)')  # Set title for Board Power
    # axs_smooth[1,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_FREQUENCY_STATUS'], label=key, s=10)
    # axs_smooth[1,1].set_title('CPU Frequency Status (Smooth)')  # Set title for CPU Frequency Status
    # axs_smooth[1,2].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_CYCLES_THREAD'], label=key, s=10)
    # axs_smooth[1,2].set_title('CPU Cycles Thread (Smooth)')  # Set title for CPU Cycles Thread
    # axs_smooth[1,3].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_CYCLES_REFERENCE'], label=key, s=10)
    # axs_smooth[1,3].set_title('CPU Cycles Reference (Smooth)')  # Set title for CPU Cycles Reference
    # axs_smooth[1,4].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_CORE_TEMPERATURE'], label=key, s=10)
    # axs_smooth[1,4].set_title('CPU Core Temperature (Smooth)')  # Set title for CPU Core Temperature
    # axs_smooth[1,0].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_INSTRUCTIONS_RETIRED'], label=key, s=10)
    # axs_smooth[1,0].set_title('CPU Instructions Retired (Smooth)')  # Set title for CPU Instructions Retired
    # axs_smooth[2,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_UNCORE_FREQUENCY_STATUS'], label=key, s=10)
    # axs_smooth[2,1].set_title('CPU Uncore Frequency Status (Smooth)')  # Set title for CPU Uncore Frequency Status
    # axs_smooth[2,2].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['GPU_ENERGY'], label=key, s=10)
    # axs_smooth[2,2].set_title('GPU Energy (Smooth)')  # Set title for GPU Energy
    # axs_smooth[2,3].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['GPU_CORE_POWER'], label=key, s=10)
    # axs_smooth[2,3].set_title('GPU Core Power (Smooth)')  # Set title for GPU Core Power
    # axs_smooth[2,4].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['GPU_CORE_ENERGY'], label=key, s=10)
    # axs_smooth[2,4].set_title('GPU Core Energy (Smooth)')  # Set title for GPU Core Energy
    # axs_smooth[1,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['MSR::IA32_PMC1:PERFCTR'], label=key, s=10)
    # axs_smooth[1,1].set_title('MSR::IA32_PMC1:PERFCTR (Smooth)')  # Set title for MSR::IA32_PMC1:PERFCTR
    # axs_smooth[1,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['MSR::IA32_PMC0:PERFCTR'], label=key, s=10)
    # axs_smooth[1,1].set_title('MSR::IA32_PMC0:PERFCTR (Smooth)')  # Set title for MSR::IA32_PMC0:PERFCTR
    # axs_smooth[3,2].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_ENERGY_instantaneous'], label='CPU Power Instantaneous (Smooth)', s=10)
    # axs_smooth[3,2].set_title('CPU Power Instantaneous (Smooth)')  # Set title for CPU Power Instantaneous
    # axs_smooth[3,3].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['GPU_CORE_ENERGY_instantaneous'], label='GPU Core Power Instantaneous (Smooth)', s=10)
    # axs_smooth[3,3].set_title('GPU Core Power Instantaneous (Smooth)')  
    # axs_smooth[3,4].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['DRAM_ENERGY_instantaneous'], label='DRAM Power Instantaneous (Smooth)', s=10)
    # axs_smooth[3,4].set_title('DRAM Power Instantaneous (Smooth)')
    # axs_smooth[4,0].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['BOARD_ENERGY_instantaneous'], label='BOARD POWER Instantaneous (Smooth)', s=10)
    # axs_smooth[4,0].set_title('BOARD POWER Instantaneous (Smooth)')
    # axs_smooth[4,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_CYCLES_THREAD_instantaneous'], label='CPU Cycles Thread Instantaneous (Smooth)', s=10)
    # axs_smooth[4,1].set_title('CPU Cycles Thread Instantaneous (Smooth)')
    # axs_smooth[4,2].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_CYCLES_REFERENCE_instantaneous'], label='CPU Cycles Reference Instantaneous (Smooth)', s=10)
    # axs_smooth[4,2].set_title('CPU Cycles Reference Instantaneous (Smooth)')
    axs_smooth[1,0].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['CPU_INSTRUCTIONS_RETIRED_instantaneous'], label='CPU Instructions Retired Instantaneous (Smooth)', s=14)    
    axs_smooth[1,0].set_title('CPU Instructions Retired Instantaneous (Smooth)')
    # axs_smooth[4,4].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['GPU_ENERGY_instantaneous'], label='GPU Energy Instantaneous (Smooth)', s=10)
    # axs_smooth[4,4].set_title('GPU Energy Instantaneous (Smooth)')
    axs_smooth[1,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['MSR::IA32_PMC1:PERFCTR_instantaneous'], label='L3_IA32_PMC1:PERFCTR Instantaneous (Smooth)', s=14)
    axs_smooth[1,1].set_title('L3 Misses Instantaneous (Smooth)')
    # axs_smooth[1,1].scatter(data[key]['smooth']['TIME'], data[key]['smooth']['MSR::IA32_PMC0:PERFCTR_instantaneous'], label='L3_IA32_PMC0:PERFCTR Instantaneous (Smooth)', s=14)
    # axs_smooth[1,1].set_title('L3 References (Smooth)')
    # axs_smooth[5,2].scatter(data[key]['smooth']['TIME'].iloc[0:100], data[key]['smooth']['GPU_CORE_POWER'].iloc[0:100], label='GPU_CORE_POWER (Smooth)', s=10)
    # axs_smooth[5,2].scatter(data[key]['smooth']['TIME'].iloc[0:100], data[key]['smooth']['CPU_POWER'].iloc[0:100], label='CPU_POWER (Smooth)', s=10)
    # axs_smooth[5,2].scatter(data[key]['smooth']['TIME'].iloc[0:100], data[key]['smooth']['DRAM_POWER'].iloc[0:100], label='DRAM_POWER (Smooth)', s=10)
    # axs_smooth[5,2].legend(loc='lower left', fontsize='small', title='Legend')
    # axs_smooth[5,2].set_title('Power Comparison Zoomed (Smooth)')

for ax in axs_smooth.flat:
    ax.grid(True)

# Add a single legend for the entire figure, using unique labels
handles, labels = axs_smooth[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(1, 0), fontsize='small', title='Legend')

# Add this code to save the figures
fig.savefig(os.path.join(present_working_directory, 'SPEC_data_raw.pdf'), bbox_inches='tight')
fig_smooth.savefig(os.path.join(present_working_directory, 'SPEC_data_smooth.pdf'), bbox_inches='tight')


# %%
import numpy as np
from ruamel.yaml import YAML
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon
from IPython.display import clear_output
import torch
import sys
import pandas as pd
import tarfile
import math
import warnings
import os
import csv
import gym
import argparse
import datetime
from IPython.display import clear_output, display
import json

warnings.filterwarnings('ignore')


# helper: data dir
def get_data_dir(subfolder):
    current_dir = os.getcwd()
    print(current_dir)
    return os.path.join(current_dir, "experiment_data", f"{subfolder}")


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def derived_papi(PAPI_data):
    DERIVED = {}
    DERIVED['TOT_INS_PER_CYC'] = pd.DataFrame({
        'value': np.array(PAPI_data['PAPI_TOT_INS']['instantaneous_value']) / np.array(PAPI_data['PAPI_TOT_CYC']['instantaneous_value']),
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

    mask = ~DERIVED['TOT_INS_PER_CYC']['value'].isna()
    for key in DERIVED:
        mask &= ~DERIVED[key]['value'].isna()

    for key in DERIVED:
        DERIVED[key] = DERIVED[key][mask]

    return DERIVED


def calculate_power_with_wraparound(current, previous, time_diff, wraparound_value=262143.328850):
    diff = current - previous
    if diff < 0:
        diff = (wraparound_value - previous) + current
    return diff / time_diff


def compute_measured_power(pubPower, all_data):
    PCAP_data = all_data['PCAP']
    first_PCAP_point = PCAP_data['timestamp'].iloc[0]
    new_pubPower = pubPower[pubPower['time'] >= first_PCAP_point]
    elapsed_time = new_pubPower['time'] - new_pubPower['time'].iloc[0]
    new_pubPower['elapsed_time'] = elapsed_time
    new_pubPower.rename(columns={'time': 'timestamp'}, inplace=True)
    return new_pubPower


def compute_power(pubEnergy):
    power = {}
    geopm_sensor0 = geopm_sensor1 = pd.DataFrame({'timestamp': [], 'value': []})
    for i, row in pubEnergy.iterrows():
        if i % 2 == 0:
            geopm_sensor0 = pd.concat([geopm_sensor0, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)
        else:
            geopm_sensor1 = pd.concat([geopm_sensor1, pd.DataFrame([{'timestamp': row['time'], 'value': row['value']}])], ignore_index=True)

    power['geopm_power_0'] = pd.DataFrame({
        'timestamp': geopm_sensor0['timestamp'][1:],
        'power': [
            calculate_power_with_wraparound(
                geopm_sensor0['value'][i],
                geopm_sensor0['value'][i - 1],
                geopm_sensor0['timestamp'][i] - geopm_sensor0['timestamp'][i - 1]
            ) for i in range(1, len(geopm_sensor0))
        ]
    })

    power['geopm_power_1'] = pd.DataFrame({
        'timestamp': geopm_sensor1['timestamp'][1:],
        'power': [
            calculate_power_with_wraparound(
                geopm_sensor1['value'][i],
                geopm_sensor1['value'][i - 1],
                geopm_sensor1['timestamp'][i] - geopm_sensor1['timestamp'][i - 1]
            ) for i in range(1, len(geopm_sensor1))
        ]
    })

    min_length = min(len(power['geopm_power_0']), len(power['geopm_power_1']))
    geopm_power_0 = power['geopm_power_0'][:min_length]
    geopm_power_1 = power['geopm_power_1'][:min_length]

    average_power = pd.DataFrame({
        'timestamp': geopm_power_0['timestamp'],
        'average_power': [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0['power'], geopm_power_1['power'])]
    })
    average_power['elapsed_time'] = average_power['timestamp'] - average_power['timestamp'].iloc[0]
    power['average_power'] = average_power
    return power

    average_power = pd.DataFrame({
        'timestamp': geopm_power_0['timestamp'],  # Use the timestamp from geopm_power_0
        'average_power': [(p0 + p1) / 2 for p0, p1 in zip(geopm_power_0['power'], geopm_power_1['power'])]
    })
    average_power['elapsed_time'] = average_power['timestamp'] - average_power['timestamp'].iloc[0]
    power['average_power'] = average_power
    return power

def measure_progress(progress_data, all_data):
    power_data = all_data['measured_power']
    PCAP_data = all_data['PCAP']
    Progress_DATA = {} 
    progress_sensor = pd.DataFrame(progress_data)
    first_PCAP_point = PCAP_data['timestamp'].iloc[0]
    new_progress_sensor = progress_sensor[progress_sensor['time'] >= first_PCAP_point]
    new_power_data = power_data[power_data['timestamp'] >= first_PCAP_point]
    first_sensor_point = min(new_power_data['timestamp'].iloc[0], new_progress_sensor['time'].iloc[0])
    new_progress_sensor['elapsed_time'] = new_progress_sensor['time'] - first_sensor_point  
    performance_elapsed_time = new_progress_sensor.elapsed_time
    frequency_values = [
        progress_data['value'].iloc[t] / (performance_elapsed_time.iloc[t] - performance_elapsed_time.iloc[t-1]) for t in range(1, len(performance_elapsed_time))
    ]
    frequency_values = [0] + frequency_values  
    new_progress_sensor['frequency'] = frequency_values
    upsampled_timestamps= PCAP_data['timestamp']
    # progress_frequency_median = pd.DataFrame({'median': np.nanmedian(new_progress_sensor['frequency'].where(new_progress_sensor['time'] <= upsampled_timestamps.iloc[0])), 'timestamp': upsampled_timestamps.iloc[0]}, index=[0])
    progress_frequency_median = pd.DataFrame()
    for t in range(1, len(upsampled_timestamps)):
        progress_frequency_median = pd.concat([progress_frequency_median, pd.DataFrame({'median': [np.nanmedian(new_progress_sensor['frequency'].where((new_progress_sensor['time'] >= upsampled_timestamps.iloc[t-1]) & (new_progress_sensor['time'] <= upsampled_timestamps.iloc[t])))],
        'timestamp': [upsampled_timestamps.iloc[t]]})], ignore_index=True)
    progress_frequency_median['elapsed_time'] = progress_frequency_median['timestamp'] - progress_frequency_median['timestamp'].iloc[0]
    Progress_DATA['progress_sensor'] = new_progress_sensor
    Progress_DATA['progress_frequency_median'] = progress_frequency_median
    Progress_DATA['progress_sensor'].rename(columns={'time': 'timestamp'}, inplace=True)
    return Progress_DATA

def collect_papi(PAPI_data,all_data):
    PCAP_data = all_data['PCAP']
    first_PCAP_point = PCAP_data['timestamp'].iloc[0]
    new_PAPI_data = PAPI_data[PAPI_data['time'] >= first_PCAP_point]
    PAPI = {}
    for scope in new_PAPI_data['scope'].unique():
        scope_parts = scope.split('.')
        if len(scope_parts) > 4:  
            extracted_scope = scope_parts[3]
            PAPI[extracted_scope] = new_PAPI_data[new_PAPI_data['scope'] == scope]
            instantaneous_values = [0] + [PAPI[extracted_scope]['value'].iloc[k] - PAPI[extracted_scope]['value'].iloc[k-1] for k in range(1,len(PAPI[extracted_scope]))]
            PAPI[extracted_scope]['instantaneous_value'] = instantaneous_values
            PAPI[extracted_scope]['elapsed_time'] = PAPI[extracted_scope]['time'] - PAPI[extracted_scope]['time'].iloc[0]
    return PAPI


def generate_PCAP(PCAP_data):
    for row in PCAP_data.iterrows():
        if row[1]['time'] == 0:
            PCAP_data = PCAP_data.drop(row[0])


    PCAP_data['elapsed_time'] = PCAP_data['time'] - PCAP_data['time'].iloc[0]
    PCAP_data.rename(columns={'time': 'timestamp'}, inplace=True)
    return PCAP_data


# %%
DATA_DIR = get_data_dir("training_data")
csv_file_path = os.path.join(DATA_DIR, 'training_dataset.csv')
training_data = {}
MAX_PROGRESS = {}

# If a prebuilt training_dataset.csv exists, skip raw extraction and processing
if os.path.exists(csv_file_path):
    print(f"Found existing {csv_file_path}, skipping raw extraction.")
    root, folders, files = next(os.walk(DATA_DIR))
else:
    root,folders,files = next(os.walk(DATA_DIR))
    for APP in folders:
        APP_DIR = os.path.join(DATA_DIR, APP)
        training_data[APP] = {}
        for file in next(os.walk(APP_DIR))[2]:
            training_data[APP][file] = {}
            if file.endswith('.tar'):
                tar_path = os.path.join(APP_DIR, file)
                extract_dir = os.path.join(APP_DIR, file[:-4])  
                
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir)
                
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=extract_dir)
                
            pubProgress = pd.read_csv(f'{extract_dir}/progress.csv')
            pubEnergy = pd.read_csv(f'{extract_dir}/energy.csv')
            pubPAPI = pd.read_csv(f'{extract_dir}/papi.csv')
            pubPCAP = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
            pubPower = pd.read_csv(f'{extract_dir}/measured_power.csv')
            # training_data[APP][file]['power'] = compute_power(pubEnergy)
            training_data[APP][file]['PCAP'] = generate_PCAP(pubPCAP)
            training_data[APP][file]['measured_power'] = compute_measured_power(pubPower, training_data[APP][file])
            training_data[APP][file]['progress'] = measure_progress(pubProgress,training_data[APP][file])
            training_data[APP][file]['papi'] = collect_papi(pubPAPI,training_data[APP][file])
            training_data[APP][file]['derived_papi'] = derived_papi(training_data[APP][file]['papi'])   
        MAX_PROGRESS[APP] = training_data[APP][file]['progress']['progress_frequency_median']['median'].max()
    APP_DIR = os.path.join(DATA_DIR, APP)
    training_data[APP] = {}
    for file in next(os.walk(APP_DIR))[2]:
        training_data[APP][file] = {}
        if file.endswith('.tar'):
            tar_path = os.path.join(APP_DIR, file)
            extract_dir = os.path.join(APP_DIR, file[:-4])  
            
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
        pubProgress = pd.read_csv(f'{extract_dir}/progress.csv')
        pubEnergy = pd.read_csv(f'{extract_dir}/energy.csv')
        pubPAPI = pd.read_csv(f'{extract_dir}/papi.csv')
        pubPCAP = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
        pubPower = pd.read_csv(f'{extract_dir}/measured_power.csv')
        # training_data[APP][file]['power'] = compute_power(pubEnergy)
        training_data[APP][file]['PCAP'] = generate_PCAP(pubPCAP)
        training_data[APP][file]['measured_power'] = compute_measured_power(pubPower, training_data[APP][file])
        training_data[APP][file]['progress'] = measure_progress(pubProgress,training_data[APP][file])
        training_data[APP][file]['papi'] = collect_papi(pubPAPI,training_data[APP][file])
        training_data[APP][file]['derived_papi'] = derived_papi(training_data[APP][file]['papi'])   
    MAX_PROGRESS[APP] = training_data[APP][file]['progress']['progress_frequency_median']['median'].max()


# %%
T_S = 1
ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
exec_steps = 10000    
TOTAL_ACTIONS = len(ACTIONS)                                                                                                  # Total clock cycles needed for the execution of program.
ACTION_MIN = min(ACTIONS)                                                                                                    # Minima of control space
ACTION_MAX = max(ACTIONS)                                                                                                     # Maxima of control space
ACT_MID = ACTION_MIN + (ACTION_MAX - ACTION_MIN) / 2                                                                    # Midpoint of the control space to compute the normalized action space                     
OBS_MIN = np.zeros((5,))  
OBS_MAX = np.array([300,165,1,1,1])                                                                                 # Minima of observation space
OBS_MID = OBS_MIN + (OBS_MAX - OBS_MIN) / 2
EXEC_ITERATIONS = 10000
TOTAL_OBS = OBS_MAX - OBS_MIN
OBS_ONEHOT = 'onehot'
OBS_RANDOM = 'random'
OBS_SMOOTH = 'smooth'

def scale_reward_uniform(r, r_min=1.4666, r_max=7.7000, target_min=-10, target_max=1):
    return target_min + (r - r_min) * (target_max - target_min) / (r_max - r_min)


class SYS(object):
    def __init__(self,observation_type=OBS_ONEHOT,dim_obs=1,teps=0.0):
        super(SYS,self).__init__()

        self.num_actions = TOTAL_ACTIONS
        self.action_space = gym.spaces.Discrete(len(ACTIONS))  
        self.actions = ACTIONS  
        self.observation_space = gym.spaces.Box(low=OBS_MIN, high=OBS_MAX, shape=(5,), dtype=np.float32)  

    
    def reward(self, s, a, ns, measured_power, max_progress):
        """ 
        Returns the reward (float)
        """
        if ns > 0:
            # self.current_step += ns
            # reward = - 5*a
            # print("="*10,ns,a,s,measured_power)
            # reward = ns/(((a**2)/(measured_power+1))+measured_power+1) # Check the behaviour across the states
            # reward = 0.001*ns**2/(a+1)
            # reward = 0.001*ns**2/(((a**2)/(measured_power+1))+measured_power+1) # Check the behaviour across the states
            # reward = 5*a
            # reward = 0.001*(ns**2/a)*(measured_power)
            # reward = 0.01*ns/(measured_power+a)
            # reward = 0.00001*ns**3*a/(measured_power**2 + 1)
            # reward = 0.0001*ns + 1/(measured_power)
            # reward = (((10*ns/max_progress)**3)/measured_power)
            # r_min = 1.46
            # r_max = 7.70

            reward = ns ** 3 / (measured_power + 1e-5)
            # reward = 2*scale_reward_uniform(reward)
            # reward = -1 + 11 * ((reward - r_min) / (r_max - r_min))
            # reward = np.clip(reward, -10, 1)  # in case of outliers

        else:
            reward = -100
        return reward

weighting_only = False
dataset_composition = 'random'
dataset_size = 1000
env_type = 'random'
env = SYS(observation_type=env_type, dim_obs=8, teps=0)

# %%
def get_roi_data(df, time_column, start_time, end_time):
    return df[(df[time_column] > start_time) & (df[time_column] <= end_time)]


# %%
def get_state(td, app, trace, start_time, end_time):
    ROI_progress = get_roi_data(td[app][trace]['progress']['progress_frequency_median'], 'timestamp', start_time, end_time)
    ROI_measured_power = get_roi_data(td[app][trace]['measured_power'], 'timestamp', start_time, end_time)
    TOT_INS_PER_CYC = get_roi_data(td[app][trace]['derived_papi']['TOT_INS_PER_CYC'], 'timestamp', start_time, end_time)
    L3_TCM_PER_TCA = get_roi_data(td[app][trace]['derived_papi']['L3_TCM_PER_TCA'], 'timestamp', start_time, end_time)
    TOT_STL_PER_CYC = get_roi_data(td[app][trace]['derived_papi']['TOT_STL_PER_CYC'], 'timestamp', start_time, end_time)
    return (
        ROI_progress['median'].mean() if not ROI_progress.empty else 0,
        ROI_measured_power['value'].mean() if not ROI_measured_power.empty else 0,
        TOT_INS_PER_CYC['value'].mean() if not TOT_INS_PER_CYC.empty else 0,
        L3_TCM_PER_TCA['value'].mean() if not L3_TCM_PER_TCA.empty else 0,
        TOT_STL_PER_CYC['value'].mean() if not TOT_STL_PER_CYC.empty else 0,
    )


# %%
training_dataset = []
t1 = float('-inf')
for app in training_data:
    # if "ones-stream-full" in app:
    #     print(app)
    for trace in training_data[app]:
        pcap_data = training_data[app][trace]['PCAP']
        for i, row in pcap_data.iterrows():
            t2 = row['timestamp']
            state = get_state(training_data, app, trace, t1, t2)
            if i + 1 < len(pcap_data):
                t3 = pcap_data.iloc[i + 1]['timestamp']
                next_state = get_state(training_data, app, trace, t2, t3)
            else:
                next_state = state  # Use current state if it's the last row
            action = row['value']  # Assuming PCAP is in the 'value' column

            reward = env.reward(state[0], action, next_state[0], next_state[1], MAX_PROGRESS[app])
            
            # Add to training dataset
            if not np.isnan(next_state).any():
                training_dataset.append((app, state, action, reward, next_state))
            
            t1 = t2
print(len(training_dataset))
MAX_REWARD = {}
MIN_REWARD = {}
for app in training_data:
    # if app == "ones-stream-full":
    #     print(app)
    reward_entries = [entry[3] for entry in training_dataset if entry[0] == app]
    MAX_REWARD[app] = max(reward_entries)
    MIN_REWARD[app] = min(reward_entries)
# ## debug
# min_reward = MIN_REWARD['ones-stream-full']
# min_indices = [i for i, entry in enumerate(training_dataset) if entry[0] == 'ones-stream-full' and entry[3] == min_reward]
# min_index = min_indices[0] if min_indices else None
# print("Index of minimum reward for ones-stream-full:", min_index)
# print("Entry:", training_dataset[min_index])
# ## debug
normalized_training_dataset = []
for entry in training_dataset:
    app = entry[0]
    reward = entry[3]
    if MAX_REWARD[app] == MIN_REWARD[app]:
        normalized_reward = 0.0
        scaled_reward = -5
    else:
        normalized_reward = (reward - MIN_REWARD[app]) / (MAX_REWARD[app] - MIN_REWARD[app])
        scaled_reward = -5 + normalized_reward * 6
    new_entry = entry[:3] + (scaled_reward,) + entry[4:]
    normalized_training_dataset.append(new_entry)

training_dataset = normalized_training_dataset
    
csv_file_name = 'training_dataset.csv'
csv_file_path = os.path.join(DATA_DIR, csv_file_name)

# Only write the CSV if we did not find an existing file (i.e., we built training_dataset here)
if not os.path.exists(csv_file_path) and len(training_dataset) > 0:
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(['App','Progress', 'Power', 'TOT_INS_PER_CYC', 'L3_TCM_PER_TCA', 'TOT_STL_PER_CYC', 
                             'Action', 'Reward', 
                             'Next_Progress', 'Next_Power', 'Next_TOT_INS_PER_CYC', 'Next_L3_TCM_PER_TCA', 'Next_STL_PER_CYC'])
        
        for app, state, action, reward, next_state in training_dataset:
            if not (np.isnan(state).any() or np.isnan(action) or np.isnan(reward) or np.isnan(next_state).any()): 
                    # or (action-next_state[1]) < 0): 
                    # or (action-next_state[1]) >= 10):
                state = np.array(state)
                next_state = np.array(next_state)
                if np.all((state >= 0) & (state <= 300)) and np.all((next_state >= 0) & (next_state <= 300)):
                    row = [app] + list(state) + [action, reward] + list(next_state)
                    csv_writer.writerow(row)

    print(f"Training dataset has been saved to {csv_file_path}")


csv_file_path = f'{DATA_DIR}/training_dataset.csv'

loaded_data = pd.read_csv(csv_file_path)
print(len(loaded_data))


# %%
# Define the CSV file path
csv_file_path = f'{DATA_DIR}/training_dataset.csv'

# Load the CSV into a DataFrame
loaded_data = pd.read_csv(csv_file_path)
print(len(loaded_data))

# %%
# Convert the DataFrame back to the original format (list of tuples)
training_dataset_loaded = [
    (
        tuple(row[1:6]),  # State
        row[6],          # Action
        row[7],          # Reward
        tuple(row[8:])   # Next State
    )
    for row in loaded_data.values
]

print(len(training_dataset_loaded))


# %%
def get_tensors(list_of_tensors, list_of_indices):
    """Return s,a,ns,r arrays for the given indices from a dataset stored as
    (state, action, reward, next_state) tuples.
    States are returned as float32 numpy arrays; actions and rewards as numpy arrays.
    """
    s, a, ns, r = [], [], [], []
    for idx in list_of_indices:
        row = list_of_tensors[idx]
        s.append(np.array(row[0], dtype=np.float32))
        a.append(row[1])
        r.append(row[2])
        ns.append(np.array(row[3], dtype=np.float32))
    return np.array(s, dtype=np.float32), np.array(a), np.array(ns, dtype=np.float32), np.array(r, dtype=np.float32)


class FCNetwork(torch.nn.Module):
    def __init__(self, env, layers=[20, 20]):
        super(FCNetwork, self).__init__()
        dim_input = 5
        dim_output = env.num_actions
        net_layers = []
        dim = dim_input
        for layer_size in layers:
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(torch.nn.ReLU())
            dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        if isinstance(states, torch.Tensor):
            states_tensor = states.float()
        else:
            states_tensor = torch.tensor(states, dtype=torch.float32)
        return self.network(states_tensor)

    def print_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.numpy()}")


def conservative_q_iteration(env,
                            network, fig, axis,
                            num_itrs=100,
                            project_steps=50,
                            cql_alpha=0.1,
                            render=False,
                            weights=None,
                            sampled=False,
                            training_dataset=None,
                            log_file=None, learning_rate=1e-3,
                            lr_decay=0.99, lr_min=1e-8,
                            **kwargs):
    """
    Runs Conservative Q-iteration.

    Args:
        env: A GridEnv object.
        num_itrs (int): Number of FQI iterations to run.
        project_steps (int): Number of gradient steps used for projection.
        cql_alpha (float): Value of weight on the CQL coefficient.
        render (bool): If True, will plot q-values after each iteration.
        sampled (bool): Whether to use sampled datasets for training or not.
        training_dataset (list): list of (s, a, r, ns) pairs
    """
    # optimizer and scheduler
    optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=1000, min_lr=1e-8, threshold=0.01
    )

    # Keep the initial learning rate and decay settings so we can update
    # the optimizer learning rate at each outer iteration (num_itrs).
    initial_lr = learning_rate

    loss_prev = 0.0
    for i in range(num_itrs):
        # decay learning rate per outer iteration (geometric decay)
        new_lr = max(lr_min, initial_lr * (lr_decay ** i))
        for g in optimizer.param_groups:
            g['lr'] = new_lr

        for j in range(project_steps):
            # choose a minibatch size up to the dataset size
            batch_size = min(128, max(1, len(training_dataset)))
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size)
            s, a, ns, r = get_tensors(training_dataset, training_idx)
            target_values = q_backup_sparse_sampled(env, network, s, a, ns, r, **kwargs)
            loss = project_qvalues_cql_sampled(
                env, s, a, target_values, network, optimizer,
                cql_alpha=cql_alpha, weights=None,
            )
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            # log iteration and target values
            if log_file is not None:
                try:
                    log_file.writerow([i, float(np.mean(target_values)), float(np.mean(target_values))])
                except Exception:
                    pass

        # plot and display progress
        try:
            axis.plot(i, float(np.mean(loss)), 'ro')
            axis.grid(True)
        except Exception:
            pass
        print("Loss value is : ", float(np.mean(loss)))
        if np.mean([loss, loss_prev]) < 4.5:
            continue
        else:
            loss_prev = loss
        clear_output(wait=True)  # Clear the output to update the plot
        display(fig)

    return network
def project_qvalues_cql_sampled(env, s, a, target_values, network, optimizer, cql_alpha=0.1, num_steps=50, weights=None):
    # Convert targets and states to torch tensors with correct dtype
    target_qvalues = torch.tensor(target_values, dtype=torch.float32)
    s_tensor = torch.tensor(s, dtype=torch.float32)

    # Robust action -> index mapping: match nearest action in ACTIONS (handles float rounding)
    a_arr = np.array(a)
    action_indices = np.array([int(np.argmin(np.abs(np.array(ACTIONS) - aa))) for aa in a_arr])
    a_indices = torch.tensor(action_indices, dtype=torch.int64)

    pred_qvalues_all = network(s_tensor)  # shape (batch, num_actions)
    logsumexp_qvalues = torch.logsumexp(pred_qvalues_all, dim=-1)

    pred_qvalues = pred_qvalues_all.gather(1, a_indices.reshape(-1, 1)).squeeze()
    cql_term = logsumexp_qvalues - pred_qvalues

    # Use Huber (SmoothL1) loss for robustness to outliers
    huber = torch.nn.SmoothL1Loss()
    loss_mse = huber(pred_qvalues, target_qvalues)
    loss = loss_mse + cql_alpha * torch.mean(cql_term)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()

    # Return scalar values for logging (total loss)
    return loss.item()

# %%
def project_qvalues_cql(q_values, network, optimizer, num_steps=50, cql_alpha=0.1, weights=None):
    # regress onto q_values (aka projection)
    q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
    for _ in range(num_steps):
       # Eval the network at each state
      pred_qvalues = network(torch.arange(q_values.shape[0]))
      if weights is None:
        loss = torch.mean((pred_qvalues - q_values_tensor)**2)
      else:
        loss = torch.mean(weights*(pred_qvalues - q_values_tensor)**2)

      # Add cql_loss
      # You can have two variants of this loss, one where data q-values
      # also maximized (CQL-v2), and one where only the large Q-values
      # are pushed down (CQL-v1) as covered in the tutorial
      cql_loss = torch.logsumexp(pred_qvalues, dim=-1, keepdim=True) # - pred_qvalues
      loss = loss + cql_alpha * torch.mean(weights * cql_loss)
      network.zero_grad()
      loss.backward()
      optimizer.step()
    return pred_qvalues.detach().numpy()

# %%
def q_backup_sparse_sampled(env, network, s, a, ns, r, discount=0.99):
    # Use torch operations for stable numeric behavior
    with torch.no_grad():
        ns_tensor = torch.tensor(ns, dtype=torch.float32)
        q_values_ns = network(ns_tensor)
        values = q_values_ns.max(dim=-1).values.cpu().numpy()
    target_value = r + discount * values
    return target_value


# %%
def conservative_q_iteration(env,
                             network, fig, axis,
                             num_itrs=100,
                             project_steps=50,
                             cql_alpha=0.1,
                             render=False,
                             weights=None,
                             sampled=False,
                             training_dataset=None,
                             log_file=None, learning_rate=1e-3,
                             **kwargs):
    """
    Runs Conservative Q-iteration.

    Args:
        env: A GridEnv object.
        num_itrs (int): Number of FQI iterations to run.
        project_steps (int): Number of gradient steps used for projection.
        cql_alpha (float): Value of weight on the CQL coefficient.
        render (bool): If True, will plot q-values after each iteration.
        sampled (bool): Whether to use sampled datasets for training or not.
        training_dataset (list): list of (s, a, r, ns) pairs
    """
    # print(log_file)
    # Use Adam for more stable optimisation on these projection/regression steps
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=1000, min_lr=1e-8, threshold=0.01
    )
    # scheduler = scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2000, min_lr=1e-8, threshold=0.1, verbose=True)
    # q_values = np.zeros((dS, dA)) #Initializing the Q-values for getting the target values
    # q_values = network(s)
    loss_prev = 0
    for i in range(num_itrs):
        for j in range(project_steps):
            # choose a minibatch size up to the dataset size
            batch_size = min(128, max(1, len(training_dataset)))
            training_idx = np.random.choice(np.arange(len(training_dataset)), size=batch_size)
            s, a, ns, r = get_tensors(training_dataset, training_idx)
            target_values = q_backup_sparse_sampled(env, network, s, a, ns, r, **kwargs)
            loss = project_qvalues_cql_sampled(
                env, s, a, target_values, network, optimizer,
                cql_alpha=cql_alpha, weights=None,
            )
            scheduler.step(loss)
            # if j == project_steps - 1:
            #   q_values = intermed_values
            current_lr = optimizer.param_groups[0]['lr']
            if log_file is not None:
                try:
                    log_file.writerow([i, np.mean(target_values), np.mean(target_values)])
                except Exception:
                    pass
        try:
            axis.plot(i, np.mean(loss), 'ro')
            axis.grid(True)
        except Exception:
            pass
        print("Loss value is : ", np.mean(loss))
        if np.mean([loss, loss_prev]) < 4.5:
            continue
        else:
            loss_prev = loss
        clear_output(wait=True)  # Clear the output to update the plot
        display(fig)
    return network

# %%
network = FCNetwork(env, layers=[10,10])
LR = 1e-4
cql_alpha_val = 0.01
weights = None
print (weighting_only)
plt.rcParams.update({'font.size': 12, "font.weight": "bold", 'axes.labelsize': 'x-large'})
fig,axs = plt.subplots(1,1,figsize=(10,5))
axs.set_xlabel('Iteration')
axs.set_ylabel('Loss value')
axs.set_title('RL actor network convergence')
plt.tight_layout()
csv_file_name = 'training_results.csv'
OUT_DIR = './trained_models'
output_file_path = os.path.join(OUT_DIR, csv_file_name)
# print(output_file_path)
################### PRETRIANED MODEL #########################
# pretrained_model = 'good_IC.pth'
# pretrained_network = torch.load(pretrained_model)
# network.load_state_dict(pretrained_network['model_state_dict'])
# start_epoch = pretrained_network['epoch']
##############################################################
initial_state = {
       'model_state_dict': network.state_dict(),
       'optimizer_state_dict': network.state_dict(),
       'epoch': 0,  # Starting epoch
   }
################### LOAD INITIAL STATE #########################
# initial_state_path = './initial_state_0.15_0.0001.pth'
# initial_state = torch.load(initial_state_path)
# network.load_state_dict(initial_state['model_state_dict'])
# start_epoch = initial_state['epoch']
# print(f"Loaded initial state from {initial_state_path}, starting from epoch {start_epoch}")
##############################################################

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # If the loaded training dataset is empty (CSV missing/empty), create a synthetic dataset
    if len(training_dataset_loaded) == 0:
        print("Loaded training dataset is empty — creating synthetic dataset for smoke test")
        def _sample_state():
            return tuple(float(np.random.uniform(OBS_MIN[i], OBS_MAX[i])) for i in range(len(OBS_MIN)))
        training_dataset_loaded = [(
            _sample_state(),
            float(np.random.choice(ACTIONS)),
            float(np.random.uniform(-5.0, 1.0)),
            _sample_state()
        ) for _ in range(1000)]

    # Short smoke-test run: fewer iterations/steps, smaller LR and smaller cql_alpha
    trained_net = conservative_q_iteration(env, network, fig, axs,
                                        num_itrs=1500, project_steps=10, discount=0.9, cql_alpha=cql_alpha_val,
                                        weights=weights, render=False,
                                        sampled=not(weighting_only),
                                        training_dataset=training_dataset_loaded,log_file=csv_writer,learning_rate=LR)


torch.save(initial_state, f'./{OUT_DIR}/initial_state_{cql_alpha_val}_{LR}.pth')
fig.savefig(f'./{OUT_DIR}/Q-value_convergence.pdf')


# %%
# network = FCNetwork(env, layers=[10,10])
# LR = 1e-4
# cql_alpha_val = 0.3
# weights = None
# print (weighting_only)
# plt.rcParams.update({'font.size': 12, "font.weight": "bold", 'axes.labelsize': 'x-large'})
# fig,axs = plt.subplots(1,1,figsize=(10,5))
# axs.set_xlabel('Iteration')
# axs.set_ylabel('Loss value')
# axs.set_title('RL actor network convergence')
# plt.tight_layout()
# csv_file_name = 'training_results.csv'
# OUT_DIR = './trained_models'
# output_file_path = os.path.join(OUT_DIR, csv_file_name)
# # print(output_file_path)
# ################### PRETRIANED MODEL #########################
# # pretrained_model = 'good_IC.pth'
# # pretrained_network = torch.load(pretrained_model)
# # network.load_state_dict(pretrained_network['model_state_dict'])
# # start_epoch = pretrained_network['epoch']
# ##############################################################

# initial_state = {
#        'model_state_dict': network.state_dict(),
#        'optimizer_state_dict': network.state_dict(),
#        'epoch': 0,  # Starting epoch
#    }

# ################### LOAD INITIAL STATE #########################
# initial_state_path = './initial_state_0.15_0.0001.pth'
# initial_state = torch.load(initial_state_path)
# network.load_state_dict(initial_state['model_state_dict'])
# start_epoch = initial_state['epoch']
# print(f"Loaded initial state from {initial_state_path}, starting from epoch {start_epoch}")
# ##############################################################
# with open(output_file_path, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     trained_net = conservative_q_iteration(env, network, fig, axs,
#                                         num_itrs=500, discount=0.9, cql_alpha=cql_alpha_val,
#                                         weights=weights, render=False,
#                                         sampled=not(weighting_only),
#                                         training_dataset=training_dataset_loaded,log_file=csv_writer,learning_rate=LR)


# torch.save(initial_state, f'./{OUT_DIR}/initial_state_{cql_alpha_val}_{LR}.pth')
# fig.savefig(f'./{OUT_DIR}/Q-value_convergence.pdf')


# %%
def compress_results_folder(results_dir,current_time,cql_alpha,LR):
    tar_file_name = f'{results_dir}/training_results_{current_time}_{cql_alpha}_{LR}.tar'
    
    # Create a tar file
    with tarfile.open(tar_file_name, 'w') as tar:
        # Iterate through the files in the results directory
        for item in os.listdir(results_dir):
            if item.endswith(('.pth', '.csv', '.pdf')):
                item_path = os.path.join(results_dir, item)
                # Check if it's a file (not a directory)
                # if os.path.isfile(item_path):
                tar.add(item_path, arcname=item)  # Add file to tar
                os.remove(item_path)

    print(f"Compressed files into {tar_file_name}")

# Specify the results directory


# %%
# training_idx = [500]
# s, a, ns, r = get_tensors(training_dataset, training_idx)
# print(s)
# actions = trained_net(s).detach().numpy()
# argmax = np.argmax(actions, axis=-1)
# print(argmax)
# print([ACTIONS[a] for a in argmax])


# trained_net.print_weights()

torch.save(trained_net.state_dict(), f'{OUT_DIR}/trained_network_weights_{current_time}_{cql_alpha_val}_{LR}.pth')  # Save the model weights
compress_results_folder(OUT_DIR,current_time,cql_alpha_val,LR)
print("Trained weights have been saved to 'trained_network_weights.pth'")





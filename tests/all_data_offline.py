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
import math
import csv
import utils
import DDPG
import gym
import BCQ
import argparse

# Ignore all warnings
warnings.filterwarnings('ignore')



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
            PAPI[extracted_scope]['instantaneous_value'] = [(value - min_val) / (max_val - min_val) * 10 for value in instantaneous_values]
            PAPI[extracted_scope]['elapsed_time'] = PAPI[extracted_scope]['time'] - PAPI[extracted_scope]['time'].iloc[0]
    return PAPI

def normalize(traces):
    normalized_PAPI = {}
    max_value = {'PAPI_L3_TCA': float('-inf'), 'PAPI_TOT_INS': float('-inf'), 'PAPI_TOT_CYC': float('-inf'), 'PAPI_RES_STL': float('-inf'), 'PAPI_L3_TCM': float('-inf')}
    min_value = {'PAPI_L3_TCA': float('inf'), 'PAPI_TOT_INS': float('inf'), 'PAPI_TOT_CYC': float('inf'), 'PAPI_RES_STL': float('inf'), 'PAPI_L3_TCM': float('inf')}
    for app in traces.keys():
        for trace in traces[app].keys():
            for scope in traces[app][trace]['papi'].keys():
                max_value[scope] = max(max_value[scope],max(traces[app][trace]['papi'][scope]['instantaneous_value']))
                min_value[scope] = min(min_value[scope],min(traces[app][trace]['papi'][scope]['instantaneous_value']))
    for app in traces.keys():
        for trace in traces[app].keys():
            for scope in traces[app][trace]['papi'].keys():
                traces[app][trace]['papi'][scope]['normalized_value'] = [(value - min_value[scope]) / (max_value[scope] - min_value[scope]) * 10 for value in traces[app][trace]['papi'][scope]['instantaneous_value']]
    return traces

def generate_PCAP(PCAP_data):
    for row in PCAP_data.iterrows():
        if row[1]['time'] == 0:
            PCAP_data = PCAP_data.drop(row[0])


    PCAP_data['elapsed_time'] = PCAP_data['time'] - PCAP_data['time'].iloc[0]
    return PCAP_data

def get_data_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "experiment_data", "data_generation")

DATA_DIR = get_data_dir()
root,folders,files = next(os.walk(DATA_DIR))
training_data = {}
for APP in folders:
    # fig, axs = plt.subplots(2,1,figsize=(12,10))
    # fig_PAPI, axs_PAPI = plt.subplots(5,1,figsize=(12,16))
    # print(APP)
    APP_DIR = os.path.join(DATA_DIR, APP)
    # print(APP_DIR)
    training_data[APP] = {}
    # training_data[APP]['data'] = {} 
    for file in next(os.walk(APP_DIR))[2]:
        # print(file)
        training_data[APP][file] = {}
        if file.endswith('.tar'):
            tar_path = os.path.join(APP_DIR, file)
            extract_dir = os.path.join(APP_DIR, file[:-4])  
            
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            
            # Extract the tar file
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
            # print(f"Extracted {file} to {extract_dir}")
        pubProgress = pd.read_csv(f'{extract_dir}/progress.csv')
        pubEnergy = pd.read_csv(f'{extract_dir}/energy.csv')
        pubPAPI = pd.read_csv(f'{extract_dir}/papi.csv')
        pubPCAP = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
        # with open(f'{extract_dir}/parameters.yaml', 'r') as f:
        #     yaml = YAML(typ='safe', pure=True)
        #     parameters = yaml.load(f)
        #     PCAP = parameters['PCAP']
        # training_data['data']['PCAP'] = pd.read_csv(f'{extract_dir}/PCAP_file.csv')
        training_data[APP][file]['power'] = compute_power(pubEnergy)
        training_data[APP][file]['progress'] = measure_progress(pubProgress,training_data[APP][file]['power'])
        training_data[APP][file]['papi'] = collect_papi(pubPAPI)
        training_data[APP][file]['PCAP'] = generate_PCAP(pubPCAP)   
        # print(training_data[APP][file]['PCAP'] )    
training_data = normalize(training_data)
# print(training_data[APP][file]['papi'])

# print(training_data[APP][file]['papi']['PAPI_L3_TCA'])

T_S = 1
ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
exec_steps = 10000    
TOTAL_ACTIONS = len(ACTIONS)                                                                                                  # Total clock cycles needed for the execution of program.
ACTION_MIN = min(ACTIONS)                                                                                                    # Minima of control space
ACTION_MAX = max(ACTIONS)                                                                                                     # Maxima of control space
ACT_MID = ACTION_MIN + (ACTION_MAX - ACTION_MIN) / 2                                                                    # Midpoint of the control space to compute the normalized action space
# OBS_MAX = 300                                                                                                           # Maxima of observation space (performance)
# OBS_MIN = 0                           
OBS_MIN = np.zeros((7,))  # Shape should be (7,)
OBS_MAX = np.array([300,165,10,10,10,10,10])                                                                                 # Minima of observation space
OBS_MID = OBS_MIN + (OBS_MAX - OBS_MIN) / 2
EXEC_ITERATIONS = 10000
TOTAL_OBS = OBS_MAX - OBS_MIN
# print(TOTAL_ACTIONS)
OBS_ONEHOT = 'onehot'
OBS_RANDOM = 'random'
OBS_SMOOTH = 'smooth'


class SYS(object):
    def __init__(self,observation_type=OBS_ONEHOT,dim_obs=1,teps=0.0):
        super(SYS,self).__init__()

        self.num_actions = TOTAL_ACTIONS
        self.action_space = gym.spaces.Discrete(len(ACTIONS))  # Use the length of the ACTIONS list for discrete actions
        # Map the selected index to the corresponding action
        self.actions = ACTIONS  # Store the actions for later use
        self.observation_space = gym.spaces.Box(low=OBS_MIN, high=OBS_MAX, shape=(7,), dtype=np.float32)  # Infinite observation space with 8 dimensions

    
    def reward(self, s, a, ns, measured_power):
        """ 
        Returns the reward (float)
        """
        # measured_power = A[cluster] * a + B[cluster]
        if ns > 0:
            # self.current_step += ns
            # reward = - 5*a
            # reward = 2*ns/(((a)/measured_power)+measured_power) # Check the behaviour across the states
            reward = -ns/(2*a**2+1)
            # reward = 5*a
        else:
            reward = -100
        # print(reward)
        return reward

weighting_only = False
dataset_composition = 'random'
dataset_size = 1000
env_type = 'random'
env = SYS(observation_type=env_type, dim_obs=8, teps=0)


# training_data_csv = pd.read_csv('./merged_data.csv')
# PCAP = 0
# CURRENT_PRO = 0
# NEXT_PRO = 0
training_dataset = []
old_PCAP = 0
action_set = []
# Helper function to get state
def get_roi_data(df, time_column, start_time, end_time):
    return df[(df[time_column] > start_time) & (df[time_column] <= end_time)]

def get_state(training_data, app, trace, start_time, end_time):
    ROI_progress = get_roi_data(training_data[app][trace]['progress']['progress_frequency_median'], 'timestamp', start_time, end_time)
    ROI_measured_power = get_roi_data(training_data[app][trace]['power']['average_power'], 'timestamp', start_time, end_time)
    ROI_L3_TCA = get_roi_data(training_data[app][trace]['papi']['PAPI_L3_TCA'], 'time', start_time, end_time)
    ROI_TOT_INS = get_roi_data(training_data[app][trace]['papi']['PAPI_TOT_INS'], 'time', start_time, end_time)
    ROI_TOT_CYC = get_roi_data(training_data[app][trace]['papi']['PAPI_TOT_CYC'], 'time', start_time, end_time)
    ROI_RES_STL = get_roi_data(training_data[app][trace]['papi']['PAPI_RES_STL'], 'time', start_time, end_time)
    ROI_L3_TCM = get_roi_data(training_data[app][trace]['papi']['PAPI_L3_TCM'], 'time', start_time, end_time)
    
    return (
        ROI_progress['median'].mean() if not ROI_progress.empty else 0,
        ROI_measured_power['average_power'].mean() if not ROI_measured_power.empty else 0,
        ROI_L3_TCA['normalized_value'].mean() if not ROI_L3_TCA.empty else 0,
        ROI_TOT_INS['normalized_value'].mean() if not ROI_TOT_INS.empty else 0,
        ROI_TOT_CYC['normalized_value'].mean() if not ROI_TOT_CYC.empty else 0,
        ROI_RES_STL['normalized_value'].mean() if not ROI_RES_STL.empty else 0,
        ROI_L3_TCM['normalized_value'].mean() if not ROI_L3_TCM.empty else 0
    )

# state_definition = [progress, measured_power, previous_PCAP, 'PAPI_L3_TCA', 'PAPI_TOT_INS', 'PAPI_TOT_CYC', 'PAPI_RES_STL', 'PAPI_L3_TCM']
initial_progress = 0
initial_power = 40
initial_PAPI = np.zeros(5)
state = (initial_progress,initial_power,initial_PAPI)
t1 = float('-inf')
for app in training_data.keys():
    for trace in training_data[app].keys():
        pcap_data = training_data[app][trace]['PCAP']
        for i, row in pcap_data.iterrows():
            t2 = row['time']
            
            # Get current state
            state = get_state(training_data, app, trace, t1, t2)
            
            # Get next state (look ahead to next row)
            if i + 1 < len(pcap_data):
                t3 = pcap_data.iloc[i + 1]['time']
                next_state = get_state(training_data, app, trace, t2, t3)
            else:
                next_state = state  # Use current state if it's the last row
            
            action = row['value']  # Assuming PCAP is in the 'value' column
            
            # Calculate the reward
            reward = env.reward(state[0], action, next_state[0], state[1])
            
            # Add to training dataset
            training_dataset.append((state, action, reward, next_state))
            
            t1 = t2


# Define the CSV file name and path
csv_file_name = 'training_dataset.csv'
csv_file_path = os.path.join(DATA_DIR, csv_file_name)

# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header
    csv_writer.writerow(['Progress', 'Power', 'L3_TCA', 'TOT_INS', 'TOT_CYC', 'RES_STL', 'L3_TCM', 
                         'Action', 'Reward', 
                         'Next_Progress', 'Next_Power', 'Next_L3_TCA', 'Next_TOT_INS', 'Next_TOT_CYC', 'Next_RES_STL', 'Next_L3_TCM'])
    
    # Write the data
    for state, action, reward, next_state in training_dataset:
        row = list(state) + [action, reward] + list(next_state)
        csv_writer.writerow(row)

print(f"Training dataset has been saved to {csv_file_path}")


def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args, replay_buffer):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

	# Load buffer
	# replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	# replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0

	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
		if "results" not in os.listdir():  # Check for the directory without "./"
			os.makedirs("results", exist_ok=True)  # Create the directory if it doesn't exist
		np.save(f"./results/BCQ_{setting}", pol_vals)  # Save the results

		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")


BATCH_SIZE = 2500
state_dim = 7
action_dim = 1

device = torch.device("cpu")
max_action = ACTION_MAX
max_timesteps = 2500
episode_reward = 0
episode_timesteps = 0
episode_num = 0
policy = DDPG.DDPG(state_dim, action_dim, max_action, device)#, args.discount, args.tau)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
# Assuming training_dataset is a list of tuples


	
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="SYS")               # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e4, type=int)   # Max time steps to run environment or train for (this defines buffer size)
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
parser.add_argument("--discount", default=0.99)                 # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
args = parser.parse_args()



for i in range(len(training_dataset) - 1):  # Avoid index out of range
    current_state, action, reward, next_state = training_dataset[i]
    next_state_info,_,_,_, = training_dataset[i + 1]  # Access the next line
    if next_state_info[0] == 0:
        done = float(True)
    else:
        done = float(False)
    replay_buffer.add(current_state, action, next_state, reward, done)
    episode_reward += reward
    # print(done)
    if done == 1.0: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T:{i} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        # state, done = env.reset(), False
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
train_BCQ(state_dim, action_dim, max_action, device, args, replay_buffer)

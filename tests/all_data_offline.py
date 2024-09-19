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
OBS_MAX = 300                                                                                                           # Maxima of observation space (performance)
OBS_MIN = 0                                                                                                             # Minima of observation space
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

        self.num_states = TOTAL_OBS
        self.num_actions = TOTAL_ACTIONS
        self.obs_type = observation_type
        # self.model = TransitionDynamics(eps=teps)
        # self._transition_matrix = None
        # self._transition_matrix = self.transition_matrix()
        self.current_step = 0

        if self.obs_type == OBS_RANDOM:
          self.dim_obs = dim_obs
          self.obs_matrix = np.random.randn(self.num_states, self.dim_obs)
        elif self.obs_type == OBS_SMOOTH:
          self.dim_obs = dim_obs
          self.obs_matrix = np.random.randn(self.num_states, self.dim_obs)
        #   trans_matrix = np.sum(self._transition_matrix, axis=1) / self.num_actions
        #   for k in range(10):
            # cur_obs_mat = self.obs_matrix[:,:]
            # for state in range(self.num_states):
                # new_obs = trans_matrix[state].dot(cur_obs_mat)
                # self.obs_matrix[state] = new_obs
        # else:
        #   self.dim_obs = self.gs.width+self.gs.height


    def observation(self, s):
        if self.obs_type == OBS_ONEHOT:
          xy_vec = np.zeros(self.gs.width+self.gs.height)
          xy = self.gs.idx_to_xy(s)
          xy_vec[xy[0]] = 1.0
          xy_vec[xy[1]+self.gs.width] = 1.0
          return xy_vec
        elif self.obs_type == OBS_RANDOM or self.obs_type == OBS_SMOOTH:
          return self.obs_matrix[s]
        else:
          raise ValueError("Invalid obs type %s" % self.obs_type)
    
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

# Helper function to get ROI data


#@ Plotting tools

PLT_NOOP = np.array([[-0.1,0.1], [-0.1,-0.1], [0.1,-0.1], [0.1,0.1]])
PLT_UP = np.array([[0,0], [0.5,0.5], [-0.5,0.5]])
PLT_LEFT = np.array([[0,0], [-0.5,0.5], [-0.5,-0.5]])
PLT_RIGHT = np.array([[0,0], [0.5,0.5], [0.5,-0.5]])
PLT_DOWN = np.array([[0,0], [0.5,-0.5], [-0.5,-0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0,0])+TXT_CENTERING
TXT_UP = np.array([0,TXT_OFFSET_VAL])+TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL,0])+TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL,0])+TXT_CENTERING
TXT_DOWN = np.array([0,-TXT_OFFSET_VAL])+TXT_CENTERING



ACT_OFFSETS = [[i,TXT_NOOP] for i in range(0,TOTAL_ACTIONS)]

PLOT_CMAP = cm.RdYlBu




def plot_sa_values(env, q_values, text_values=True, 
                   invert_y=True, update=False,
                   title=None):
  w = TOTAL_OBS
  h = TOTAL_ACTIONS
  
  if update:
    clear_output(wait=True)
  plt.figure(figsize=(2*w, 2*h))
  ax = plt.gca()
  normalized_values = q_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  for x, y in itertools.product(range(w), range(h)):
    #   state_idx = env.gs.xy_to_idx((x, y))
    #   if invert_y:
        #   y = h-y-1
      xy = np.array([x, y])
      xy3 = np.expand_dims(xy, axis=0)
      state_idx = x
    #   print(x)

      for a in range(ACTION_MAX-1,ACTION_MIN,-1):
          a = a - ACTION_MIN
        #   print(a)
          val = normalized_values[state_idx,a]
          og_val = q_values[state_idx,a]
          patch_offset, txt_offset = ACT_OFFSETS[a]
          if text_values:
              xy_text = xy+txt_offset
              ax.text(xy_text[0], xy_text[1], '%.2f'%og_val, size='small')
          color = PLOT_CMAP(val)
          ax.add_patch(Polygon(xy3+patch_offset, True,
                                     color=color))
  ax.set_xticks(np.arange(-1, w+1, 1))
  ax.set_yticks(np.arange(-1, h+1, 1))
  plt.grid()
  if title:
    plt.title(title)
  # plt.draw()
  # frame_number+=1
  plt.savefig(f'{title}.png')
  frame_number += 1
  # plt.close()

def plot_sa_values_simple(env, q_values, text_values=True, 
                   invert_y=True, update=False,
                   title=None):
  w = TOTAL_OBS
  h = TOTAL_ACTIONS
  
  # plt.figure(figsize=(2*w, 2*h))
  # ax = plt.gca()
  normalized_values = q_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  
  img = plt.imshow(normalized_values, cmap=PLOT_CMAP, interpolation='nearest', aspect='auto')
  ax = plt.gca()
  for x, y in itertools.product(range(h), range(w)):
      # a = x + ACTION_MIN
      # val = normalized_values[x,y]
      og_val = q_values[y,x]
      # print(og_val)
      _, txt_offset = TXT_NOOP
      if text_values:
          xy_text = np.array([x, y])+txt_offset
          # print(xy_text,np.array([x,y]))
          ax.text(xy_text[0], xy_text[1], '%.2f'%og_val, fontsize=2)
  ax.set_xticks(np.arange(-1, h+1, 1))
  ax.set_yticks(np.arange(-1, w+1, 1))
  plt.grid()
  if title:
    plt.title(title)
  # plt.draw()
  # frame_number+=1
  plt.savefig(f'{title}.png')
  # frame_number += 1
  # plt.close()  

def plot_s_values(env, v_values, text_values=True, 
                  invert_y=True, update=False,
                  title=None):
  w = TOTAL_OBS
  h = TOTAL_OBS
  if update:
    clear_output(wait=True)
  plt.figure(figsize=(2*w, 2*h))
  ax = plt.gca()
  normalized_values = v_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  for x, y in itertools.product(range(w), range(h)):
      state_idx = x

      xy = np.array([x, y])

      val = normalized_values[state_idx]
      og_val = v_values[state_idx]
      if text_values:
          xy_text = xy
          ax.text(xy_text[0], xy_text[1], '%.2f'%og_val, size='small')
      color = PLOT_CMAP(val)
      ax.add_patch(Rectangle(xy-0.5, 1, 1, color=color))
  ax.set_xticks(np.arange(-1, w+1, 1))
  ax.set_yticks(np.arange(-1, h+1, 1))
  plt.grid()  
  if title:
    plt.title(title)
  plt.draw()
  plt.savefig(f'frame_{frame_number}.png')
  # plt.close()
# print(training_dataset)
  
def plot_test_actions(env, q_values, text_values=True, 
                   invert_y=True, update=False,
                   title=None):
  h = TOTAL_OBS
  w = TOTAL_ACTIONS          # ax.add_patch(Polygon(xy3+patch_offset, True,
          #                            color=color))
  
  if update:
    clear_output(wait=True)
  # plt.figure(figsize=(2*w, 2*h))
  # ax1 = plt.gca()
  # plt.figure(figsize=(2*w, 2*h))
  # ax2 = plt.gca()
  fig1,ax1 = plt.subplots()
  fig2,ax2 = plt.subplots()
  normalized_values = q_values
  normalized_values = normalized_values - np.min(normalized_values)
  normalized_values = normalized_values/np.max(normalized_values)
  # for x, y in itertools.product(range(w), range(h)):
    #   state_idx = env.gs.xy_to_idx((x, y))
    #   if invert_y:
        #   y = h-y-1
      # xy = np.array([x, y])
      # xy3 = np.expand_dims(xy, axis=0)
      # state_idx = x
    #   print(x)
  complete = False
  new_s = 0
  count = 0
  RUNTIME = 100
  PROGRESS = []
  PCAP = []
  TIME = []
  while not complete:
      state_idx = new_s
      # for a in range(ACTION_MAX-1,ACTION_MIN,-1):
          # a = a - ACTION_MIN
        #   print(a)
      # val = normalized_values[state_idx,a]
      og_act = np.max(q_values[state_idx,:])
      act = np.argmax(q_values[state_idx,:])+ACTION_MIN
      new_s,_,MP = progress_funct(state_idx,act)
      PROGRESS.append(new_s)
      PCAP.append(act)
      TIME.append(count)
      # ax1.plot(state_idx,act,'r--')
      # ax2.plot(MP,act,'k.')
      # print(count)
      if count == RUNTIME:
         complete = True
      count += 1
  ax1.scatter(TIME,PCAP,marker='.', color='k', s=30, label=f'{"PCAP"}')
  ax1.grid(True)
  ax1.set_ylabel('PCAP',fontsize = 15)
  ax1.set_xlabel('time [s]', fontsize = 15)
  ax1.tick_params(axis='x', labelsize=8)
  ax1.tick_params(axis='y', labelsize=8)
  ax1.legend(fontsize = 13)
  title1 = f"{cluster}"
  ax1.set_title(title1, fontsize=15, color = 'blue')
  fig1.savefig(f'PCAP_{title}.png')

  ax2.scatter(TIME,PROGRESS,marker='.', color='r', s=30, label=f'{"PROGRESS"}')
  ax2.grid(True)
  ax2.set_ylabel('PROGRESS',fontsize = 15)
  ax2.set_xlabel('time [s]', fontsize = 15)
  ax2.tick_params(axis='x', labelsize=8)
  ax2.tick_params(axis='y', labelsize=8)
  ax2.legend(fontsize = 13)
  title1 = f"{cluster}"
  ax2.set_title(title1, fontsize=15, color = 'blue')
  fig2.savefig(f'progress_{title}.png')

  print(f"Total energy consumed is {sum(PCAP)} kJ")


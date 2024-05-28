import os
import pandas as pd
import matplotlib.pyplot as plt
import ruamel.yaml
import math
import scipy.optimize as opt
import numpy as np
import tarfile
from matplotlib import cm
import seaborn as sns
import warnings
import yaml
yaml_format = ruamel.yaml.YAML()
warnings.filterwarnings('ignore')


exp_type = 'identification' 
experiment_dir = './experiment_data/'
clusters = [item for item in next(os.walk(experiment_dir))[1] if exp_type in item]
print(clusters)
if (exp_type == 'stairs') or (exp_type == 'static_characteristic'):
    experiment_type = 'identification'
else:
    experiment_type = exp_type

traces = {} 
traces_tmp = {}
for cluster in clusters:
    traces[cluster] = pd.DataFrame()
    print(cluster,"...")
    csv_files = next(os.walk(experiment_dir+cluster))[2]
    if next(os.walk(experiment_dir+cluster))[1] == []:
        files = os.listdir(experiment_dir+cluster)
        for fname in files:
            if fname.endswith("tar.xz"):
                tar = tarfile.open(experiment_dir+cluster+'/'+fname, "r:xz") 
                tar.extractall(path=experiment_dir+cluster+'/'+fname[:-7])
                tar.close()
    traces[cluster][0] = next(os.walk(experiment_dir+cluster))[1] 

print(traces)
print(csv_files)

for cluster in clusters:
    folder_path = experiment_dir+cluster
    pubMeasurements = pd.read_csv(folder_path+"/dump_pubMeasurements.csv")
    pubProgress = pd.read_csv(folder_path+"/dump_pubProgress.csv")
    print(pubProgress['sensor.value'])
    total_packages = np.nansum(pubProgress['sensor.value'])
    print(total_packages)

    rapl_sensor0 = rapl_sensor1 = rapl_sensor2 = rapl_sensor3 = downstream_sensor = pd.DataFrame({'timestamp':[],'value':[]})
    for i, row in pubMeasurements.iterrows():
        if row['sensor.id'] == 'RaplKey (PackageID 0)':
            rapl_sensor0 = rapl_sensor0.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
        elif row['sensor.id'] == 'RaplKey (PackageID 1)':
            rapl_sensor1 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
        elif row['sensor.id'] == 'RaplKey (PackageID 2)':
            rapl_sensor2 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
        elif row['sensor.id'] == 'RaplKey (PackageID 3)':
            rapl_sensor3 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
    sensor_values = pd.DataFrame({'timestamp':rapl_sensor0['timestamp'],'value0':rapl_sensor0['value'],'value1':rapl_sensor1['value'],'value2':rapl_sensor2['value'],'value3':rapl_sensor3['value']})
    print(sensor_values)
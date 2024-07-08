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
warnings.filterwarnings('ignore')

current_working_directory = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_working_directory)



exp_type = 'identification' 
experiment_dir = f'{current_working_directory}/experiment_data/{exp_type}/'

root, apps, files = next(os.walk(experiment_dir))



for app in apps:
    cwd = root+'/'+app
    traces = {} 
    traces_tmp = {}
    traces[app] = pd.DataFrame()
    if next(os.walk(cwd))[1] == []:
        files = os.listdir(cwd)
        for fname in files:
            if fname.endswith("tar"):
                print(fname)
                tar = tarfile.open(cwd+'/'+fname, "r") 
                tar.extractall(path=cwd+'/'+fname[:-4])
                tar.close()
    traces[app][0] = next(os.walk(cwd))[1]
    data = {} 
    for trace in traces[app][0]:
        data[trace] = pd.DataFrame()
        pwd = f'{cwd}/{trace}'
        # print(pwd)
        # files = next(os.walk(pwd))[2]
        papi_file = pd.read_csv(f'{pwd}/papi.csv')
        for row in papi_file.iterrows():
            # print(row[1])
            timestamp = row[1]['time']
            scope_info = row[1]['scope']
            value = row[1]['value']
            if scope_info in data[trace].index:
                print(f"{scope_info} is in the dataframe index")
                data[trace][scope_info] = pd.Series()
            else:
                print(f"{scope_info} is not in the dataframe index")
                data[trace][scope_info].loc[timestamp] = value
    
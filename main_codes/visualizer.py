import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import csv
import sys 

filename = 'losses_seed_2.csv'
i = 0
while i < len(sys.argv):
    if sys.argv[i] == '--filename':
        filename = sys.argv[i+1]
        i += 1
    else:
        print("ENTER A VALID FILENAME!!!")
    i +=1

fieldnames = ["iteration","actor_loss", "critic_loss"]
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)  # Update the path to use the current script's directory


plt.style.use('fivethirtyeight')

os.path.exists(data_path)

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

FS = 12

data = pd.read_csv(data_path)
axs[0].scatter(data["iteration"], data["actor_loss"], color='red')  # Changed to scatter plot with red color
# axs[0].set_title('Actor Loss')  # {{ edit_1 }}
axs[0].grid(True)
axs[0].set_ylabel("Actor Loss")
axs[1].scatter(data["iteration"], data["critic_loss"], color='green')  # Changed to scatter plot with green color
axs[1].grid(True)
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Critic Loss")
# axs[1].set_title('Critic Loss')  # {{ edit_1 }}

plt.show()
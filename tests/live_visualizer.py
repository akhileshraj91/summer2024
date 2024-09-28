import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import csv
import argparse  # Import argparse

fieldnames = ["iteration","actor_loss", "critic_loss"]

# Add argument parsing for the filename
parser = argparse.ArgumentParser(description='Visualize loss data from a CSV file.')
parser.add_argument('--filename', type=str, default='losses_1.csv', help='Name of the CSV file to read data from')
args = parser.parse_args()

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.filename)  # Use the parsed filename

plt.style.use('fivethirtyeight')

os.path.exists(data_path)

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

FS = 12

data = pd.read_csv(data_path)
axs[0].scatter(data["iteration"], data["actor_loss"], color='red')  # Changed to scatter plot with red color
axs[0].grid(True)
axs[1].scatter(data["iteration"], data["critic_loss"], color='green')  # Changed to scatter plot with green color
axs[1].grid(True)

def update(frame):
    data = pd.read_csv(data_path)  # Read the CSV file each time
    axs[0].clear()  # Clear previous scatter plot
    axs[1].clear()  # Clear previous scatter plot
    axs[0].scatter(data["iteration"], data["actor_loss"], color='red')
    axs[0].set_title('Actor Loss Over Iterations')  # Added title for clarity
    axs[0].grid(True)
    axs[1].scatter(data["iteration"], data["critic_loss"], color='green')
    axs[1].set_title('Critic Loss Over Iterations')  # Added title for clarity
    axs[1].grid(True)

# Create animation
ani = FuncAnimation(fig, update, interval=1000)  # Update every second

plt.show()
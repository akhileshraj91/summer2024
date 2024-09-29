import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# CSV file path from command line argument
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'losses_seed_0.csv'

# Read initial data
data = pd.read_csv(csv_file)

# Create the figure and axis objects
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot initial data
line1, = ax1.plot(data.index, data['actor_loss'], label='Actor Loss')
line2, = ax2.plot(data.index, data['critic_loss'], label='Critic Loss')

# Set up the axes
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Actor Loss')
ax1.set_title('Actor Loss over Iterations')
ax1.legend()

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Critic Loss')
ax2.set_title('Critic Loss over Iterations')
ax2.legend()

# Function to update the plot
def update(frame):
    global data
    new_data = pd.read_csv(csv_file)
    if len(new_data) > len(data):
        # Update data
        line1.set_data(new_data.index, new_data['actor_loss'])
        line2.set_data(new_data.index, new_data['critic_loss'])
        
        # Adjust axes limits if necessary
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Update the global data variable
        data = new_data
    
    return line1, line2

# Create the animation
anim = FuncAnimation(fig, update, interval=1000, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()
# Add this to your CQL_renewed.ipynb before training loop:

import numpy as np
import pandas as pd

# Read your dataset
df = pd.read_csv('/home/cc/summer2024/main_codes/experiment_data/training_data/training_dataset.csv')

# Calculate statistics per app
reward_stats = {}
for app in df['App'].unique():
    app_rewards = df[df['App'] == app]['Reward'].values
    reward_stats[app] = {
        'min': np.min(app_rewards),
        'max': np.max(app_rewards),
        'mean': np.mean(app_rewards),
        'std': np.std(app_rewards)
    }

# Print to see the difference
for app, stats in reward_stats.items():
    print(f"{app}: Range=[{stats['min']:.3f}, {stats['max']:.3f}], Std={stats['std']:.4f}")

# Define stream apps requiring amplification
STREAM_APPS = {'ones-stream-full', 'ones-stream-copy', 'ones-stream-add', 
               'ones-stream-triad', 'ones-stream-scale'}

def amplify_stream_rewards(reward, app_name, scaling_factor=2.5):
    """
    Amplify reward signal for stream apps while preserving NPB rewards
    """
    if app_name in STREAM_APPS:
        return reward * scaling_factor  # Stretch the signal
    return reward

# Use this in your training pipeline - modify rewards during batch loading:
# amplified_reward = amplify_stream_rewards(reward, app_name, scaling_factor=2.0)
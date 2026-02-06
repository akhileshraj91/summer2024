import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Clean and smooth reinforcement learning training data"""
    
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.filepath = filepath
        self.original_size = len(self.df)
        self.cleaning_log = []
    
    def remove_duplicates(self):
        """Remove duplicate trajectories"""
        initial_size = len(self.df)
        
        # Remove exact duplicate rows
        self.df = self.df.drop_duplicates(
            subset=['App', 'Progress', 'Power', 'Action', 'Reward'],
            keep='first'
        ).reset_index(drop=True)  # Reset index after filtering
        
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} duplicate rows")
        return removed
    
    def remove_episode_resets(self):
        """Remove or handle episode resets (zeros appearing in middle of trajectory)"""
        initial_size = len(self.df)
        
        # Reset index to ensure proper alignment
        self.df = self.df.reset_index(drop=True)
        
        # Create boolean mask for valid rows (all True initially)
        valid_mask = [True] * len(self.df)
        
        # Iterate through rows to identify invalid resets
        for i in range(1, len(self.df)):
            is_reset = (self.df.iloc[i]['Progress'] == 0) and (self.df.iloc[i]['Power'] == 0)
            
            if is_reset:
                # Check if previous row was from different app or end of episode
                prev_app = self.df.iloc[i-1]['App']
                curr_app = self.df.iloc[i]['App']
                prev_progress = self.df.iloc[i-1]['Progress']
                
                if curr_app != prev_app:
                    # Different app = valid reset (keep it)
                    valid_mask[i] = True
                elif prev_progress < 15:
                    # Low progress in previous = likely end of episode (keep it)
                    valid_mask[i] = True
                else:
                    # Middle of episode = invalid reset (remove it)
                    valid_mask[i] = False
        
        # Apply the mask
        self.df = self.df[valid_mask].reset_index(drop=True)
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} invalid episode resets")
        return removed
    
    def filter_outliers_zscore(self, columns=['Progress', 'Power', 'Next_Progress', 'Next_Power'], threshold=3):
        """Remove outliers using Z-score method"""
        initial_size = len(self.df)
        self.df = self.df.reset_index(drop=True)
        
        # Start with all valid
        valid_mask = [True] * len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                col_values = self.df[col].values
                # Only compute on non-zero values to avoid skewing
                nonzero_vals = col_values[col_values > 0]
                
                if len(nonzero_vals) > 1:
                    col_mean = np.mean(nonzero_vals)
                    col_std = np.std(nonzero_vals)
                else:
                    col_mean = np.mean(col_values)
                    col_std = np.std(col_values)
                
                if col_std > 0:
                    z_scores = np.abs((self.df[col].values - col_mean) / col_std)
                    valid_mask = [v and (z < threshold) for v, z in zip(valid_mask, z_scores)]
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} outliers (Z-score > {threshold})")
        return removed
    
    def filter_outliers_iqr(self, columns=['Progress', 'Power', 'Next_Progress', 'Next_Power'], multiplier=1.5):
        """Remove outliers using Interquartile Range (IQR) method"""
        initial_size = len(self.df)
        self.df = self.df.reset_index(drop=True)
        
        # Start with all valid
        valid_mask = [True] * len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                col_valid = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                valid_mask = [v and c for v, c in zip(valid_mask, col_valid)]
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} outliers (IQR method)")
        return removed
    
    def filter_by_trajectory_consistency(self, max_jump=50):
        """Remove transitions with unrealistic state changes"""
        initial_size = len(self.df)
        self.df = self.df.reset_index(drop=True)
        
        # Check for large jumps in state
        progress_jump = np.abs(self.df['Next_Progress'].values - self.df['Progress'].values)
        power_jump = np.abs(self.df['Next_Power'].values - self.df['Power'].values)
        
        valid_mask = (progress_jump < max_jump) & (power_jump < max_jump)
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} rows with unrealistic state jumps (> {max_jump})")
        return removed
    
    def smooth_trajectory(self, window_size=5):
        """Smooth state variables within each trajectory using Savitzky-Golay filter"""
        smoothed_df = self.df.copy()
        smoothed_df = smoothed_df.reset_index(drop=True)
        
        for app in smoothed_df['App'].unique():
            mask = (smoothed_df['App'] == app).values
            indices = np.where(mask)[0]
            
            if len(indices) < window_size:
                continue
            
            # Smooth Progress and Power for this app
            for col in ['Progress', 'Power', 'Next_Progress', 'Next_Power']:
                if col in smoothed_df.columns:
                    values = smoothed_df.loc[mask, col].values
                    if len(values) >= window_size and len(values) > 2:
                        try:
                            # Ensure window size is odd and less than array size
                            ws = min(window_size, len(values))
                            if ws % 2 == 0:
                                ws -= 1
                            if ws >= 3:
                                smoothed = signal.savgol_filter(values, ws, polyorder=2)
                                smoothed_df.loc[indices, col] = smoothed
                        except Exception as e:
                            # Fallback to moving average if Savitzky-Golay fails
                            try:
                                smoothed = pd.Series(values).rolling(
                                    window=window_size, center=True, min_periods=1
                                ).mean().values
                                smoothed_df.loc[indices, col] = smoothed
                            except:
                                pass  # Keep original values if smoothing fails
        
        self.cleaning_log.append(f"Applied Savitzky-Golay smoothing (window={window_size})")
        return smoothed_df
    
    def filter_constant_features(self, tolerance=0.01):
        """Remove rows where features don't respond to actions (too constant)"""
        initial_size = len(self.df)
        self.df = self.df.reset_index(drop=True)
        
        # Check if metrics are changing significantly
        metric_cols = ['TOT_INS_PER_CYC', 'L3_TCM_PER_TCA', 'TOT_STL_PER_CYC']
        next_metric_cols = ['Next_TOT_INS_PER_CYC', 'Next_L3_TCM_PER_TCA', 'Next_STL_PER_CYC']
        
        # Start with all valid
        valid_mask = [True] * len(self.df)
        
        for metric, next_metric in zip(metric_cols, next_metric_cols):
            if metric in self.df.columns and next_metric in self.df.columns:
                # Avoid division by zero
                base = self.df[metric].values.copy()
                base[base == 0] = 1e-6
                
                relative_change = np.abs(
                    (self.df[next_metric].values - self.df[metric].values) / base
                )
                
                # Keep rows where at least some change occurs
                metric_valid = (relative_change > tolerance) | (self.df['Action'].values != 0)
                valid_mask = [v and m for v, m in zip(valid_mask, metric_valid)]
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        removed = initial_size - len(self.df)
        self.cleaning_log.append(f"Removed {removed} rows with no feature response (tolerance={tolerance})")
        return removed
    
    def normalize_rewards(self):
        """Normalize rewards to [-1, 1] range"""
        if len(self.df) > 0:
            min_reward = self.df['Reward'].min()
            max_reward = self.df['Reward'].max()
            
            if max_reward - min_reward > 1e-6:
                self.df['Reward'] = 2 * (self.df['Reward'] - min_reward) / (max_reward - min_reward) - 1
                self.cleaning_log.append("Normalized rewards to [-1, 1] range")
    
    def clean(self, remove_dups=True, remove_resets=True, filter_zscore=True, 
              filter_iqr=False, smooth=True, filter_consistency=True, 
              filter_constants=False, normalize_rewards=True):
        """Run full cleaning pipeline"""
        print("Starting data cleaning pipeline...")
        print(f"Original dataset size: {self.original_size}")
        
        if remove_dups:
            self.remove_duplicates()
        
        if remove_resets:
            self.remove_episode_resets()
        
        if filter_zscore:
            self.filter_outliers_zscore()
        
        if filter_iqr:
            self.filter_outliers_iqr()
        
        if filter_consistency:
            self.filter_by_trajectory_consistency()
        
        if filter_constants:
            self.filter_constant_features()
        
        if smooth:
            self.df = self.smooth_trajectory()
        
        if normalize_rewards:
            self.normalize_rewards()
        
        print("\nCleaning Log:")
        for log in self.cleaning_log:
            print(f"  - {log}")
        
        print(f"\nFinal dataset size: {len(self.df)}")
        if self.original_size > 0:
            print(f"Rows removed: {self.original_size - len(self.df)} ({100*(self.original_size - len(self.df))/self.original_size:.1f}%)")
        
        return self.df
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        # Drop temporary columns if they exist
        cols_to_drop = [col for col in self.df.columns if col in ['trajectory_id']]
        df_to_save = self.df.drop(columns=cols_to_drop, errors='ignore')
        df_to_save.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    def plot_comparison(self, column='Power'):
        """Plot original vs cleaned data for comparison"""
        original = pd.read_csv(self.filepath)
        cleaned = self.df
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original data
        axes[0].hist(original[column], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0].set_title(f'Original {column} Distribution (n={len(original)})')
        axes[0].set_ylabel('Frequency')
        axes[0].set_xlabel(column)
        
        # Cleaned data
        axes[1].hist(cleaned[column], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title(f'Cleaned {column} Distribution (n={len(cleaned)})')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xlabel(column)
        
        plt.tight_layout()
        plt.savefig(f'data_cleaning_comparison_{column}.png', dpi=100)
        print(f"Comparison plot saved to data_cleaning_comparison_{column}.png")
        plt.close()


# Usage Example
if __name__ == "__main__":
    # Initialize cleaner
    cleaner = DataCleaner('/home/cc/summer2024/main_codes/experiment_data/training_data/training_dataset.csv')
    
    # Run cleaning pipeline
    cleaned_df = cleaner.clean(
        remove_dups=True,
        remove_resets=True,
        filter_zscore=True,
        filter_iqr=False,
        smooth=True,
        filter_consistency=True,
        filter_constants=False,
        normalize_rewards=True
    )
    
    # Save cleaned data
    cleaner.save_cleaned_data('/home/cc/summer2024/main_codes/experiment_data/training_data/training_dataset_cleaned.csv')
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    cleaner.plot_comparison('Power')
    cleaner.plot_comparison('Progress')
    cleaner.plot_comparison('Reward')
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print("\nReward statistics:")
    print(cleaned_df['Reward'].describe())
    print("\nProgress statistics:")
    print(cleaned_df['Progress'].describe())
    print("\nPower statistics:")
    print(cleaned_df['Power'].describe())
    
    # Check for remaining duplicates
    if len(cleaned_df) > 0:
        duplicates = cleaned_df.duplicated(subset=['Progress', 'Power', 'Action']).sum()
        print(f"\nRemaining duplicate state-action pairs: {duplicates}")
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def get_random_walk_weights(current_action, actions, sigma=2.0, center_bias=0.0):
    """Generate weights for random walk - higher probability for nearby actions
    
    Args:
        current_action: Current PCAP value
        actions: List of possible actions
        sigma: Controls random walk tightness (lower = tighter)
        center_bias: Controls preference for middle actions (0 = no bias, higher = stronger bias toward center)
    """
    current_idx = actions.index(current_action)
    middle_idx = len(actions) // 2
    weights = []
    
    for i, action in enumerate(actions):
        # Random walk component: favor nearby actions
        distance = abs(i - current_idx)
        walk_weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        # Center bias component: favor middle actions
        if center_bias > 0:
            distance_from_center = abs(i - middle_idx)
            center_weight = np.exp(-(distance_from_center ** 2) / (2 * center_bias ** 2))
            weight = walk_weight * center_weight
        else:
            weight = walk_weight
            
        weights.append(weight)
    
    # Normalize weights
    total = sum(weights)
    return [w / total for w in weights]


def sample_random_walk(current_action, actions, n_samples=100, sigma=2.0, center_bias=0.0):
    """Sample actions using random walk weights
    
    Args:
        current_action: Starting action
        actions: List of possible actions
        n_samples: Number of samples to draw
        sigma: Controls random walk tightness
        center_bias: Controls center bias
        
    Returns:
        List of sampled actions
    """
    weights = get_random_walk_weights(current_action, actions, sigma=sigma, center_bias=center_bias)
    sampled_actions = np.random.choice(actions, size=n_samples, p=weights)
    return sampled_actions


# Test suite
def test_weights_sum_to_one():
    """Test that weights always sum to approximately 1.0"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    test_cases = [
        (78.0, 2.0, 0.0),
        (118.0, 2.0, 0.0),
        (165.0, 2.0, 0.0),
        (118.0, 0.5, 1.0),
        (78.0, 1.0, 2.0),
    ]
    
    for current_action, sigma, center_bias in test_cases:
        weights = get_random_walk_weights(current_action, ACTIONS, sigma=sigma, center_bias=center_bias)
        total = sum(weights)
        assert abs(total - 1.0) < 1e-10, f"Weights don't sum to 1.0: {total} (current={current_action}, sigma={sigma}, center_bias={center_bias})"
    
    print("✓ All weights sum to 1.0")


def test_all_weights_positive():
    """Test that all weights are non-negative"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    weights = get_random_walk_weights(118.0, ACTIONS, sigma=2.0, center_bias=1.0)
    assert all(w >= 0 for w in weights), f"Found negative weights: {weights}"
    
    print("✓ All weights are non-negative")


def test_current_action_highest_probability():
    """Test that current action has highest probability (no center_bias)"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    for current_action in [78.0, 118.0, 165.0]:
        weights = get_random_walk_weights(current_action, ACTIONS, sigma=2.0, center_bias=0.0)
        current_idx = ACTIONS.index(current_action)
        max_weight_idx = weights.index(max(weights))
        assert max_weight_idx == current_idx, f"Current action {current_action} doesn't have max weight"
    
    print("✓ Current action has highest probability (without center_bias)")


def test_symmetry():
    """Test that weights are symmetric around current action"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    # Use middle action for true symmetry
    current_action = ACTIONS[len(ACTIONS) // 2]
    weights = get_random_walk_weights(current_action, ACTIONS, sigma=2.0, center_bias=0.0)
    
    current_idx = ACTIONS.index(current_action)
    for offset in range(1, min(4, current_idx + 1)):
        left_weight = weights[current_idx - offset]
        right_weight = weights[current_idx + offset]
        assert abs(left_weight - right_weight) < 1e-10, f"Asymmetric weights at offset {offset}"
    
    print("✓ Weights are symmetric around current action")


def test_sigma_effect():
    """Test that smaller sigma creates tighter distribution"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    weights_tight = get_random_walk_weights(118.0, ACTIONS, sigma=0.5, center_bias=0.0)
    weights_loose = get_random_walk_weights(118.0, ACTIONS, sigma=5.0, center_bias=0.0)
    
    # Tighter distribution should have higher weight at current action
    current_idx = ACTIONS.index(118.0)
    assert weights_tight[current_idx] > weights_loose[current_idx], "Smaller sigma should increase current action weight"
    
    print("✓ Smaller sigma creates tighter distribution")


def test_center_bias_effect():
    """Test that center_bias increases weight toward center"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    # Start from edge
    weights_no_bias = get_random_walk_weights(78.0, ACTIONS, sigma=2.0, center_bias=0.0)
    weights_with_bias = get_random_walk_weights(78.0, ACTIONS, sigma=2.0, center_bias=2.0)
    
    middle_idx = len(ACTIONS) // 2
    # Center should get more weight with bias
    assert weights_with_bias[middle_idx] > weights_no_bias[middle_idx], "Center_bias should increase middle weight"
    
    print("✓ Center bias pulls weight toward center actions")


def test_invalid_action():
    """Test behavior with invalid action"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    try:
        weights = get_random_walk_weights(999.0, ACTIONS, sigma=2.0, center_bias=0.0)
        assert False, "Should raise ValueError for invalid action"
    except ValueError:
        print("✓ Raises ValueError for invalid action")


def test_sampling_around_action():
    """Test sampling around a specific action"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    # Sample around 95
    samples = sample_random_walk(95.0, ACTIONS, n_samples=1000, sigma=1.0, center_bias=0.0)
    mean_sample = np.mean(samples)
    
    # Mean should be close to 95
    assert abs(mean_sample - 95.0) < 10.0, f"Samples should be centered around 95, got mean {mean_sample}"
    print(f"✓ Sampling around 95: mean={mean_sample:.2f}, std={np.std(samples):.2f}")


def visualize_weights():
    """Create visualizations of the weight distributions"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: No bias, tight sigma
    ax = axes[0, 0]
    weights = get_random_walk_weights(118.0, ACTIONS, sigma=0.5, center_bias=0.0)
    ax.bar(range(len(ACTIONS)), weights, color='steelblue')
    ax.axvline(ACTIONS.index(118.0), color='red', linestyle='--', label='Current (118.0)')
    ax.set_title('Tight Distribution (sigma=0.5, center_bias=0)')
    ax.set_ylabel('Weight')
    ax.legend()
    
    # Plot 2: No bias, loose sigma
    ax = axes[0, 1]
    weights = get_random_walk_weights(118.0, ACTIONS, sigma=5.0, center_bias=0.0)
    ax.bar(range(len(ACTIONS)), weights, color='steelblue')
    ax.axvline(ACTIONS.index(118.0), color='red', linestyle='--', label='Current (118.0)')
    ax.set_title('Loose Distribution (sigma=5.0, center_bias=0)')
    ax.set_ylabel('Weight')
    ax.legend()
    
    # Plot 3: From edge with center bias
    ax = axes[1, 0]
    weights = get_random_walk_weights(78.0, ACTIONS, sigma=2.0, center_bias=2.0)
    ax.bar(range(len(ACTIONS)), weights, color='steelblue')
    ax.axvline(ACTIONS.index(78.0), color='red', linestyle='--', label='Current (78.0)')
    ax.axvline(len(ACTIONS) // 2, color='green', linestyle='--', label='Center')
    ax.set_title('From Edge with Center Bias (sigma=2.0, center_bias=2.0)')
    ax.set_ylabel('Weight')
    ax.legend()
    
    # Plot 4: Strong center bias
    ax = axes[1, 1]
    weights = get_random_walk_weights(165.0, ACTIONS, sigma=2.0, center_bias=5.0)
    ax.bar(range(len(ACTIONS)), weights, color='steelblue')
    ax.axvline(ACTIONS.index(165.0), color='red', linestyle='--', label='Current (165.0)')
    ax.axvline(len(ACTIONS) // 2, color='green', linestyle='--', label='Center')
    ax.set_title('Strong Center Bias (sigma=2.0, center_bias=5.0)')
    ax.set_ylabel('Weight')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('random_walk_weights_test.png', dpi=100)
    print("✓ Visualization saved to 'random_walk_weights_test.png'")
    plt.show()


def visualize_sampling_around_95():
    """Visualize continuous sampling around 95"""
    ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample with tight sigma
    samples_tight = sample_random_walk(95.0, ACTIONS, n_samples=1000, sigma=0.5, center_bias=0)
    ax = axes[0]
    ax.hist(samples_tight, bins=ACTIONS, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(95.0, color='red', linestyle='--', linewidth=2, label='Target (95.0)')
    ax.set_title(f'Sampling Around 95 (sigma=0.5)\nMean: {np.mean(samples_tight):.2f}, Std: {np.std(samples_tight):.2f}')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Sample with loose sigma
    samples_loose = sample_random_walk(95.0, ACTIONS, n_samples=1000, sigma=2.0, center_bias=0)
    ax = axes[1]
    ax.hist(samples_loose, bins=ACTIONS, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(95.0, color='red', linestyle='--', linewidth=2, label='Target (95.0)')
    ax.set_title(f'Sampling Around 95 (sigma=2.0)\nMean: {np.mean(samples_loose):.2f}, Std: {np.std(samples_loose):.2f}')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('sampling_around_95.png', dpi=100)
    print("✓ Sampling visualization saved to 'sampling_around_95.png'")
    plt.show()


if __name__ == "__main__":
    print("Running tests for get_random_walk_weights()...\n")
    
    test_weights_sum_to_one()
    test_all_weights_positive()
    test_current_action_highest_probability()
    test_symmetry()
    test_sigma_effect()
    test_center_bias_effect()
    test_invalid_action()
    test_sampling_around_action()
    
    print("\n✓ All tests passed!\n")
    print("Generating visualizations...")
    visualize_weights()
    visualize_sampling_around_95()
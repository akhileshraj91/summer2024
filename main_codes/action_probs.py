import numpy as np

ACTIONS = [78.0, 83.0, 89.0, 95.0, 101.0, 107.0, 112.0, 118.0, 124.0, 130.0, 136.0, 141.0, 147.0, 153.0, 159.0, 165.0]

def get_random_walk_weights(current_action, actions, sigma=2.0, center_bias=0.0):
    current_idx = actions.index(current_action)
    middle_idx = len(actions) // 2
    weights = []
    
    for i, action in enumerate(actions):
        distance = abs(i - current_idx)
        walk_weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        if center_bias > 0:
            distance_from_center = abs(i - middle_idx)
            center_weight = np.exp(-(distance_from_center ** 2) / (2 * center_bias ** 2))
            weight = walk_weight * center_weight
        else:
            weight = walk_weight
            
        weights.append(weight)
    
    total = sum(weights)
    return [w / total for w in weights]

print('WITHOUT center bias (center_bias=0):')
print('='*60)
print('From edge (78.0):')
weights = get_random_walk_weights(118.0, ACTIONS, sigma=2.0, center_bias=0.0)
for action, prob in zip(ACTIONS, weights):
    bar = '#' * int(prob * 100)
    print(f'{action:6.1f} W: {prob:6.2%} {bar}')

print()
print('WITH center bias (center_bias=3.0):')
print('='*60)
print('From edge (78.0):')
weights = get_random_walk_weights(118.0, ACTIONS, sigma=0.5, center_bias=1.0)
for action, prob in zip(ACTIONS, weights):
    bar = '#' * int(prob * 100)
    print(f'{action:6.1f} W: {prob:6.2%} {bar}')

print()
print('From middle (118.0):')
weights = get_random_walk_weights(118.0, ACTIONS, sigma=2.0, center_bias=3.0)
for action, prob in zip(ACTIONS, weights):
    bar = '#' * int(prob * 100)
    print(f'{action:6.1f} W: {prob:6.2%} {bar}')

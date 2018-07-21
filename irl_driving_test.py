"""
This scripts runs an approximate form of maximum causal
entropy IRL on the driving domain.

We generate a fixed number of demonstrations of the target
task, then iteratively add more data to the set of trajectories
used for value iteration to compute the cost function. MCE-IRL
uses cost functions that are linear in some feature space, so
for right now we can use radial basis functions over the
2D coordinates of the agent's car.

We may not end up using MCE-IRL because of its complexity, perhaps
generative adversarial imitation learning instead?
"""

import domains.driving as driving
import tensorflow as tf

# Experiment parameters
timestep_size = 0.05
max_timesteps = 500


num_demonstrations = 10
num_irl_phases = 10
num_exploration_episodes = 10
num_estimation_episodes = 10
num_value_iterations = 100
num_gradient_updates = 100

# Initialize environment

# Generate demonstrations

# Build cost function representation

# Build Q-function representation

# Run MCE-IRL, while evaluating policies

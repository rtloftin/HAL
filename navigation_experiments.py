"""
Conducts a set of experiments in the robot navigation domain
"""

import navigation as nav
import tensorflow as tf
import os
import sys

# Get data directory
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = "/home/tyler/Desktop/navigation_results"

dir_index = 0

while os.path.isdir(data_dir + "/results_" + str(dir_index)):
    dir_index += 1

results_dir = data_dir + "/results_" + str(dir_index)
os.makedirs(results_dir)

# Record current source file
with open(__file__) as source:
    with open(results_dir + "/source.py", "w") as source_copy:
        source_copy.write(source.read())

# Suppress TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Construct environment
env, sensor = nav.one_room()
# env, sensor = nav.three_rooms()

depth = (env.width + env.height) * 2

# Maximum likelihood IRL
ml_irl = nav.ml_irl(env,
                    beta=1.0,
                    gamma=0.99,
                    planning_depth=depth,
                    penalty=100.,
                    learning_rate=0.001,
                    batch_size=128,
                    num_batches=500)

# Model-Based IRL
model_based = nav.model_based(beta=1.0,
                              gamma=0.99,
                              planning_depth=depth,
                              obstacle_prior=0.2,
                              penalty=100.,
                              learning_rate=0.001,
                              batch_size=128,
                              pretrain_batches=100,
                              online_batches=50)

# Standard BAM
bam = nav.bam(beta=1.0,
              gamma=0.99,
              planning_depth=depth,
              obstacle_mean=-0.5,
              obstacle_variance=0.2,
              penalty=100.,
              learning_rate=0.001,
              batch_size=128,
              pretrain_batches=100,
              online_batches=50)

# Abstract BAM
grid = nav.abstract_grid(env.width, env.height,
                         h_step=2,
                         v_step=2,
                         planning_depth=depth,
                         gamma=0.99,
                         beta=1.0,
                         link_mean=1.,
                         link_penalty=10.0,
                         reward_penalty=100.)

abstract_bam = nav.abstract_bam(grid,
                                beta=1.0,
                                learning_rate=0.001,
                                batch_size=128,
                                pretrain_batches=100,
                                online_batches=50)

# Select algorithms
algorithms = dict()
algorithms["ML-IRL"] = ml_irl
algorithms["Model-Based"] = model_based
algorithms["BAM"] = bam
algorithms["Abstract-BAM"] = abstract_bam

# run experiments
nav.experiment(algorithms, env, sensor,
               sessions=50,
               demonstrations=10,
               episodes=10,
               baselines=100,
               evaluations=200,
               max_steps=depth,
               results_dir=results_dir)

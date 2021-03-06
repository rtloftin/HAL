"""
Conducts a set of experiments in the robot navigation domain
"""

import navigation as nav
import tensorflow as tf
import os
import sys

# env, sensor = nav.one_room()
# env, sensor = nav.three_rooms()
# env, sensor = nav.big_one_room()
# env, sensor = nav.big_three_rooms()
# env, sensor = nav.barricades()
# nav.visualize(env, sensor)
# sys.exit(0)

# Get data directory
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = "/home/tyler/Desktop/navigation_results"
    # data_dir = "/home/rtloftin/nav_results"

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
# env, sensor = nav.one_room()
# env, sensor = nav.three_rooms()
# env, sensor = nav.big_one_room()
# env, sensor = nav.big_three_rooms()
env, sensor = nav.barricades()
# env, sensor = nav.more_barricades()

# Override sensor
sensor = nav.RoundSensor(env, 3)

depth = (env.width + env.height) * 2

# Maximum likelihood IRL
ml_irl = nav.ml_irl(beta=1.0,
                    gamma=0.99,
                    planning_depth=depth,
                    penalty=100.,
                    learning_rate=0.01,
                    batch_size=128,
                    num_batches=1000,
                    rms_prop=True,
                    use_baseline=True)

# Model-Based IRL
model_based = nav.model_based(beta=1.0,
                              gamma=0.99,
                              planning_depth=depth,
                              obstacle_prior=0.0,
                              penalty=100.,
                              learning_rate=0.01,
                              batch_size=128,
                              pretrain_batches=500,
                              online_batches=100,
                              rms_prop=True,
                              use_baseline=True)

# Standard BAM
bam = nav.bam(beta=1.0,
              gamma=0.99,
              planning_depth=depth,
              obstacle_mean=-0.5,
              obstacle_variance=1.0,
              penalty=100.,
              learning_rate=0.01,
              batch_size=128,
              pretrain_batches=500,
              online_batches=0,
              rms_prop=True,
              use_baseline=True)

# Abstract BAM
grid_10 = nav.abstract_grid(env.width, env.height,
                            h_step=10,
                            v_step=10,
                            planning_depth=depth,
                            gamma=0.99,
                            beta=1.0,
                            abstract_mean=5.,
                            abstract_penalty=.01,
                            reward_penalty=100.,
                            use_baseline=True)

grid_5 = nav.abstract_grid(env.width, env.height,
                           h_step=5,
                           v_step=5,
                           planning_depth=depth,
                           gamma=0.99,
                           beta=1.0,
                           abstract_mean=5.,
                           abstract_penalty=.01,
                           reward_penalty=100.,
                           use_baseline=True)

grid_2 = nav.abstract_grid(env.width, env.height,
                           h_step=2,
                           v_step=2,
                           planning_depth=depth,
                           gamma=0.99,
                           beta=1.0,
                           abstract_mean=5.,
                           abstract_penalty=.01,
                           reward_penalty=100.,
                           use_baseline=True)

abstract_bam_10 = nav.abstract_bam(grid_10,
                                   beta=1.0,
                                   learning_rate=0.01,
                                   batch_size=128,
                                   pretrain_batches=500,
                                   online_batches=100,
                                   rms_prop=True)

abstract_bam_5 = nav.abstract_bam(grid_5,
                                  beta=1.0,
                                  learning_rate=0.01,
                                  batch_size=128,
                                  pretrain_batches=500,
                                  online_batches=100,
                                  rms_prop=True)

abstract_bam_2 = nav.abstract_bam(grid_2,
                                  beta=1.0,
                                  learning_rate=0.01,
                                  batch_size=128,
                                  pretrain_batches=500,
                                  online_batches=100,
                                  rms_prop=True)


# Select algorithms
algorithms = dict()
# algorithms["ML-IRL"] = ml_irl
algorithms["Model-Based"] = model_based
algorithms["BAM"] = bam
# algorithms["Abstract-BAM-dummy"] = abstract_bam_dummy
algorithms["Abstract-BAM-10x10"] = abstract_bam_10
algorithms["Abstract-BAM-5x5"] = abstract_bam_5
# algorithms["Abstract-BAM-2x2"] = abstract_bam_2

# run experiments
nav.experiment(algorithms, env, sensor,
               sessions=10,
               demonstrations=5,
               episodes=8,
               baselines=100,
               evaluations=200,
               max_steps=depth,
               results_dir=results_dir)

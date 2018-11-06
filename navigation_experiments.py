"""
Conducts a set of experiments in the robot navigation domain
"""

import navigation as nav
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# env, sensor = nav.one_room()
env, sensor = nav.three_rooms()

depth = (env.width + env.height) * 2

ml_irl = nav.ml_irl(env, planning_depth=depth)
model_based = nav.model_based(planning_depth=depth)
bam = nav.bam(planning_depth=depth)

grid = nav.abstract_grid(env.width, env.height, planning_depth=depth)
abstract_bam = nav.abstract_bam(grid)

algorithms = dict()
algorithms["ML-IRL"] = ml_irl
algorithms["Model-Based"] = model_based
algorithms["BAM"] = bam
# algorithms["Abstract-BAM"] = abstract_bam

nav.experiment(algorithms, env, sensor,
               demonstrations=1,
               sessions=10,
               episodes=10,
               max_steps=depth,
               results_file="/home/tyler/Desktop/nav_data")

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

bam = nav.bam(planning_depth=depth)
ml_irl = nav.ml_irl(planning_depth=depth)

grid = nav.abstract_grid(env.width, env.height, planning_depth=depth)
abstract_bam = nav.abstract_bam(grid)

algorithms = {
    "BAM": bam,
    "ML-IRL": ml_irl
    # "Abstract BAM": abstract_bam
}

nav.experiment(algorithms, env, sensor,
               demonstrations=5,
               sessions=10,
               episodes=10,
               max_steps=depth,
               results_file="/home/tyler/Desktop/nav_data")

"""
Conducts a set of experiments in the robot navigation domain
"""

import navigation as nav
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env, sensor = nav.one_room()
# env, sensor = nav.three_rooms()

# bam = nav.bam()
ml_irl = nav.ml_irl()

nav.sensor_experiment(env, sensor, {
    # "BAM": bam,
    "ML-IRL": ml_irl
})

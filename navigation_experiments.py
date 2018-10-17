"""
Conducts a set of experiments in the robot navigation domain
"""

import navigation as nav

# env, sensor = nav.one_room()
env, sensor = nav.three_rooms()

ml_irl = nav.ml_irl()

nav.sensor_experiment(env, sensor, {
    "ML-IRL": ml_irl
})

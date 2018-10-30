"""
Defines methods for running learning experiments in the robot navigation domain.
"""

from .demonstrations import Demonstrations
from .expert import Expert

import time
import os
import numpy as np
import matplotlib as pl


def experiment(algorithms, env, sensor,
               sessions=5,
               demonstrations=1,
               episodes=10,
               baselines=100,
               max_steps=None,
               results_file=None):
    """
    Runs an experiment comparing a collection of algorithms in a given  environment.

    :param algorithms: a dictionary of algorithms to evaluate
    :param env: the environment in which to run the experiments
    :param sensor: the base sensor model for this environment
    :param sessions: the number of training sessions to run for each algorithm
    :param demonstrations: the number of demonstrations to generate for each session
    :param episodes: the number of episodes of each task to run for each session
    :param baselines: the number of episodes to run to estimate the expert's performance
    :param max_steps: the maximum number of steps per episode
    :param results_file: the file which to store the results, if it exists, create a new file
    """

    # Construct expert
    expert = Expert(env)

    # Compute step limit if not provided
    if max_steps is None:
        max_steps = (env.width + env.height) * 2

    # Generate baseline estimate of expert performance
    baseline = .0

    for task, _ in env.tasks:
        expert.task(task)

        for _ in range(baselines):
            env.reset(task=task)
            step = 0

            while not env.complete and step < max_steps:
                env.update(expert.act(env.x, env.y))
                step += 1

            baseline += step

    baseline /= baselines

    # Initialize results data structure
    results = dict()

    for name in algorithms.keys():
        results[name] = []

    # Run experiments
    for sess in range(sessions):
        print("session " + str(sess))

        # Generate demonstrations
        session_data = Demonstrations()
        session_sensor = sensor.clone()

        for task, _ in env.tasks:
            expert.task(task)

            for _ in range(demonstrations):
                env.reset(task=task)
                session_sensor.update()
                session_data.new(task)
                step = 0

                while not env.complete and step < max_steps:
                    action = expert.act(env.x, env.y)
                    session_data.step(env.x, env.y, action)
                    env.update(action)
                    session_sensor.update()
                    step += 1

        # Evaluate algorithms
        for name, algorithm in algorithms.items():
            print("algorithm - " + name)
            agent_sensor = session_sensor.clone()

            with algorithm(agent_sensor, session_data) as agent:
                returns = session(agent, env, agent_sensor, episodes=episodes, max_steps=max_steps)

            results[name].append(returns / baseline)

    # Compute means
    means = dict()

    for algorithm, result in results.items():
        mean = np.zeros(episodes, dtype=np.float32)

        for returns in result:
            mean += returns

        means[algorithm] = mean / len(result)

    # Print results
    for algorithm, mean in means.items():
        print(algorithm + ": " + str(mean))

    # Save results
    if results_file is not None:
        index = 0

        while os.path.exists(results_file + "_" + str(index)):
            index += 1

        with open(results_file + "_" + str(index), "w") as file:
            algorithms = means.keys()

            columns = ["episodes"]
            columns.extend(algorithms)
            file.write(" ".join(columns) + "\n")

            for episode in range(episodes):
                row = [str(episode + 1)]

                for algorithm in algorithms:
                    row.append(str(means[algorithm][episode]))

                file.write(" ".join(row) + "\n")


def session(agent, env, sensor, episodes=10, max_steps=100):
    """
    Runs a single learning session with a single agent.

    :param agent: the learning agent
    :param sensor: the agent's local sensor model
    :param env: the environment in which the agent learns
    :param episodes: the number of learning episodes to run
    :param max_steps: the maximum number of steps per episode
    :param interval: the number of steps between each learning update
    :return: an array of total costs (across all tasks) for each episode
    """

    costs = np.empty(episodes, dtype=np.float32)

    for episode in range(episodes):
        total = 0.
        success = 0
        start = time.time()

        for task, _ in env.tasks:
            env.reset(task=task)
            agent.task(task)
            sensor.update()
            step = 0

            while not env.complete and step < max_steps:
                env.update(agent.act(env.x, env.y))
                sensor.update()
                step += 1

            agent.update()
            total += step

            if env.complete:
                success += 1

        print("episode took " + str(time.time() - start) + " seconds")
        print("completed " + str(success) + " tasks")

        costs[episode] = total

    return costs

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
               evaluations=200,
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
    :param evaluations: the number of non-training episodes to run to evaluate the agent's policies
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
    costs = dict()
    successes = dict()

    for name in algorithms.keys():
        costs[name] = []
        successes[name] = []

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

                session_data.step(env.x, env.y, expert.act(env.x, env.y))

        # Evaluate algorithms
        for name, algorithm in algorithms.items():
            print("algorithm - " + name)
            agent_sensor = session_sensor.clone()

            with algorithm(agent_sensor, session_data) as agent:
                cost, success = session(agent, env, agent_sensor,
                                        episodes=episodes,
                                        evaluations=evaluations,
                                        max_steps=max_steps)

            costs[name].append(cost / baseline)
            successes[name].append(success)

    # Compute means
    mean_costs = dict()
    mean_successes = dict()

    for algorithm in algorithms.keys():
        mean_cost = np.zeros(episodes, dtype=np.float32)
        mean_success = np.zeros(episodes, dtype=np.float32)

        for cost in costs[algorithm]:
            mean_cost += cost

        for success in successes[algorithm]:
            mean_success += success

        mean_costs[algorithm] = mean_cost / len(costs[algorithm])
        mean_successes[algorithm] = mean_success / len(successes[algorithm])

    # Print results
    for algorithm in algorithms.keys():
        print(algorithm + " ----")
        print("mean cost: " + str(mean_costs[algorithm]))
        print("mean success rate: " + str(mean_successes[algorithm]))

    # Save results
    if results_file is not None:
        index = 0

        while os.path.exists(results_file + "_" + str(index)):
            index += 1

        with open(results_file + "_" + str(index), "w") as file:
            algorithms = algorithms.keys()

            columns = ["episodes"]
            columns.extend(algorithms)
            file.write(" ".join(columns) + "\n")

            for episode in range(episodes):
                row = [str(episode + 1)]

                for algorithm in algorithms:
                    row.append(str(mean_costs[algorithm][episode]))

                file.write(" ".join(row) + "\n")


def session(agent, env, sensor, episodes=10, evaluations=50, max_steps=100):
    """
    Runs a single learning session with a single agent.

    :param agent: the learning agent
    :param sensor: the agent's local sensor model
    :param env: the environment in which the agent learns
    :param episodes: the number of learning episodes to run
    :param evaluations: the number of non-training episodes to run to evaluate the agent's policies
    :param max_steps: the maximum number of steps per episode
    :return: the average number of steps required at teach episode, the average number of tasks completed
    """

    costs = np.empty(episodes, dtype=np.float32)
    successes = np.empty(episodes, dtype=np.float32)

    for episode in range(episodes):
        start = time.time()

        # Run learning episode
        for task, _ in env.tasks:
            env.reset(task=task)
            sensor.update()
            step = 0

            while not env.complete and step < max_steps:
                agent.task(task)
                env.update(agent.act(env.x, env.y))
                sensor.update()
                step += 1

        # Update agent
        agent.update()

        # Evaluate policies
        steps = 0.
        success = 0.

        for task, _ in env.tasks:
            agent.task(task)

            for _ in range(evaluations):
                env.reset(task=task)
                step = 0

                while not env.complete and step < max_steps:
                    env.update(agent.act(env.x, env.y))
                    step += 1

                steps += step

                if env.complete:
                    success += 1

        costs[episode] = steps / evaluations
        successes[episode] = success / (evaluations * len(env.tasks))

        print("episode took " + str(time.time() - start) + " seconds, success rate: " + str(successes[episode]))

    return costs, successes

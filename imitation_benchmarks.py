"""
Methods to evaluate and tune the different imitation
learning algorithms we are developing.
"""

import imitation
import models
import domains.robots

import collections
import sys
import numpy as np


def generate(environment, demonstrations=100, steps=500):
    """
    Generates a dataset of demonstrations for all of the tasks
    defined in the given environment.

    :param environment: the environment in which to generate the demonstrations
    :param demonstrations: the number of demonstration to give of each task
    :param steps: the maximum number of steps to allow
    :return: the generated data set
    """

    data = imitation.Dataset()

    for task in environment.get_tasks():
        for demonstration in range(demonstrations):
            print("Demonstration " + str(demonstration))

            environment.reset(task=task)
            data.new(task)
            step = 0

            while not environment.complete and (step < steps):
                action = environment.expert()
                data.step(environment.state, action)
                environment.update(action)
                step += 1

    return data


def benchmark(agent, environment, data, episodes=1000, steps=500, window=20):
    """
    Trains an imitation learning agent on a given set of demonstrations then
    allows the agent to interact with the environment itself while its return
    on each task is recorded.

    :param agent: the agent being trained
    :param environment: the environment in which the agent is learning
    :param data: the task demonstrations
    :param episodes: the number of evaluation episodes to allow
    :param steps: the number of steps to allow
    :param window: the window used for the running average
    """

    # Train the agent
    agent.demonstrate(data)

    # Evaluate the agent
    results = collections.deque()
    average = 0.0

    for episode in range(episodes):
        value = 0

        for task in environment.get_tasks():
            environment.reset(task=task)
            agent.reset(task=task)
            step = 0

            while not environment.complete and (step < steps):
                environment.update(agent.act(environment.state))
                value += environment.reward
                step += 1

        average += value
        results.append(value)

        if len(results) > window:
            average -= results.popleft()

        print("Episode " + str(episode) + ", total return: " + str(average / len(results)))


# env = domains.robots.ant()
env = domains.robots.hopper()

actor_fn = models.dense_sigmoid([2] + list(env.action_space.shape), hidden_layers=2, hidden_nodes=128)
critic_fn = models.dense_sigmoid([1], hidden_layers=2, hidden_nodes=128)
cost_fn = models.dense_sigmoid([1], hidden_layers=2, hidden_nodes=64)

# cloning = imitation.cloning(actor_fn, env.state_space, env.action_space,
#                             learning_rate=0.001,
#                             batch_size=256,
#                             num_batches=1000)

gail_ppo = imitation.gail_ppo(actor_fn, critic_fn, cost_fn, env.state_space, env.action_space,
                              penalty=0.0,
                              discount=0.99,
                              mixing=0.9,
                              learning_rate=0.001,
                              clip_epsilon=0.1,
                              batch_size=128,
                              num_batches=1,
                              num_episodes=1)

data = generate(env, demonstrations=10)

with gail_ppo as agent:
    benchmark(agent, env, data)

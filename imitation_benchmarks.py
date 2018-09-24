"""
Methods to evaluate and tune the different imitation
learning algorithms we are developing.
"""

import algorithms
import models
import domains.robots

import collections


def benchmark(agent, environment, demonstrations=100, episodes=100, steps=500, window=20):
    """
    Trains an imitation learning agent on all the tasks in the given environment using a fixed
    number of demonstrations, then allows the agent to interact with the environment itself while
    its return on each task is recorded.

    :param agent: the agent being trained
    :param environment: the environment in which the agent is learning
    :param demonstrations: the number of demonstrations of each task
    :param episodes: the number of evaluation episodes to allow
    :param steps: the number of steps to allow
    :param window: the window used for the running average
    """

    with agent:

        # Demonstrate each task
        for task in environment.get_tasks():
            environment.set_task(task)
            agent.set_task(task)

            for demonstration in range(demonstrations):
                print('Task: ' + task + ', demonstration ' + str(demonstration))

                environment.reset()
                agent.reset()
                step = 0

                while not environment.complete and (step < steps):
                    action = environment.expert()
                    environment.update(action)
                    agent.demonstrate(environment.state, action)
                    step += 1

        agent.incorporate()

        # Evaluate the agent
        results = collections.deque()
        average = 0.0

        for episode in range(episodes):
            value = 0

            for task in environment.get_tasks():
                environment.set_task(task)
                environment.reset()

                agent.set_task(task)
                agent.reset()

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


env = domains.robots.ant()
# env = domains.robots.hopper()

actor_fn = models.dense_sigmoid(env.state_space.shape, [2] + list(env.action_space.shape),
                                hidden_layers=2, hidden_nodes=200)

agent = algorithms.behavioral_cloning(actor_fn, env.state_space, env.action_space,
                                      learning_rate=0.001,
                                      batch_size=128,
                                      num_batches=3000)

benchmark(agent, env)

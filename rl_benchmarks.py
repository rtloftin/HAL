"""
Methods to evaluate the different RL algorithms that
form the basis of the interactive learning algorithms.
"""

import algorithms
import models
import domains.robots

import collections


def benchmark(agent, environment, episodes=1000, steps=500, window=20):

    results = collections.deque()
    average = 0.0

    for episode in range(episodes):
        environment.reset()
        agent.reset()
        value = 0.0
        step = 0

        while not environment.complete and (step < steps):
            environment.update(agent.act(environment.state))
            agent.reward(environment.reward)
            value += environment.reward
            step += 1

        average += value
        results.append(value)

        if len(results) > window:
            average -= results.popleft()

        print("Episode " + str(episode) + ", return: " + str(average / len(results)))


env = domains.robots.ant()
# env = domains.robots.hopper()

model_fn = models.dense_sigmoid(env.state_space.shape, [2] + list(env.action_space.shape),
                                hidden_layers=2, hidden_nodes=100)

agent = algorithms.ppo(model_fn, env.state_space, env.action_space,
                       discount=0.99,
                       learning_rate=0.0005,
                       clip_epsilon=0.05,
                       batch_size=10,
                       num_batches=20,
                       num_episodes=10)

benchmark(agent, env, episodes=10000, window=10)

"""
Methods to evaluate the different RL algorithms that
form the basis of the interactive learning algorithms.
"""

import rl
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

actor_fn = models.dense_sigmoid([2] + list(env.action_space.shape), hidden_layers=2, hidden_nodes=200)
critic_fn = models.dense_sigmoid([1], hidden_layers=2, hidden_nodes=200)

# ppo = rl.ppo(actor_fn, env.state_space, env.action_space,
#                  discount=0.99,
#                       learning_rate=0.0005,
#                       clip_epsilon=0.1,
#                       batch_size=200,
#                       num_batches=10,
#                       num_episodes=10)

ppo_ac = rl.ppo_ac(actor_fn, critic_fn, env.state_space, env.action_space,
                   discount=0.99,
                   mixing=0.9,
                   learning_rate=0.001,
                   clip_epsilon=0.1,
                   batch_size=256,
                   num_batches=50,
                   num_episodes=5)

with ppo_ac as agent:
    benchmark(agent, env, episodes=3000, window=30)

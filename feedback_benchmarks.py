"""
Methods to evaluate and tune the different algorithms
for learning from feedback we are developing.
"""

import models
import feedback
import feedback.teachers
import domains.robots

import collections


def benchmark(agent, environment, teacher, episodes=1000, steps=500, window=50):
    results = collections.deque()
    average = 0.0

    teacher.environment(environment)

    for episode in range(episodes):
        value = 0.0

        for task in environment.get_tasks():
            environment.reset(task=task)
            agent.reset(task=task)
            step = 0

            while not environment.complete and (step < steps):
                state = environment.state
                action = agent.act(state)
                feedback = teacher.feedback(action)
                agent.feedback(state, action, feedback)

                environment.update(action)
                value += environment.reward
                step += 1

        average += value
        results.append(value)

        if len(results) > window:
            average -= results.popleft()

        print("Episode " + str(episode) + ", total return: " + str(average / len(results)))


# env = domains.robots.ant()
env = domains.robots.hopper()

sabl = feedback.teachers.sabl(mu_plus=0.0, mu_minus=0.0, action_tolerance=0.05)

value_fn = models.dense_sigmoid([1], hidden_layers=2, hidden_nodes=128)
advantage_fn = models.dense_sigmoid([2] + list(env.action_space.shape), hidden_layers=2, hidden_nodes=128)

tamer = feedback.tamer(value_fn, advantage_fn, env.state_space, env.action_space,
                       epsilon=0.2,
                       learning_rate=0.01,
                       batch_size=512,
                       num_batches=100)

with tamer as agent, sabl as teacher:
    benchmark(agent, env, teacher)

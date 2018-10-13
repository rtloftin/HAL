"""
Defines methods for running learning experiments in the robot navigation domain.
"""

from .demonstrations import Demonstrations
from .expert import Expert


def session(env, sensor, agent, episodes=10, steps=None, interval=10):
    """
    Runs a single learning session with a single agent

    TODO: INCOMPLETE

    :param env: the environment in which the agent learns
    :param sensor: the agent's local sensor model
    :param agent: the learning agent
    :param episodes: the number of learning episodes to run
    :param steps: the maximum number of steps per episode
    :param interval: the number of steps between each learning update
    :return: a list of total costs (across all tasks) for each episode
    """

    steps = (env.width + env.height) * 4 if steps is None else steps
    costs = []

    for episode in range(episodes):
        total = 0

        for task, _ in env.tasks:
            print("Episode " + str(episode) + ", task: " + task)

            env.reset(task=tasks)
            agent.task(task)
            sensor.update()
            step = 0

            while not env.complete and step < steps:
                env.update(agent.act(env.x, env.y))
                sensor.update()
                step += 1

            total += step

        costs.append(total)

    return costs


def sensor_experiment(env, sensor, algorithms, sessions=10, demonstrations=5,  episodes=100):
    """
    Runs an evaluation of a given set of learning algorithms, uses a sensor model
    that allows the agent to directly observe the environment

    :param env: the navigation environment in which to evaluate the algorithms
    :param sensor: the base sensor attached to the environment
    :param algorithms: a dictionary of builders for the different learning agents
    :param sessions: the number of learning sessions to run for each algorithm
    :param demonstrations: the number of demonstrations to provide for each task
    :param episodes: the number of episodes used to evaluate the expert baseline
    """

    # Construct expert
    expert = Expert(env)

    # Generate demonstrations and baseline
    data = Demonstrations()
    baseline = 0.

    for task, _ in env.tasks:

        # Generate demonstrations
        expert.task(task)

        for _ in range(demonstrations):
            env.reset(task=task)
            sensor.update()
            data.new(task)
            step = 0

            while not env.complete and step < ((env.width + env.height) * 4):
                action = expert.act(env.x, env.y)
                data.step(env.x, env.y, action)
                env.update(action)
                sensor.update()
                step += 1

        # Estimate baseline
        task_baseline = 0.

        for _ in range(episodes):
            env.reset(task=task)
            step = 0

            while not env.complete and step < ((env.width + env.height) * 4):
                data.step(env.x, env.y, action)
                env.update(action)
                step += 1

            task_baseline += step

        baseline += task_baseline / demonstrations

    # Evaluate algorithms
    for name, algorithm in algorithms.items():
        for s in range(sessions):
            print(name + ", session: " + str(s))
            agent_sensor = sensor.clone()

            with algorithm(agent_sensor, data) as agent:
                session(env, agent_sensor, agent)

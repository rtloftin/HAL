import gym

from domains.robots.RoboschoolReacher_v0 import Policy as ReacherAgent
from domains.robots.RoboschoolHopper_v0 import Policy as HopperAgent
from domains.robots.RoboschoolAnt_v0 import Policy as AntAgent
from domains.robots.RoboschoolInvertedPendulumSwingup_v0 import Policy as PendulumAgent

# Reacher
# env = gym.make("RoboschoolReacher-v1")
# agent = ReacherAgent(env.observation_space, env.action_space)

# Hopper
# env = gym.make("RoboschoolHopper-v1")
# agent = HopperAgent(env.observation_space, env.action_space)

# Ant
env = gym.make("RoboschoolAnt-v1")
agent = AntAgent(env.observation_space, env.action_space)

# Pendulum
# env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
# agent = PendulumAgent(env.observation_space, env.action_space)

num_episodes = 100
total_reward = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    step = 0
    total = 0

    while (not done) and (step < 500):
        # env.render()
        # time.sleep(0.1)
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        step += 1
        total += reward

    total_reward += total
    print("Episode " + str(episode + 1) + ", " + str(step) + " steps, return: " + str(total))

print("Total reward, agent: ", (total_reward / num_episodes))

total_reward = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    step = 0
    total = 0

    while (not done) and (step < 500):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        step += 1
        total += reward

    total_reward += total
    print("Episode " + str(episode + 1) + ", " + str(step) + " steps, return: " + str(total))

print("Total reward, random: ", (total_reward / num_episodes))

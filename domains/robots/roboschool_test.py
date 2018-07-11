import gym

from domains.robots import SmallReactivePolicy as PendulumAgent

# Reacher
# env = gym.make("RoboschoolReacher-v1")
# agent = ReacherAgent(env.observation_space, env.action_space)

# Hopper
# env = gym.make("RoboschoolHopper-v1")
# agent = HopperAgent(env.observation_space, env.action_space)

# Ant
# env = gym.make("RoboschoolAnt-v1")
# agent = AntAgent(env.observation_space, env.action_space)

# Pendulum
env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
agent = PendulumAgent(env.observation_space, env.action_space)

num_episodes = 100;

for _ in range(num_episodes):
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

print("Total reward, agent: ", (total / num_episodes))

for _ in range(num_episodes):
    state = env.reset()
    done = False
    step = 0
    total = 0

    while (not done) and (step < 500):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        step += 1
        total += reward

print("Total reward, random: ", (total / num_episodes))


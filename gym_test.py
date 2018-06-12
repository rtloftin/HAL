import time
import gym
import numpy as np

# env = gym.make("LunarLander-v2")
# env = gym.make("Acrobot-v1")
env = gym.make("MountainCar-v0")
# env = gym.make("CartPole-v1")

print("State space: ", env.observation_space)
print("Action space: ", env.action_space)
print("Num actions: ", env.action_space.n)

num_episodes = 50000

total_reward = 0.0
success_count = 0.0

for _ in range(num_episodes):
    state = env.reset()
    done = False
    step = 0

    while (not done) and (step < 200):
        # env.render()
        # time.sleep(0.1)
        # action = env.action_space.sample()

        action = np.random.choice([0, 2], 1)[0]

        state, reward, done, _ = env.step(action)
        step += 1
        total_reward += reward

        if done and step < 200:
            success_count += 1.0

print("Random policy, success rate: ", (success_count / num_episodes),
      ", average return: ", (total_reward / num_episodes))

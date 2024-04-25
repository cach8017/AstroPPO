# testing the installation of pytorch
import torch
x = torch.rand(5, 3)
print(x)

# testing PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# config = (  # 1. Configure the algorithm,
#     PPOConfig()
#     .environment("Taxi-v3")
#     .rollouts(num_rollout_workers=2)
#     .framework("torch")
#     .training(model={"fcnet_hiddens": [64, 64]})
#     .evaluation(evaluation_num_workers=1)
# )

# algo = config.build()  # 2. build the algorithm,

# for _ in range(5):
#     print(algo.train())  # 3. train it,

# algo.evaluate()  # 4. and evaluate it.

import gym 
# import gym_box2D
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# import gym
# env = gym.make("CarRacing-v2")
# env.reset()


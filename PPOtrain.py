import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from two_body_gym import BaseSpacecraftEnv
from gymnasium.wrappers import EnvCompatibility
from gymnasium.envs.registration import register
import gymnasium as gym  # Use gymnasium if you're following the most recent updates, or use 'import gym' if not
from shimmy import GymV21CompatibilityV0

# Ensure the environment is accessible by adding its directory to the path 
import sys
sys.path.append('/Users/carloschavez/Documents/Academic & Professional Development/Academia /\
                MS Aerospace CU Boulder/ASEN 5264 Decision Making Uncertainty/DMU Project/AstroPPO')  
# Register environment

register(
    id='BaseSpacecraftEnv-v0',
    entry_point='two_body_gym:BaseSpacecraftEnv',
)
env = gym.make('BaseSpacecraftEnv-v0')
env = GymV21CompatibilityV0(env)


# 1. Configure the PPO algorithm using PPOConfig

config = (
PPOConfig() 
    .environment(env="BaseSpacecraftEnv-v0") 
    .rollouts(num_rollout_workers=1) 
    .framework("torch") 
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(
        evaluation_interval=10,
        evaluation_num_episodes=10,
        evaluation_num_workers=1
    )
)

# 2. Build the PPO trainer with the defined configuration
trainer = config.build()  

# 3. Train the model in a loop
for i in range(100):
    result = trainer.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}") 

# 4. Evaluate it
eval_results = trainer.evaluate() 

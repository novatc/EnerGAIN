import numpy as np
from gymnasium import register, make
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("agents/ppo_energy")

register(
    id='energy_test-v1',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/test_set.csv"}
)

eval_env = make('energy_test-v1')
obs, _ = eval_env.reset()
# Evaluate the agent
episode_reward = 0
for _ in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    print("Action:", action, "Reward:", reward)


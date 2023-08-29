import argparse

import numpy as np
import pandas as pd
from gymnasium import register, make
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
import warnings
warnings.filterwarnings("ignore")
env_params = {
    'unscaled': {'id': 'unscaled_env-v0', 'entry_point': 'envs.unscaled_env:UnscaledEnv',
                 'data_path': 'data/in-use/unscaled_eval_data.csv'}
}

# Set chosen environment parameters
env_id = env_params["unscaled"]['id']
entry_point = env_params["unscaled"]['entry_point']
data_path = env_params["unscaled"]['data_path']

# Load the model
try:
    # find the model that name starts with sac_{args.env}
    model_name = "sac_unscaled_20230829-134908.zip"
    print(f"Loading model {model_name}")
    model = SAC.load(f"agents/{model_name}")
except Exception as e:
    print("Error loading model: ", e)
    exit()

register(id=env_id, entry_point=entry_point, kwargs={'data_path': data_path})
try:
    eval_env = make(env_id)
    action_low = np.array([-1.0, -100.0])
    action_high = np.array([1.0, 100.0])
    env = CustomNormalizeObservation(eval_env)
    env = RescaleAction(env, action_low, action_high)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

ep_length = env.dataframe.shape[0]

episode_rewards = []
num_episodes = 1
obs, _ = env.reset()
for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

env.render()
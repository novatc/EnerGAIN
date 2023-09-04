import numpy as np
from gymnasium import register, make
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
import warnings
warnings.filterwarnings("ignore")
env_params = {
    'unscaled': {'id': 'base-v0', 'entry_point': 'envs.trend_env:TrendEnv',
                 'data_path': 'data/in-use/unscaled_eval_data.csv'}
}

# Set chosen environment parameters
env_id = env_params["unscaled"]['id']
entry_point = env_params["unscaled"]['entry_point']
data_path = env_params["unscaled"]['data_path']

# Load the model
try:
    # find the model that name starts with sac_{args.env}
    model_name = "sac_trend_20230904-143804.zip"
    print(f"Loading model {model_name}")
    model = SAC.load(f"agents/{model_name}")
except Exception as e:
    print("Error loading model: ", e)
    exit()

register(id=env_id, entry_point=entry_point, kwargs={'data_path': data_path, 'validation': True})
try:
    eval_env = make(env_id)
    env = CustomNormalizeObservation(eval_env)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

ep_length = 120


episode_rewards = []
num_episodes = 1
obs, _ = env.reset()
for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

env.render()
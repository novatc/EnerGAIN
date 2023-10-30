import numpy as np
import pandas as pd
from gymnasium import register, make
from gymnasium.wrappers import RescaleAction, NormalizeReward
from stable_baselines3 import SAC
from envs.assets import env_utilities as utilities

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
import warnings

warnings.filterwarnings("ignore")

env_params = {
    'multi': {'id': 'multi-v0', 'entry_point': 'envs.multi_market:MultiMarket',
              'data_path_prl': 'data/prm/preprocessed_prl.csv',
              'data_path_da': 'data/in-use/unscaled_train_data.csv'},
}

# Set chosen environment parameters
env_id = env_params["multi"]['id']
entry_point = env_params["multi"]['entry_point']
data_path_prl = env_params["multi"]['data_path_prl']
data_path_da = env_params["multi"]['data_path_da']
try:
    # find the model that name starts with sac_{args.env}
    model_name = [name for name in utilities.get_model_names() if name.startswith(f"multi")][0]
    print(f"Loading model {model_name}")
    model = SAC.load(f"agents/{model_name}")
except Exception as e:
    print("Error loading model: ", e)
    exit()
register(id=env_id, entry_point=entry_point,
         kwargs={'da_data_path': data_path_da, 'prl_data_path': data_path_prl, 'validation': True})
try:
    eval_env = make(env_id)
    eval_env = CustomNormalizeObservation(eval_env)
except Exception as e:
    print("Error creating environment: ", e)
    exit()

ep_length = 24 * 30

# Evaluate the agent
episode_rewards = []
num_episodes = 1
obs, _ = eval_env.reset()
for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(info)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
trades = eval_env.get_trades()
# list of tuples (step, price, amount, trade_type) to dataframe
trades_log = pd.DataFrame(trades, columns=["step", "price", "amount", "trade_type", "reward"])
trades_log.to_csv(f"trade_logs/{model_name}_trades.csv", index=False)

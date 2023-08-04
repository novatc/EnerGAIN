import argparse
import pandas as pd
from gymnasium import register, make
from stable_baselines3 import SAC
from envs.assets import env_utilities as utilities
import warnings
# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a SAC model.')
parser.add_argument('--env', choices=['base', 'trend', 'savings'], required=True, help='Environment to use.')
parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run.')
args = parser.parse_args()

# Define environment parameters
env_params = {
    'base': {'id': 'base_env-v0', 'entry_point': 'envs.base_env:BaseEnergyEnv', 'data_path': 'data/in-use/eval_data.csv'},
    'trend': {'id': 'trend_env-v0', 'entry_point': 'envs.trend_env:TrendEnv', 'data_path': 'data/in-use/eval_data.csv'},
    'savings': {'id': 'savings_env-v0', 'entry_point': 'envs.savings_env:SavingsEnv', 'data_path': 'data/in-use/eval_data.csv'}
}

# Check if chosen environment is valid
if args.env not in env_params:
    raise ValueError(f"Invalid environment '{args.env}'. Choices are 'base', 'trend', and 'savings'.")

# Set chosen environment parameters
env_id = env_params[args.env]['id']
entry_point = env_params[args.env]['entry_point']
data_path = env_params[args.env]['data_path']

# suppress any warnings
warnings.filterwarnings("ignore")


# Load the model
try:
    # find the model that name starts with sac_{args.env}
    model_name = [name for name in utilities.get_model_names() if name.startswith(f"sac_{args.env}_")][0]
    print(f"Loading model {model_name}")
    model = SAC.load(f"agents/{model_name}")
except Exception as e:
    print("Error loading model: ", e)
    exit()

# Register and make the environment
register(id=env_id, entry_point=entry_point, kwargs={'data_path': data_path})
try:
    eval_env = make(env_id)
except Exception as e:
    print("Error creating environment: ", e)
    exit()


obs, _ = eval_env.reset()

ep_length = eval_env.dataframe.shape[0]

# Evaluate the agent
episode_rewards = []
num_episodes = args.episodes

for _ in range(num_episodes):
    episode_reward = 0
    done = False
    for _ in range(ep_length - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.validation_step(action)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()
# list of tuples (step, price, amount, trade_type) to dataframe
df_trades = pd.DataFrame(trades, columns=['step', 'price', 'amount', 'trade_type'])
df_trades.to_csv("trades.csv", index=False)

# count how many times a buy or sell action was taken
buy_count = 0
sell_count = 0
for trade in trades:
    if trade[2] > 0:
        buy_count += 1
    elif trade[2] < 0:
        sell_count += 1

avg_price = sum([trade[1] for trade in trades]) / len(trades)
rescaled_avg_price = utilities.rescale_value_price(avg_price)
avg_amount = sum([trade[2] for trade in trades]) / len(trades)
rescaled_amount = utilities.rescale_value_amount(avg_amount)

print("Total reward:", sum(episode_rewards))
print("Total trades:", len(trades))
print("Buy count:", buy_count)
print("Sell count:", sell_count)
print("Average reward:", sum(episode_rewards) / len(trades))
print("Average price:", rescaled_avg_price)
print("Average amount:", rescaled_amount)

eval_env.render()

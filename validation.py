import joblib
import numpy as np
from gymnasium import register, make
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC

try:
    model = SAC.load("agents/ppo_energy_testing.zip")
except Exception as e:
    print("Error loading model: ", e)
    exit()

register(
    id='energy-validation-v0',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/test_set.csv"}
)

try:
    eval_env = make('energy-validation-v0')
except Exception as e:
    print("Error creating environment: ", e)
    exit()

obs, _ = eval_env.reset()

# Evaluate the agent
episode_rewards = []
num_episodes = 15

for _ in range(num_episodes):
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        print("Action:", action, "Reward:", reward, "Done:", done, "Truncated:", truncated, "Info:", info)
        episode_reward += reward
    episode_rewards.append(episode_reward)
    obs, _ = eval_env.reset()

trades = eval_env.get_trades()
print("Trades:", trades)

print("Total reward:", sum(episode_rewards))
print("Total trades:", len(trades))
print("Average reward:", sum(episode_rewards) / len(trades))
print("Average price:", sum([trade[1] for trade in trades]) / len(trades))
print("Average amount:", sum([trade[2] for trade in trades]) / len(trades))
print("Average reward per trade:", sum(episode_rewards) / len(trades))
print("Average reward per time step:", sum(episode_rewards) / (24 * num_episodes))
print("Average reward per time step per trade:", sum(episode_rewards) / (24 * num_episodes) / len(trades))

# Separate the trades into buys and sells
buy_trades = [trade for trade in trades if trade[4] == 'buy']
sell_trades = [trade for trade in trades if trade[4] == 'sell']

# Create the bar plot
plt.bar([trade[0] for trade in buy_trades], [trade[3] for trade in buy_trades], color='g', label='Buy')
plt.bar([trade[0] for trade in sell_trades], [trade[3] for trade in sell_trades], color='r', label='Sell')
# add the total reward to the plot as a text
# Add labels and title
plt.xlabel('Time step')
plt.ylabel('Amount')
plt.title('Trades Over Time with Total Reward: ' + str(sum(episode_rewards)))
plt.legend()

# Show the plot
plt.show()

savings_log = eval_env.get_savings()
charge_log = eval_env.get_charge()

full_savings_log = [savings_log[0]] * 5040
full_charge_log = [charge_log[0]] * 5040

# Fill the full logs with the logged values
for i in range(1, len(savings_log)):
    full_savings_log[i] = savings_log[i]
for i in range(1, len(charge_log)):
    full_charge_log[i] = charge_log[i]

# Plot the savings over time
plt.figure(figsize=(10, 5))
plt.plot(full_savings_log, label='Savings')
plt.xlabel('Time step')
plt.ylabel('Savings')
plt.title('Savings Over Time')
plt.legend()
plt.show()

# Plot the battery charge over time
plt.figure(figsize=(10, 5))
plt.plot(full_charge_log, label='Battery Charge')
plt.xlabel('Time step')
plt.ylabel('Battery Charge')
plt.title('Battery Charge Over Time')
plt.legend()
plt.show()

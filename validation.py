from gymnasium import register, make
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

model = PPO.load("agents/ppo_energy_testing_2_mio")

register(
    id='energy-validation-v0',
    entry_point='environment:EnergyEnv',
    max_episode_steps=37272,
    kwargs={'data_path': "data/clean/test_set.csv"}
)

eval_env = make('energy-validation-v0')
obs, _ = eval_env.reset()
# Evaluate the agent
episode_reward = 0
for _ in range(24):  # 168 steps = 1 week
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = eval_env.step(action)
trades = eval_env.get_trades()

print("Total reward:", sum(eval_env.rewards))
print("Total trades:", len(trades))
print("Average reward:", sum(eval_env.rewards) / len(trades))
print("Average price:", sum([trade[1] for trade in trades]) / len(trades))
print("Average amount:", sum([trade[2] for trade in trades]) / len(trades))
print("Average reward per trade:", sum(eval_env.rewards) / len(trades))
print("Average reward per time step:", sum(eval_env.rewards) / 168)
print("Average reward per time step per trade:", sum(eval_env.rewards) / 168 / len(trades))

print(trades)
# plot the trades with the first entry as x-axis and the third entry as y-axis but keep the x axis like 0, 1, 2, ...
plt.plot([trade[0] for trade in trades], [trade[2] for trade in trades], 'ro')
plt.xlabel("Time step")
plt.ylabel("Amount")
plt.title("Trades")

plt.show()


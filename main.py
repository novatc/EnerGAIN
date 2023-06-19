import gym
from gym.utils.env_checker import check_env
from stable_baselines3 import PPO
from gym import register


# Register the environment
register(
    id='EnergyMarketEnv-v0',
    entry_point='enviroment:EnergyMarketEnv',
    max_episode_steps=37272,
)

energy = gym.make('EnergyMarketEnv-v0')
check_env(energy, warn=True)

print("Observation Space: ", energy.observation_space)
print("Action Space: ", energy.action_space)

# model = PPO("MlpPolicy", energy, verbose=0)
# model.learn(total_timesteps=100_000)

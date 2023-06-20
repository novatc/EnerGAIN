from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from stable_baselines3 import PPO

from energy_RL.env import EnergyEnv

register(
    id='energy-v0',
    entry_point='energy_RL.env:EnergyEnv',
    max_episode_steps=1000,
)

env = make('energy-v0')
check_env(env)

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=500_000)
model.save("ppo_energy")
env.render()





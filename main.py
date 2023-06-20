from stable_baselines3.common.env_checker import check_env
from gymnasium import register
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

register(
    id='energy-v0',
    entry_point='environment:EnergyEnv',
    max_episode_steps=27272,
)

env = make('energy-v0')
env = Monitor(env, filename="logging/", allow_early_resets=True)
check_env(env)

model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="logging/")
model.learn(total_timesteps=500_000)
model.save("ppo_energy")
env.render()

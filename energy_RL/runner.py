from stable_baselines3.common.env_checker import check_env

from energy_RL.env import EnergyEnv

env = EnergyEnv()
check_env(env, warn=True)

obs = env.reset()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

n_steps = 20
for step in range(n_steps):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())

env.render()




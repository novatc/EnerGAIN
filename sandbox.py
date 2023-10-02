import numpy as np
from gymnasium import register, make
from gymnasium.wrappers import RescaleAction, NormalizeReward
from stable_baselines3 import SAC

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation
import warnings

warnings.filterwarnings("ignore")

env_params = {
    'base_prl': {'id': 'base_prl-v0', 'entry_point': 'envs.base_prl:BasePRL',
                 'data_path_prl': 'data/prm/preprocessed_prl.csv',
                 'data_path_da': 'data/in-use/unscaled_train_data.csv'},
}

# Set chosen environment parameters
env_id = env_params["base_prl"]['id']
entry_point = env_params["base_prl"]['entry_point']
data_path_prl = env_params["base_prl"]['data_path_prl']
data_path_da = env_params["base_prl"]['data_path_da']

register(id=env_id, entry_point=entry_point,
         kwargs={'da_data_path': data_path_da, 'prl_data_path': data_path_prl, 'validation': False})

eval_env = make(env_id)
env = CustomNormalizeObservation(eval_env)
env = NormalizeReward(env)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)


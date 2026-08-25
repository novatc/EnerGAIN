"""
Train several env variants across several seeds and compare them on the evaluation set.

Training in this repository is not reproducible by default: the market simulators draw their
random-walk start from the global random module, so reset(seed=...) does not control the
trajectory. A single before/after run therefore says nothing. This script seeds every RNG that
matters and repeats each variant, reporting mean and standard deviation so a difference can be
read against the spread.

Run from the repository root, for example:

    python benchmark.py --envs base base_state --training_steps 20000 --seeds 0 1 2 --month 0
"""
import argparse
import random
import statistics
import time
import warnings

import numpy as np
from gymnasium import make, register
from gymnasium.wrappers import NormalizeReward
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise

from cutsom_wrappers.custom_wrappers import CustomNormalizeObservation

warnings.filterwarnings("ignore")

TRAIN_DA = 'data/in-use/unscaled_train_data.csv'
TRAIN_DA_EXT = 'data/in-use/unscaled_train_data_ext.csv'
TRAIN_PRL = 'data/prm/preprocessed_prl.csv'
TRAIN_PRL_EXT = 'data/prm/preprocessed_prl_ext.csv'

# env key -> (entry point, training da path, training prl path or None, uses extended eval data)
VARIANTS = {
    'base':              ('envs.base_env:BaseEnv',           TRAIN_DA,     None,          False),
    'no_savings':        ('envs.no_savings_env:NoSavingsEnv', TRAIN_DA,    None,          False),
    'trend':             ('envs.trend_env:TrendEnv',         TRAIN_DA,     None,          False),
    'base_prl':          ('envs.base_prl:BasePRL',           TRAIN_DA,     TRAIN_PRL,     False),
    'multi':             ('envs.multi_market:MultiMarket',   TRAIN_DA,     TRAIN_PRL,     False),
    'multi_no_savings':  ('envs.multi_no_savings:MultiNoSavings', TRAIN_DA, TRAIN_PRL,    False),
    'multi_trend':       ('envs.multi_trend:MultiTrend',     TRAIN_DA,     TRAIN_PRL,     False),
    'base_state':        ('envs.base_state:BaseState',       TRAIN_DA,     None,          False),
    'trend_compact':     ('envs.trend_compact:TrendCompact', TRAIN_DA,     None,          False),
    'multi_compact':     ('envs.multi_compact:MultiCompact', TRAIN_DA,     TRAIN_PRL,     False),
    'base_state_ext':    ('envs.base_state:BaseState',       TRAIN_DA_EXT, None,          True),
    'trend_compact_ext': ('envs.trend_compact:TrendCompact', TRAIN_DA_EXT, None,          True),
    'multi_compact_ext': ('envs.multi_compact:MultiCompact', TRAIN_DA_EXT, TRAIN_PRL_EXT, True),
    # Ablation: multi_compact with the boundary penalty switched off, to test whether charging
    # for a clipped request actually helps or whether clipping alone is enough.
    'multi_compact_nopen': ('envs.multi_compact:MultiCompact', TRAIN_DA, TRAIN_PRL, False),
}

# Extra constructor kwargs per variant.
EXTRA_KWARGS = {
    'multi_compact_nopen': {'boundary_penalty': False},
}


def eval_paths(month: int, extended: bool):
    """
    Resolve the evaluation CSVs for a month.

    :param month: 1-12 for a single month, 0 for the synthetic average year.
    :param extended: whether to use the *_ext sets with the solar and PRL time columns.
    :return: (day-ahead path, PRL path).
    """
    tag = '_ext' if extended else ''
    if month == 0:
        return (f'data/in-use/eval_data/average_da_year{tag}.csv',
                f'data/in-use/eval_data/average_prl_year{tag}.csv')
    da_tag = '_ext_da' if extended else '_data_da'
    prl_tag = '_ext_prl' if extended else '_data_prl'
    prefix = f'data/in-use/eval_data/month_{month}'
    return (f'{prefix}{da_tag}.csv' if extended else f'{prefix}_data_da.csv',
            f'{prefix}{prl_tag}.csv' if extended else f'{prefix}_data_prl.csv')


def build(env_key: str, run_id: str, da_path: str, prl_path, validation: bool):
    """
    Register and construct one env instance under a unique id.

    :param env_key: the variant key.
    :param run_id: a suffix making the gym id unique for this run.
    :param da_path: day-ahead CSV.
    :param prl_path: PRL CSV, or None for day-ahead-only variants.
    :param validation: whether to step sequentially instead of random-walking.
    :return: the constructed env.
    """
    entry_point = VARIANTS[env_key][0]
    kwargs = {'da_data_path': da_path, 'validation': validation}
    if prl_path is not None:
        kwargs['prl_data_path'] = prl_path
    kwargs.update(EXTRA_KWARGS.get(env_key, {}))
    env_id = f'bench_{env_key}_{run_id}-v0'
    register(id=env_id, entry_point=entry_point, kwargs=kwargs)
    return make(env_id)


def run_once(env_key: str, seed: int, training_steps: int, month: int,
             tensorboard: bool = False) -> dict:
    """
    Train one seed of one variant and evaluate it.

    :param env_key: the variant key.
    :param seed: the seed applied to the stdlib, numpy and SAC RNGs.
    :param training_steps: SAC training timesteps.
    :param month: evaluation month, 0 for the average year.
    :param tensorboard: write TensorBoard logs. Off by default because setting tensorboard_log
                        makes SAC import tensorboard, which benchmark.py otherwise does not need.
    :return: a dict of evaluation metrics.
    """
    _, da_train, prl_train, extended = VARIANTS[env_key]

    random.seed(seed)
    np.random.seed(seed)

    train_env = NormalizeReward(CustomNormalizeObservation(
        build(env_key, f'train_{seed}', da_train, prl_train, False)))
    n_actions = train_env.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    log_dir = f'logging/tensorboard_logs/bench_{env_key}/' if tensorboard else None
    model = SAC("MlpPolicy", train_env, verbose=0, device='auto', action_noise=noise, seed=seed,
                tensorboard_log=log_dir)
    model.learn(total_timesteps=training_steps, progress_bar=False,
                tb_log_name=f'seed_{seed}')

    da_eval, prl_eval = eval_paths(month, extended)
    eval_env = CustomNormalizeObservation(
        build(env_key, f'eval_{seed}', da_eval, prl_eval if prl_train else None, True))
    inner = eval_env.unwrapped

    obs, _ = eval_env.reset()
    for _ in range(inner.da_dataframe.shape[0] - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = eval_env.step(action)

    trades = inner.get_trades()
    invalid = inner.get_invalid_trades()
    battery_rejects = sum(1 for t in invalid if t[6] == 'battery')
    # 'boundary' entries only exist in MultiCompact, which records a refusal MultiMarket
    # silently swallows. Reported separately so the invalid column stays comparable.
    boundary = sum(1 for t in invalid if t[6] == 'boundary')
    return {
        'profit': trades[-1][8] if trades else 0.0,
        'trades': len(trades),
        'invalid': len(invalid) - boundary,
        'boundary': boundary,
        'battery_rejects': battery_rejects,
        'holds': len(inner.get_holdings()),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare env variants across seeds.')
    parser.add_argument('--envs', nargs='+', required=True, choices=sorted(VARIANTS),
                        help='Variants to compare.')
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--month', type=int, default=0)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Write TensorBoard logs to logging/tensorboard_logs/bench_<env>/. '
                             'Requires tensorboard to be installed.')
    args = parser.parse_args()

    print(f"steps={args.training_steps}  seeds={args.seeds}  month={args.month}", flush=True)
    if args.tensorboard:
        print("tensorboard: logging/tensorboard_logs/bench_<env>/", flush=True)
    total = len(args.envs) * len(args.seeds)
    print(f"{total} runs to do\n", flush=True)

    results = {}
    done = 0
    for env_key in args.envs:
        runs = []
        started = time.time()
        for seed in args.seeds:
            run_start = time.time()
            runs.append(run_once(env_key, seed, args.training_steps, args.month, args.tensorboard))
            done += 1
            # Print as each run lands. A long sweep would otherwise show nothing until every
            # seed of a variant had finished.
            print(f"  [{done:>2}/{total}] {env_key:<20} seed {seed}  "
                  f"Kapital {runs[-1]['profit']:>10.2f}  "
                  f"invalid {runs[-1]['invalid']:>5}  battery {runs[-1]['battery_rejects']:>5}  "
                  f"bound {runs[-1]['boundary']:>5}  "
                  f"({time.time() - run_start:.0f}s)", flush=True)
        results[env_key] = (runs, time.time() - started)

    header = (f"{'env':<20}{'Kapital mean':>14}{'std':>10}{'trades':>9}{'invalid':>9}"
              f"{'battery':>9}{'bound':>8}{'holds':>8}")
    print(f"\n{header}")
    print('-' * len(header))
    for env_key, (runs, elapsed) in results.items():
        profits = [r['profit'] for r in runs]
        mean = statistics.mean(profits)
        std = statistics.stdev(profits) if len(profits) > 1 else 0.0
        avg = lambda k: statistics.mean(r[k] for r in runs)
        print(f"{env_key:<20}{mean:>14.2f}{std:>10.2f}{avg('trades'):>9.0f}"
              f"{avg('invalid'):>9.0f}{avg('battery_rejects'):>9.0f}{avg('boundary'):>8.0f}"
              f"{avg('holds'):>8.0f}   [{elapsed:.0f}s]")


if __name__ == '__main__':
    main()

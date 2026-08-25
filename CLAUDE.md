# CLAUDE.md

Guidance for AI assistants working in the **EnerGAIN** repository.

---

## Quick start (read this first)

This is a 2023 research codebase (last substantive commit 2023-12-31). Six things break a
first session; all six are verified, with the exact errors, in §3–§4.

```bash
# 1. Deps: requirements.txt is incomplete. All four extras are MANDATORY, not optional.
pip install -r requirements.txt scipy tensorboard tqdm rich

# 2. gymnasium 1.x BREAKS validation.py. Pin 0.29.x. (sb3 2.4.1 works; the README's
#    stable-baselines3==2.0.0a13 pin is NOT required.)
pip install "gymnasium==0.29.1"

# 3. Always run top-level scripts from the repo root (no __init__.py; namespace packages).
# 4. Create the output dirs — neither script does it, and validation.py only fails AFTER
#    running the whole evaluation loop.
mkdir -p trade_logs/invalid agent_data logging
# 5. Copy the model you want UP into agents/ — the lookup is non-recursive.
cp agents/results/base/sac_base_500.0k_22.12-13-49.zip agents/
# 6. For --plot, also create agent_data/<ENV NAME>/ (e.g. base, not the zip filename).
mkdir -p agent_data/base

python main.py --training_steps 500_000 --env base --save
MPLBACKEND=Agg python validation.py --env base --month 0 --plot

# 7. Only for the *_ext variants (§13): build their data once. Deterministic, gitignored.
python preprocess_ext.py
```

**Do not** trust `agents/results/base/sac_no_savings_500.0k_09.12-21-09.zip` — it is **0 bytes**
and fails with `Error: the file ... wasn't a zip-file`. There is no usable `no_savings` model
in the repo.

**Training is not reproducible** even with a seed — see §6. **The README's results table does
not reproduce exactly** — see §8.

Prefer minimal, surgical changes. Do not restructure directories, rename modules, or
"modernize" this code unless explicitly asked.

---

## 1. What this project is

Code for a German-language thesis: *"Entwicklung einer intelligenten Gebotsstrategie für den
Strommarkt mittels Reinforcement Learning"* — learning bidding strategies for a grid-scale
battery on two German electricity markets:

- **Day-Ahead (DA)** — hourly buy/sell of energy (`price` €/kWh, `amount` kWh).
- **Primärregelleistung / PRL** (Frequency Containment Reserve) — pay-as-bid capacity market
  sold in **4-hour blocks** (`price` €/kWh, `amount` kW).

A single **SAC** agent (Stable-Baselines3) acts in a custom Gymnasium env wrapping a battery
plus one or both markets. No service, no API, no tests, no CI, no linter config. The workflow
is: preprocess CSVs → train → validate → emit SVG plots for the thesis.

`diagram/step_flow.drawio` is a flowchart of the multi-market `step()` and is an accurate
primary source for §5.

## 2. Repository layout

```
main.py                     Training entry point (SAC)
validation.py               Evaluation entry point (loads a model, writes logs + stats + plots)
plotting.py                 Cross-agent comparison plots; RUNS AT IMPORT (no main guard)
preprocess_data.py          data/clean/* -> data/in-use/{unscaled_train_data,env_data}.csv
preprocess_prl.py           data/prm/prl.csv -> data/prm/{env_prl,preprocessed_prl}.csv
create_da_eval_set.py       data/in-use/env_data.csv -> data/in-use/eval_data/*_da.csv
create_prl_eval_set.py      data/prm/env_prl.csv     -> data/in-use/eval_data/*_prl.csv
preprocess_ext.py           Builds the *_ext datasets: DA + solar columns, PRL + time features
benchmark.py                Seeded multi-run comparison of env variants (§13)

envs/                       Gymnasium envs — one file per experiment variant
  base_env.py               BaseEnv         — DA only
  no_savings_env.py         NoSavingsEnv    — DA only, "no savings constraint" (see §7)
  trend_env.py              TrendEnv        — DA only + 8-hour lookback
  base_prl.py               BasePRL         — DA + PRL, explicit market-choice action
  multi_market.py           MultiMarket     — DA + PRL in parallel each step
  multi_no_savings.py       MultiNoSavings  — MultiMarket without the savings check
  multi_trend.py            MultiTrend      — MultiMarket + 8-hour lookback
  base_state.py             BaseState       — BaseEnv + own state in obs + trade clipping (§13)
  trend_compact.py          TrendCompact    — BaseState + compact price-history features (§13)
  multi_compact.py          MultiCompact    — MultiMarket + those features + own state (§13)
  assets/
    features.py                       Price-history features + trade clipping helpers
    battery.py                        Battery (capacity, SOC, charge/discharge efficiency)
    dayahead.py                       DA market sim (stepping, offer acceptance)
    frequency_containment_reserve.py  PRL/FCR market sim
    plot_engien.py                    Per-agent plots called from env.render()  [sic]
    env_utilities.py                  get_model_names(), moving_average()

cutsom_wrappers/            [sic] CustomNormalizeObservation
callbacks/summary_writer.py SummaryWriterCallback — imported by main.py but NEVER USED
dummy_bots/                 Non-RL baselines (threshold, moving average); run as scripts
data_preprocess/            One-off scripts: data/original/* -> data/clean/*
vis/                        One-off dataset plots for the thesis
plots/                      Committed SVG output from vis/
diagram/step_flow.drawio    Flowchart of the multi-market step()
agents/results/{base,multi} Committed trained models (.zip) — NOT where the loader looks
*.job                       SLURM batch scripts
```

**Two intentional typos in module paths** — `cutsom_wrappers` (not `custom_`) and
`envs/assets/plot_engien.py` (not `plot_engine`). Imports depend on them. Match them exactly;
do not rename unless asked.

There are **no `__init__.py` files**; these work as namespace packages, which is why every
top-level script must be run from the repo root (or with `PYTHONPATH=<repo root>`).

## 3. Dependencies — verified

`requirements.txt` pins `numpy pandas torch stable-baselines3 matplotlib scikit-learn
gymnasium seaborn`. Four more are **required for the documented workflow**, each confirmed by
running it:

| package | why it is mandatory | error without it |
|---|---|---|
| `scipy` | `plot_engien.py` imports `gaussian_kde` | `ModuleNotFoundError` on any `--plot` |
| `tensorboard` | `main.py` **always** passes `tensorboard_log=...` to SAC | `ImportError: Trying to log data to tensorboard but tensorboard is not installed.` |
| `tqdm`, `rich` | `main.py` hardcodes `progress_bar=True` | `ImportError: You must install tqdm and rich in order to use the progress bar callback.` |

`openpyxl` is needed only by `data_preprocess/solar_power.py` (`to_excel`).

### The gymnasium version is load-bearing

| | gymnasium 0.29.1 | gymnasium 1.0.0 |
|---|---|---|
| `main.py` (training) | works | works |
| `validation.py` | works | **fails immediately** |

On 1.x: `AttributeError: 'CustomNormalizeObservation' object has no attribute 'da_dataframe'`.
Gymnasium 1.0 removed `Wrapper.__getattr__` forwarding, and `validation.py` reaches through
the wrapper for `da_dataframe`, `get_trades()`, `get_invalid_trades()`, `get_holdings()`,
`trade_log`, and `render()`. **Verified working combination: `gymnasium==0.29.1` +
`stable-baselines3==2.4.1`.** The README's `stable-baselines3==2.0.0a13` is not required.

If asked to support gymnasium 1.x, the fix is in `cutsom_wrappers/custom_wrappers.py` (forward
attributes, or have `validation.py` use `env.unwrapped`) — not in each env.

## 4. Running things

### Train

```bash
python main.py --training_steps 500_000 --env base --save
```

`--env`: `base | trend | no_savings | base_prl | multi | multi_no_savings | multi_trend`.
Without `--save` the model is trained and discarded.

Reads `data/in-use/unscaled_train_data.csv` (+ `data/prm/preprocessed_prl.csv` for PRL envs),
registers with `validation=False`, wraps in `CustomNormalizeObservation` + `NormalizeReward`,
trains `SAC("MlpPolicy", ...)` with `NormalActionNoise(sigma=0.1)`. Writes
`agents/sac_{env}_{steps/1000}k_{DD.MM-HH-MM}.zip` and TensorBoard logs to
`logging/tensorboard_logs/{env}/`. `main.py` creates `logging/` itself.

### Evaluate

```bash
MPLBACKEND=Agg python validation.py --env base --month 0 --plot
```

`--month`: `1`–`12` selects `data/in-use/eval_data/month_{n}_data_{da,prl}.csv`; `0` uses the
synthetic average year. `--episodes` defaults to `1`. Set `MPLBACKEND=Agg` when headless —
`plot_engien.py` calls `plt.show()` on every figure.

Four failure modes, all verified:

1. **No model in `agents/`** → `Error loading model: list index out of range`.
   `env_utilities.get_model_names()` scans `agents/` **non-recursively** and takes the *first*
   file starting with `sac_{env}_`. The committed models live one level down in
   `agents/results/{base,multi}/`, so copy the one you want up.
   **Stage exactly one model at a time.** The match is a bare prefix, so `sac_base_prl_*.zip`
   also matches `--env base`, and `os.listdir` order decides which wins — `--env base` can
   silently load the `base_prl` model (5-dim action space against a 2-dim env).
2. **The 0-byte `no_savings` model** → `Error: the file agents/sac_no_savings_500.0k_09.12-21-09.zip wasn't a zip-file`.
3. **Missing output dirs** → `OSError: Cannot save file into a non-existent directory: 'trade_logs'`.
   This fires *after* the full evaluation loop, so you lose the run. `mkdir -p trade_logs/invalid agent_data`.
4. **`--plot` without the plot dir** → `FileNotFoundError: agent_data/base/base_savings.svg`.
   Only `plot_reward()` calls `os.makedirs`, and **no env's `render()` calls `plot_reward`**.
   Create `agent_data/<env name>/` — the *env* name hardcoded in `render()` (`base`, `multi`,
   …), not the zip filename used for the CSV/JSON outputs.

Outputs: `trade_logs/{model}_trades.csv`, `trade_logs/invalid/{model}_invalid_trades.csv`,
`agent_data/{model}_stats.json`, and with `--plot`, 7–8 SVGs in `agent_data/{env}/`.

### Compare agents / regenerate figures

`plotting.py` has **no `if __name__ == "__main__"` guard** — it loads every CSV in
`trade_logs/` and plots on import. Run it only after `validation.py` has produced those CSVs,
and create `img/` first (`plotting.py` writes there without creating it).

Scripts in `vis/`, `data_preprocess/`, and `dummy_bots/` use paths relative to *their own*
directory (`../data/...`), so run them from inside that directory:

```bash
cd vis && MPLBACKEND=Agg python year_price_plotter.py
```

### Cluster

`*.job` are SLURM scripts (`sbatch base.job`); some use `--array=1-3` to sweep step counts and
all hardcode a `--mail-user`. None request a GPU; training runs on CPU by default
(`device='auto'`).

## 5. Environment design

### Spaces (dims confirmed by instantiating every env)

| env | class | markets | obs dim | action dim | action vector |
|---|---|---|---|---|---|
| `base` | `BaseEnv` | DA | 9 | 2 | `[da_price, da_amount]` |
| `no_savings` | `NoSavingsEnv` | DA | 9 | 2 | `[da_price, da_amount]` |
| `trend` | `TrendEnv` | DA | 72 (9×8h) | 2 | `[da_price, da_amount]` |
| `base_prl` | `BasePRL` | DA+PRL | 14 | 5 | `[prl_choice, prl_price, prl_amount, da_price, da_amount]` |
| `multi` | `MultiMarket` | DA+PRL | 14 | 4 | `[prl_price, prl_amount, da_price, da_amount]` |
| `multi_no_savings` | `MultiNoSavings` | DA+PRL | 14 | 4 | as `multi` |
| `multi_trend` | `MultiTrend` | DA+PRL | 89 ((9+2)×8h+1) | 4 | as `multi` |
| `base_state` | `BaseState` | DA | 11 (9+2) | 2 | as `base` |
| `trend_compact` | `TrendCompact` | DA | 19 (9+8+2) | 2 | as `base` |
| `multi_compact` | `MultiCompact` | DA+PRL | 24 (9+2+8+5) | 4 | as `multi` |
| `base_state_ext` | `BaseState` | DA | 18 (16+2) | 2 | as `base` |
| `trend_compact_ext` | `TrendCompact` | DA | 26 (16+8+2) | 2 | as `base` |
| `multi_compact_ext` | `MultiCompact` | DA+PRL | 37 (16+8+8+5) | 4 | as `multi` |

Dims assume the current CSVs (9 DA columns, 2 PRL columns) and were checked against the
declared `observation_space` — all seven match. Multi-market observations append
`[prl_cooldown, lower_bound, upper_bound]`; `multi_trend` appends only `prl_cooldown`.
Action bounds: DA price `[0, 1]` €/kWh, DA amount `[-1000, 1000]` kWh (negative = sell),
PRL price `[0.001, 0.5]`, PRL amount `[0, 1000]`.

### Shared constants (identical across all envs)

`Battery(capacity=1000, soc=500)`, charge/discharge efficiency `0.925`; `savings = 50` €;
`penalty = -10`; `trade_threshold = 10` (a DA amount in `(-10, 10)` is a **hold**, not a
trade); `trend_horizon = 8` in the trend variants.

Holding reward is `1` everywhere **except `TrendEnv`, which returns `5`** — deliberate, not a
typo. Do not "harmonize" it.

### Step semantics

- **Training** (`validation=False`): `random_walk(24 * 30)` picks a random start, runs 30
  simulated days, then resamples and returns `truncated=True`.
- **Validation** (`validation=True`): plain sequential `step()` through the eval CSV.
- In multi-market envs the **PRL market drives the clock**; DA is force-synced via
  `day_ahead.set_step(prl.get_current_step())`.

### Market acceptance

`DayAhead.accept_offer` accepts a **buy** when `offer_price > market_price` and a **sell** when
`offer_price < market_price` — you must bid *against* yourself to clear, and profit books at
the **market** price, not the offered one. `FrequencyContainmentReserve.accept_offer` accepts
when `offer_price < market_price`; PRL is pay-as-bid, so revenue is `price * amount * 4`.

### PRL constraints

`prl_cooldown` is set to `4` after a PRL trade and decremented per step; bids are only allowed
at `prl.get_current_step() % 4 == 0` with cooldown expired. `set_boundaries(amount_prl)`
narrows the allowed SOC band so reserved capacity stays available, and `clip_trade_amount`
clips DA trades into it.

`multi_market.py` reads oddly: `if self.prl_cooldown <= 0 == self.prl.get_current_step() % 4:`.
That is a Python **chained comparison** — `cooldown <= 0 and 0 == step % 4` — and is correct as
written. Leave it alone unless the behaviour itself is in question.

## 6. Reproducibility — training is NOT deterministic

`reset(seed=...)` forwards to `super().reset()`, which seeds only the Gymnasium RNG. But
`dayahead.py:109` and `frequency_containment_reserve.py:70` call the **global** `random.randint`
to choose the random-walk start. Seeding the env therefore does **not** control the trajectory.
Demonstrated:

```
reset(seed=42) run A -> steps [26874, 26875, 26876, 26877, 26878]
reset(seed=42) run B -> steps [   81,    82,    83,    84,    85]
```

Workaround — seed the stdlib global RNG before constructing the env (verified reproducible):

```python
import random; random.seed(7)
```

`main.py` now takes `--seed`, which seeds the stdlib RNG, numpy and SAC together; `benchmark.py`
does the same per run. Without it nothing is reproducible. Still unseeded: the
`np.random.normal` noise in `preprocess_data.py` / `preprocess_prl.py` (`preprocess_ext.py`
omits that noise entirely, so its output *is* reproducible). Validation was always deterministic
(sequential stepping, `model.predict(deterministic=True)`).

### `reset()` is partial, and is called from inside `step()`

`reset()` restores **only** `savings` and `battery`. It does **not** reset
`day_ahead.current_step`, `trade_log`, `invalid_trades`, `holding`, `reward_log`,
`prl_cooldown`, or the SOC bounds — so logs accumulate across episode boundaries during
training. (`Battery.reset()` used to hardcode `soc = 500`; it now restores the constructor's
`soc`, so changing `Battery(1000, 500)` takes effect.)

Separately, `step()` calls `self.reset()` itself when `random_walk` truncates *and* returns
`truncated=True`, which inverts the Gymnasium contract (the algorithm owns reset) and
interacts with the `NormalizeObservation` / `NormalizeReward` wrappers. This is the most
likely thing to confuse someone modifying an env.

## 7. Known inconsistencies between variants

### Still open — flag before "fixing"

These are real differences in committed code that may be how the published results were
produced. Leave them unless asked.

- `BaseEnv` / `NoSavingsEnv` / `TrendEnv` sell via `battery.charge(negative_amount)` and update
  `savings` directly; the multi-market envs call `battery.discharge()` and do
  `savings += profit`.
- `main.py` builds the env twice (once outside the `try`, once inside) — redundant, harmless.
- `SummaryWriterCallback` is imported by `main.py` but never passed to `model.learn()`.
- `multi_no_savings` (the job file) has no `.job` extension, unlike its siblings.
- The README lists fewer `--env` choices than the code supports, and its install section pins
  an `sb3` alpha that is not required (§3).

### Fixed (2026-08) — history before this point behaves differently

- **`NoSavingsEnv` now actually drops the savings check.** It used to be byte-for-byte
  `base_env.py` apart from the class name and plot labels, so it still rejected buys the agent
  could not afford. `is_trade_valid` now checks battery capacity only, mirroring
  `MultiNoSavings`. **This changes the variant's semantics**: retraining `no_savings` no longer
  reproduces whatever produced the README's "No Savings" row. Revert the `is_trade_valid` hunk
  if you need the old behaviour.
- **`reward_log` is no longer double-counted.** `base_prl`, `multi`, `multi_no_savings` and
  `multi_trend` each appended to `self.reward_log` in `step()` *and* again in `log_step()`, so
  the log held two entries per step. The direct append in `step()` was removed. Nothing reads
  `self.reward_log` or `self.rewards` — they are write-only — so results are unaffected;
  `plot_reward()` derives its own local `reward_log` from `trade_log`.
- **`Battery.reset()` honours the constructor SOC** via a new `self.initial_soc`, instead of
  hardcoding `500`. No behaviour change while every env uses `Battery(1000, 500)`.
- **`--env reward_boosting` is gone** from the `argparse` `choices` of both scripts, along with
  the dead `reward_boosting.job`. It had no `env_params` entry and raised `ValueError`
  immediately. The stale "Invalid environment" messages in both scripts (which listed a
  nonexistent `unscaled` and omitted `multi_trend`) were corrected at the same time.

`base` and `multi` were re-validated against their committed models after these changes and
reproduce their previous numbers exactly.

## 8. Reproducing the README results table

Row identity is unambiguous, but the figures are **indicative, not reproducible**. Mapping
(confirmed by running each model on `--month 0` and matching the trade-count signature):

| README row | `--env` | model |
|---|---|---|
| Base | `base` | `agents/results/base/sac_base_500.0k_22.12-13-49.zip` |
| No Savings | `no_savings` | **0-byte file — unusable** (and see §7: the variant's semantics changed) |
| Trend | `trend` | `agents/results/base/sac_trend_1000.0k_22.12-16-16.zip` |
| Multi-Markt | `base_prl` | `agents/results/multi/sac_base_prl_500.0k_21.12-12-02.zip` |
| parallel Multi Markt | `multi` | `agents/results/multi/sac_multi_1500.0k_25.12-19-47.zip` |
| parallel Multi Markt No Savings | `multi_no_savings` | `agents/results/multi/sac_multi_no_savings_1500.0k_21.12-16-23.zip` |
| parallel Multi Markt Trend | `multi_trend` | `agents/results/multi/sac_multi_trend_1500.0k_21.12-17-44.zip` |

Note "Multi-Markt" is `base_prl` (explicit market choice) and "parallel Multi Markt" is `multi`
(both markets each step) — the naming inverts what you might guess.

### The DA-only rows reproduce exactly; the PRL rows do not

Every usable model was re-run on `--month 0`. Kauf / Verkauf / Reserve / Halte / invalide,
published vs. observed:

| `--env` | README | observed | |
|---|---|---|---|
| `base` | 1179 / 814 / – / 428 / 6362 → €1982,69 | 1179 / 814 / 0 / 428 / 6362 → €1982,69 | **exact** |
| `trend` | 1421 / 1536 / – / 500 / 5826 → €3868 | 1421 / 1536 / 0 / 500 / 5826 → €3868,74 | **exact** |
| `base_prl` | 3 / 3 / 7532 / 10 / 22 → €16 853,94 | 3 / 0 / 7632 / 6 / 13 → €20 202,63 | differs |
| `multi` | 118 / 392 / 3592 / 359 / 533 → €13 863,22 | 144 / 414 / 3964 / 197 / 628 → €12 015,04 | differs |
| `multi_no_savings` | 192 / 461 / 4404 / 207 / 519 → €12 510,99 | 31 / 72 / 8152 / 36 / 106 → €12 334,03 | differs |
| `multi_trend` | 113 / 249 / 4388 / 133 / 521 → €12 149,91 | 195 / 408 / 3880 / 38 / 414 → €14 251,64 | differs |

So the two DA-only rows are reproducible to the cent, and all four PRL rows are not. The
likeliest explanation is that the PRL models were replaced after the table was written (see the
`added final models` / `removed old models` commits) while the DA-only models were not. Quote
the `base` and `trend` rows freely; re-run the PRL rows rather than quoting them.

Note this is measured with the current code. The `no_savings` row cannot be checked at all —
its model is 0 bytes — and that variant's semantics have since changed (§7).

## 9. Data pipeline

```
data/original/    Raw German downloads (SMARD, DWD, PV sim); ';'-separated, German decimal
                  commas, '-' for missing. Listed in .gitignore yet committed (use `git add -f`).
      |  data_preprocess/*.py   (run from inside data_preprocess/)
data/clean/       Per-source cleaned CSVs + dataset_01102018_01012023.csv
      |  preprocess_data.py     (run from repo root)
data/in-use/      unscaled_train_data.csv  <- DA training data
                  env_data.csv             <- dated intermediate (NOT COMMITTED, see below)
      |  create_da_eval_set.py / create_prl_eval_set.py
data/in-use/eval_data/  month_{1..12}_data_{da,prl}.csv, average_{da,prl}_year.csv

data/prm/prl.csv  Raw PRL export (';'-separated, German dates)
      |  preprocess_prl.py
data/prm/         env_prl.csv (dated, NOT COMMITTED) + preprocessed_prl.csv (env input)
```

Column schemas:

- **DA training** (`unscaled_train_data.csv`, 9 cols): `price, consumption, prediction,
  hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, month_sin, month_cos`
- **DA eval** (`eval_data/*_da.csv`, 9 cols): `price, consumption, prediction, month_sin,
  month_cos, day_sin, day_cos, hour_sin, hour_cos` — same count, **different cyclical
  columns** (day-of-month, not day-of-week).
- **PRL** (`preprocessed_prl.csv`, `*_prl.csv`, 2 cols): `price, amount`

Conventions: prices converted to **€/kWh** (raw €/MWh ÷ 1000); time features sine/cosine
encoded; `price` is written as the CSV index by the preprocessors but read back as an ordinary
column (`pd.read_csv` with no `index_col`), which is why `shape[1]` includes it. Envs derive
observation bounds directly from `df.min()` / `df.max()`.

**The pipeline is not runnable end-to-end as committed.** `preprocess_data.py` writes
`data/in-use/env_data.csv` and `preprocess_prl.py` writes `data/prm/env_prl.csv`; neither is in
the repo, so `create_da_eval_set.py`, `create_prl_eval_set.py`, and `vis/year_price_plotter.py`
all fail until you re-run preprocessing. (`data/clean/env_data.csv` is a *scaled* artifact and
is not a drop-in substitute.) The generated `eval_data/` CSVs *are* committed, so training and
validation work without re-running any of this.

Two `data_preprocess/` outputs are dead: `energy_reserve_*.csv` and `solar_values_*.csv` are
produced but never consumed — `preprocess_data.py` reads only consumption, prediction,
trading_prices, and solar_power, and the live PRL path starts from `data/prm/prl.csv` instead.

## 10. Trade log format

Every env logs trades as **9-tuples**. The same order is used by the dummy bots, by
`plot_engien.py` (positional indexing), and by the CSV headers in `validation.py`:

```
(step, type, market_price, offered_price, amount, reward, case, soc, savings)
   0     1         2             3           4       5      6     7      8
```

`type` ∈ `{'buy', 'sell', 'reserve'}`. `case` is `'accepted'` / `'prl accepted'` for valid
trades, or the rejection reason (`'battery'`, `'savings'`, `'market rejected'`) for entries in
`invalid_trades`. Holds go to a separate `self.holding` list of `(step, 'hold')`.

**If you change this tuple, update every consumer**: all seven envs, both `dummy_bots/`, every
positional index in `plot_engien.py` (`trade[3]`, `trade[4]`, `trade[8]`, …), and the
`columns=[...]` lists in `validation.py`.

## 11. Conventions

- **Style**: plain functions/classes, reST-ish docstrings (`:param x:` / `:return:`), 4-space
  indent, ~120-char lines, f-strings. No formatter/linter/type checker is configured — match
  the surrounding file rather than reformatting.
- **Adding an env variant**: copy the closest existing env file wholesale, rename the class,
  change the behaviour, update the `model_name` string in every `render()` call, then register
  it in the `env_params` dict **and** the `choices=[...]` list in *both* `main.py` and
  `validation.py`. Add a `*.job` file if it is for the cluster. Duplication between env files
  is the established pattern — do not refactor the seven envs into a shared base class unless
  explicitly asked.
- **Plots**: SVG at `dpi=1200` into `agent_data/{env_name}/`, `figsize=(14, 7)`, `fontsize=12`.
  **Axis labels, legends and titles are in German** ("Schritte", "Kapital (€)", "Preis
  (€/kWh)", "Kaufen"/"Verkaufen", "Marktpreis"). Keep new plots German to match the thesis.
- **Language**: code, comments and docstrings are English; user-facing plot text and many
  comments in `plotting.py` are German. Follow the file you are editing.
- **Generated output is gitignored**: `trade_logs/`, `agent_data/`, `logging/`, `img/` and
  `agents/*.zip` (staged model copies). The rule is `/agents/*.zip`, anchored and non-recursive,
  so the committed models under `agents/results/` stay tracked. The `*_ext` data files are
  ignored too (§13). Nothing else generated should be committed.

## 12. Git workflow

- Default branch `main`; remote `novatc/energain`.
- Work on your assigned feature branch; push with `git push -u origin <branch>`.
- Commit messages are short, lowercase, imperative-ish ("added final models", "latest env
  changes"). Match that style.
- Do not open a pull request unless explicitly asked.

## 13. Improved variants (2026-08)

Six env variants added on top of the original seven. **The original seven and their data files
are untouched**, so every committed model still loads and `base` / `trend` still reproduce the
README to the cent (§8). Register-and-copy was used deliberately rather than editing the
existing envs, because changing an observation shape breaks the committed models.

| variant | what it adds |
|---|---|
| `base_state` | `[soc, savings]` in the observation; DA amount clipped to battery + budget |
| `trend_compact` | `base_state` + a compact price-history block (lags 1/2/24/168 h, 24 h mean & std, price/mean ratio, EWMA) |
| `multi_compact` | `MultiMarket` + that price block + `[soc, savings]`, plus budget clipping |
| `*_ext` | the same three pointed at `*_ext` data: DA + 7 solar columns, PRL + 6 time columns |

### Why each one

- **The agent could not see its own battery.** `BaseEnv.get_observation()` returns the market
  row only — no SOC, no savings — yet `is_trade_valid` rejects on exactly those. On the average
  year 5860 of `base`'s 6362 rejected trades carried the `'battery'` label. The multi-market
  envs had already solved this mechanically with `clip_trade_amount`; the DA-only envs never
  got it. Measured with a **random** policy (so the policy cannot confound it), over one pass of
  the average year:

  | | invalid | battery | market rejected | trades | holds |
  |---|---|---|---|---|---|
  | `base` | 7676 | 4221 | 3455 | 1035 | 72 |
  | `base_state` | 3994 | **0** | 3994 | 1091 | 3698 |

  Infeasible-trade rejections go to zero. What remains is `'market rejected'`, which is a real
  market outcome — you must bid above market to buy and below it to sell. Holds rise sharply
  because a clipped-to-nothing trade falls under `trade_threshold` and scores `+1` instead of
  the `-10` penalty.

- **`is_trade_valid` conflates two rejection causes.** It labels an unaffordable buy `'battery'`
  whenever `savings > 0`, and only says `'savings'` when savings has actually hit zero. So the
  published `'battery'` counts overstate genuine battery infeasibility. The new envs report the
  real cause and clip to affordability as well; the original envs are left as-is.

- **The signal is the price's own history, not the exogenous columns.** Measured on the training
  set: price autocorrelation is 0.988 at 1 h, 0.914 at 24 h, 0.844 at 168 h, while hour-of-day
  alone explains only 2 % of price variance (R² = 0.020) and the best exogenous column is
  `prediction` at r = 0.148. `TrendEnv` spends 48 of its 72 dimensions on cyclical time columns
  tiled over 8 hours — near-duplicates — and its window is too short to reach the daily or
  weekly lag. `TrendCompact` gets the daily and weekly structure in 19 dimensions.

- **`get_average_price()` already existed and was never called.** Both market sims maintain a
  `price_history` deque and expose the accessor; no env used it. Worse, only `step()` appended,
  never `random_walk()`, so during *training* the deque stayed empty and the accessor returned
  0. `random_walk` now appends too. This is behaviourally inert for the original envs, which
  never read it.

- **Solar and PRL time features are computed and thrown away.** `preprocess_data.py` loads
  `solar_power_*.csv` purely to borrow its date index, then selects
  `['price','consumption','prediction']` and drops all 7 solar columns. `preprocess_prl.py`
  computes `day_of_week`/`month`/`hour`, writes them to `env_prl.csv`, then drops them, leaving
  PRL observations as bare `(price, amount)`. `preprocess_ext.py` keeps both, writing new
  `*_ext` files. **Temper expectations**: the solar columns correlate with price at |r| ≤ 0.13,
  about the same as `consumption` (0.070) which is already used. The one mild standout is
  `sun_elevation` against the price *change* (r = +0.084), higher than any currently-used column.

### Measuring a change

Training is not reproducible by default (§6), so a single before/after run is noise. Use
`benchmark.py`, which seeds every RNG and repeats each variant:

```bash
python benchmark.py --envs base base_state --training_steps 20000 --seeds 0 1 2 --month 0
```

It reports mean and standard deviation of closing capital plus the trade/invalid/battery/hold
counts. Read any profit difference against the std before believing it.

### Pilot results (20 000 steps, 3 seeds, `--month 0`)

**This is a pilot, not a verdict.** The committed models were trained for 500 k-1 500 k steps;
these ran for 20 000. Closing capital, mean ± sd over seeds 0/1/2:

| env | Kapital mean | sd | trades | invalid | battery | holds |
|---|---|---|---|---|---|---|
| `base` | 2533,73 | 979,19 | 817 | 7894 | 5891 | 72 |
| `base_state` | 4163,61 | 2712,97 | 790 | 1689 | **0** | 6304 |
| `trend` | 935,82 | 998,71 | 606 | 8177 | 6844 | 39 |
| `trend_compact` | 5299,25 | 4051,43 | 1026 | 2729 | **0** | 5028 |
| `multi` | 4831,06 | 1891,81 | 1303 | 722 | 49 | 2061 |
| `multi_compact` † | 5084,76 | 1846,22 | 1188 | 347 | **0** | 3860 |

† Measured **before** the flexibility-band fix in §14. `multi_compact` has changed since, so this
row no longer describes the code — re-run it before quoting.

**None of the profit differences are statistically significant at n = 3** (Welch:
base→base_state p = 0,41; trend→trend_compact p = 0,20; multi→multi_compact p = 0,88). Seed variance in this environment is
large enough to swamp the effect at this sample size. Do not quote the capital figures as a
result — run more seeds and more steps first.

What *is* solid is structural, because clipping removes the failure mode by construction rather
than by learning: battery rejections go to zero in all three new variants, and total invalid
trades drop 79 % (`base`), 67 % (`trend`) and 52 % (`multi`).

The multi-market pair is the weakest case for the change, and expectedly so: `MultiMarket`
already clipped against the PRL flexibility band via `clip_trade_amount`, so it only had 49
battery rejections to remove. `multi_compact` gains 5 % of capital, which at p = 0,88 is
indistinguishable from noise. Item 1 of the improvement list only ever mattered for the
day-ahead-only envs.

One incidental observation worth noting: at this budget `trend` (935) underperforms plain `base`
(2533), while `trend_compact` (5299) leads. That is consistent with the sample-efficiency
argument — a 72-dimensional observation needs far more samples than 19 to become useful — and
with the committed `trend` model needing 1 000 k steps to beat `base`.

### Regenerating the extended data

The `*_ext` CSVs are **gitignored, not committed** — unlike the repo's committed `eval_data/`,
they are fully reproducible from committed inputs, so run `python preprocess_ext.py` once before
using any `*_ext` variant (about 16 MB, a few seconds). `preprocess_ext.py` writes only `*_ext`
files and never touches the committed CSVs. It joins the
solar columns onto `unscaled_train_data.csv` positionally, asserting first that the row counts
match and the price columns agree row-for-row. It skips the unseeded noise injection that
`preprocess_prl.py` applies, so its output is reproducible.


## 14. The PRL flexibility band is broken in the original multi-market envs

Spotted in the `*_soc_and_boundaries.svg` plots: the SOC sits **above** the upper bound for long
stretches. It is real, not a plotting artefact. On the average year the committed `multi` model
spends **580 steps (6,6 %) above the upper bound, by up to 400,7 kWh**.

### Cause

`set_boundaries()` derives the band from the PRL offer alone and never looks at the SOC:

```python
self.upper_bound = ((capacity - 0.5 * amount_prl) / capacity) * 1000   # = 1000 - 0.5a
self.lower_bound = ((0.5 * amount_prl) / capacity) * 1000              # =        0.5a
```

The band is always centred on 500 and narrows as the commitment grows, and nothing ever moves
the SOC into it. Because `amount_prl = min(amount_prl, soc)`, an agent committing its whole
charge satisfies `soc > 1000 - 0.5·soc` whenever **`soc > 2/3 · capacity ≈ 667`**, which puts the
SOC outside its own band. The worst observed step matches exactly: `soc = 933,8`, band
`[466,9, 533,1]`, i.e. `amount_prl = 933,8`.

Two consequences:

1. **It sells reserve it cannot deliver.** FCR is symmetric — committing 933,8 kW means being
   able to absorb *and* deliver about 467 kWh, but at SOC 933,8 there are only 66 kWh of upward
   headroom. The revenue is booked anyway. PRL revenue dominates the multi-market results, so
   this likely inflates the published PRL rows (§8, where they already do not reproduce).
2. **`clip_trade_amount` silently inverts trades.** Once the SOC is outside the band,
   `min(amount, upper - soc)` goes negative for a buy and `perform_da_trade` then classifies it
   as a sell. On the average year: **403 buys executed as sells, 177 sells as buys** — exactly
   the 580 out-of-band steps. The sell branch is independently wrong too: `potential_soc =
   soc - amount` with `amount` already negative computes `soc + |amount|`, the wrong direction.

The `* 1000` is also a hardcoded capacity; it is only correct for `Battery(1000, ...)`.

### Fixed in `multi_compact` only

`multi`, `base_prl`, `multi_no_savings` and `multi_trend` are **left untouched** — they produced
the published results, and their committed models still load. `MultiCompact` gets:

- `clamp_prl_to_band()` — caps the offer at `2 · min(soc, capacity − soc)` so the band always
  contains the SOC.
- a rewritten `clip_trade_amount()` that never flips sign and treats the sell direction
  correctly.
- `set_boundaries()` scaled by the real capacity.

Verified with a random policy over the average year:

| | above upper | below lower | sign flips | max excess |
|---|---|---|---|---|
| `multi` | 184 | 0 | 184 | 487,9 kWh |
| `multi_compact` | **0** | **0** | **0** | **0** |

### The boundary penalty

`MultiMarket` adds **no reward at all** when `check_boundaries()` refuses a day-ahead trade —
not a penalty, not the hold bonus, just `reward += 0`. The policy gets no signal that it asked
for something impossible. Clipping alone has the same blind spot: it corrects the request
silently.

`MultiCompact` therefore charges `boundary_cost()`, **proportional to the fraction of the
request the band refused** — zero when the request survives untouched, the full `penalty` when
none of it does. A flat penalty was rejected because it would fire on nearly every step and
drown out the trade rewards.

A penalty is deliberately **not** used to deter the infeasible PRL commitment: a median accepted
block earns €34,05 (p90 €56,64) against a penalty of −10, so the agent would simply pay it and
sell undeliverable reserve. That case is fixed structurally by `clamp_prl_to_band()` instead.

**The first implementation was miscalibrated and its measurement is void.** `boundary_cost()`
originally charged for *any* clip, including the ordinary battery-capacity clipping that happens
with no PRL commitment active. Measured over the average year it fired on **50,1 % of steps** at
a mean of −3,26, totalling −28 647 against a closing capital in the low thousands — a constant
tax, not a signal. It now charges only when a PRL commitment has actually narrowed the band
(1,4 % of steps), and `clip_trade_amount()` shaves 1e-9 off the limit so a routine clip no longer
lands exactly on the bound and get refused by `check_boundaries()`'s strict `<` (boundary
refusals 4293 → 23).

Any ablation run before this fix compared against a broken penalty and needs re-running.

`benchmark.py` has the ablation arm:

```bash
python benchmark.py --envs multi multi_compact multi_compact_nopen \
                    --training_steps 100000 --seeds 0 1 2 3 4 5 6 7 --month 0
```

`multi_compact_nopen` is the same env with `boundary_penalty=False`.

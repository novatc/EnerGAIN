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

envs/                       Gymnasium envs — one file per experiment variant
  base_env.py               BaseEnv         — DA only
  no_savings_env.py         NoSavingsEnv    — DA only, "no savings constraint" (see §7)
  trend_env.py              TrendEnv        — DA only + 8-hour lookback
  base_prl.py               BasePRL         — DA + PRL, explicit market-choice action
  multi_market.py           MultiMarket     — DA + PRL in parallel each step
  multi_no_savings.py       MultiNoSavings  — MultiMarket without the savings check
  multi_trend.py            MultiTrend      — MultiMarket + 8-hour lookback
  assets/
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

Also unseeded: the `np.random.normal` noise in `preprocess_data.py` / `preprocess_prl.py`, and
SAC itself (`main.py` passes no `seed=`). Validation *is* deterministic (sequential stepping,
`model.predict(deterministic=True)`).

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
- **Do not commit** generated `agent_data/`, `trade_logs/`, `logging/`, `img/`, or newly
  trained `.zip` models, or model copies staged into `agents/`. `.gitignore` covers none of
  these.

## 12. Git workflow

- Default branch `main`; remote `novatc/energain`.
- Work on your assigned feature branch; push with `git push -u origin <branch>`.
- Commit messages are short, lowercase, imperative-ish ("added final models", "latest env
  changes"). Match that style.
- Do not open a pull request unless explicitly asked.

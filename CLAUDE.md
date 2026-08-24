# CLAUDE.md

Guidance for AI assistants working in the **EnerGAIN** repository.

## 1. What this project is

EnerGAIN is the code for a German-language thesis: *"Entwicklung einer intelligenten
Gebotsstrategie für den Strommarkt mittels Reinforcement Learning"* — learning bidding
strategies for a grid-scale battery on two German electricity markets:

- **Day-Ahead (DA)** market — hourly buy/sell of energy (`price` in €/kWh, `amount` in kWh).
- **Primärregelleistung / PRL** (Frequency Containment Reserve, FCR) — pay-as-bid capacity
  market sold in **4-hour blocks** (`price` in €/kWh, `amount` in kW).

A single **SAC** agent (Stable-Baselines3) acts in a custom Gymnasium environment that wraps a
battery plus one or both markets. There is no service, no API, no test suite — this is a
research codebase: preprocess CSVs → train → validate → produce SVG plots for the thesis.

**It is an archived research project.** Prefer minimal, surgical changes. Do not restructure
directories, rename modules, or "modernize" the code unless explicitly asked.

## 2. Repository layout

```
main.py                     Training entry point (SAC)
validation.py               Evaluation entry point (loads a saved model, writes logs + stats)
plotting.py                 Cross-agent comparison plots; runs at import (script, not a module)
preprocess_data.py          data/clean/* -> data/in-use/unscaled_train_data.csv (DA training data)
preprocess_prl.py           data/prm/prl.csv -> data/prm/{env_prl,preprocessed_prl}.csv
create_da_eval_set.py       data/in-use/env_data.csv -> data/in-use/eval_data/*_da.csv
create_prl_eval_set.py      data/prm/env_prl.csv -> data/in-use/eval_data/*_prl.csv

envs/                       Gymnasium environments (one file per experiment variant)
  base_env.py               BaseEnv           — DA only
  no_savings_env.py         NoSavingsEnv      — DA only, "no savings constraint" variant
  trend_env.py              TrendEnv          — DA only + 8-hour lookback observation
  base_prl.py               BasePRL           — DA + PRL, explicit market-choice action
  multi_market.py           MultiMarket       — DA + PRL in parallel each step
  multi_no_savings.py       MultiNoSavings    — MultiMarket without the savings check
  multi_trend.py            MultiTrend        — MultiMarket + 8-hour lookback observation
  assets/
    battery.py                      Battery model (capacity, SOC, charge/discharge efficiency)
    dayahead.py                     DayAhead market simulator (stepping, offer acceptance)
    frequency_containment_reserve.py  PRL/FCR market simulator
    plot_engien.py                  Per-agent plots called from env.render()  [sic: "engien"]
    env_utilities.py                get_model_names(), moving_average()

cutsom_wrappers/            [sic: "cutsom"] CustomNormalizeObservation wrapper
callbacks/summary_writer.py SummaryWriterCallback — pushes info-dict values to TensorBoard
dummy_bots/                 Non-RL baselines (threshold, moving average) — run as scripts
data_preprocess/            One-off scripts turning data/original/* into data/clean/*
vis/                        One-off dataset plots for the thesis (data exploration)
plots/                      Committed SVG output from vis/
diagram/                    draw.io diagram of the env step flow
agents/results/{base,multi} Committed trained SAC models (.zip)
*.job                       SLURM batch scripts for the university cluster
```

**Note the two intentional typos in directory/module names** — `cutsom_wrappers` (not
`custom_`) and `envs/assets/plot_engien.py` (not `plot_engine`). Imports depend on them; do
not rename unless asked, and match them exactly when writing imports.

There are no `__init__.py` files; the packages work as Python 3 namespace packages, which is
why **all top-level scripts must be run from the repository root**.

## 3. Environment setup

```bash
pip install -r requirements.txt
pip install stable-baselines3==2.0.0a13   # README pins this alpha explicitly
```

`requirements.txt` is incomplete for the full workflow. Also needed:

- **`scipy`** — `envs/assets/plot_engien.py` imports `scipy.stats.gaussian_kde`, so any
  `--plot` run fails without it.
- **`openpyxl`** — only for `data_preprocess/solar_power.py` (`to_excel`).
- **`tensorboard`** — only if you want to read the logs `main.py` writes.

If you add a dependency that the code actually imports, add it to `requirements.txt`.

## 4. Running things

### Train

```bash
python main.py --training_steps 500_000 --env base --save
```

- `--training_steps` (int, required)
- `--env` (required): `base | trend | no_savings | base_prl | multi | multi_no_savings | multi_trend`
- `--save`: without it, the model is trained and thrown away.

Training always reads `data/in-use/unscaled_train_data.csv` (and
`data/prm/preprocessed_prl.csv` for PRL envs), registers the env with `validation=False`,
wraps it in `CustomNormalizeObservation` + `NormalizeReward`, and trains
`SAC("MlpPolicy", ...)` with `NormalActionNoise(sigma=0.1)`.

Output: `agents/sac_{env}_{steps/1000}k_{DD.MM-HH-MM}.zip` and TensorBoard logs under
`logging/tensorboard_logs/{env}/` (`main.py` creates `logging/` itself).

### Evaluate

```bash
python validation.py --env base --month 0 --plot
```

- `--env`: same choices as training.
- `--month` (default `5`): `1`–`12` picks `data/in-use/eval_data/month_{n}_data_{da,prl}.csv`;
  `0` uses the whole synthetic average year (`average_da_year.csv` / `average_prl_year.csv`).
- `--episodes` (default `1`), `--plot` (calls `env.render()`, writing SVGs).

**Two things bite every time:**

1. `env_utilities.get_model_names()` only scans `agents/` **non-recursively** and picks the
   *first* file starting with `sac_{env}_`. The committed models live in
   `agents/results/base/` and `agents/results/multi/`, so copy the one you want up first:
   ```bash
   cp agents/results/base/sac_base_500.0k_22.12-13-49.zip agents/
   ```
2. `validation.py` writes to directories it does not create. Make them first:
   ```bash
   mkdir -p trade_logs/invalid agent_data
   ```
   (`plotting.py` additionally writes to `img/`, and `plot_engien.py` creates
   `agent_data/{model_name}/` itself.)

Evaluation writes `trade_logs/{model}_trades.csv`,
`trade_logs/invalid/{model}_invalid_trades.csv`, and `agent_data/{model}_stats.json`.

### Compare agents / regenerate thesis figures

`plotting.py` has **no `main` guard** — it loads every CSV in `trade_logs/` and plots on
import. Run it only after `validation.py` has produced those CSVs:

```bash
mkdir -p img && python plotting.py
```

Scripts in `vis/`, `data_preprocess/`, and `dummy_bots/` use paths relative to *their own*
directory (`../data/...`), so run them from inside that directory:

```bash
cd vis && python year_price_plotter.py
```

### Cluster

`*.job` files are SLURM scripts (`sbatch base.job`). Some use `--array=1-3` to sweep step
counts; they hardcode a `--mail-user`. `reward_boosting.job` also requests a GPU.

## 5. Environment / agent design

### Observation and action spaces

| env | class | markets | obs dim | action dim | action vector |
|---|---|---|---|---|---|
| `base` | `BaseEnv` | DA | 9 | 2 | `[da_price, da_amount]` |
| `no_savings` | `NoSavingsEnv` | DA | 9 | 2 | `[da_price, da_amount]` |
| `trend` | `TrendEnv` | DA | 72 (9 × 8h) | 2 | `[da_price, da_amount]` |
| `base_prl` | `BasePRL` | DA + PRL | 14 | 5 | `[prl_choice, prl_price, prl_amount, da_price, da_amount]` |
| `multi` | `MultiMarket` | DA + PRL | 14 | 4 | `[prl_price, prl_amount, da_price, da_amount]` |
| `multi_no_savings` | `MultiNoSavings` | DA + PRL | 14 | 4 | same as `multi` |
| `multi_trend` | `MultiTrend` | DA + PRL | 89 ((9+2) × 8h + 1) | 4 | same as `multi` |

Obs dims assume the current CSVs: 9 DA columns, 2 PRL columns. Multi-market observations
append `[prl_cooldown, lower_bound, upper_bound]` (the trend variant appends only
`prl_cooldown`). Action bounds: DA price `[0, 1]` €/kWh, DA amount `[-1000, 1000]` kWh
(negative = sell), PRL price `[0.001, 0.5]`, PRL amount `[0, 1000]`.

### Shared constants (identical across all envs)

- `Battery(capacity=1000, soc=500)`, charge/discharge efficiency `0.925`, `reset()` → SOC 500.
- `self.savings = 50` € initial capital; `reset()` restores it.
- `self.penalty = -10` for any invalid or rejected action.
- `self.trade_threshold = 10` — a DA amount in `(-10, 10)` counts as **hold**, not a trade.
- `trend_horizon = 8` hours in the two trend variants.
- Holding reward is `1` everywhere **except `TrendEnv`, which returns `5`** — deliberate, not
  a typo; do not "harmonize" it.

### Step semantics

- **Training** (`validation=False`): the market advances via `random_walk(24 * 30)` — a random
  start position, then 30 simulated days before it resamples and returns `truncated=True`
  (which triggers `self.reset()`).
- **Validation** (`validation=True`): plain sequential `step()` through the eval CSV.
- In multi-market envs the PRL market drives the clock and the DA step is force-synced via
  `day_ahead.set_step(prl.get_current_step())`.

### Market acceptance rules

`DayAhead.accept_offer` accepts a **buy** when `offer_price > market_price` and a **sell**
when `offer_price < market_price` (you must bid *against* yourself to clear). Profit is then
booked at the *market* price, not the offered price. `FrequencyContainmentReserve.accept_offer`
accepts when `offer_price < market_price`; PRL is pay-as-bid, so revenue is
`price * amount * 4` (the 4-hour block).

### PRL constraints (multi-market envs)

- `prl_cooldown` is set to `4` after a PRL trade and decremented once per step; PRL bids are
  only allowed at `prl.get_current_step() % 4 == 0` with cooldown expired.
- `set_boundaries(amount_prl)` narrows the allowed SOC band so the reserved capacity stays
  available; `clip_trade_amount` then clips DA trades into that band.
- `check_prl_constraints` in `multi_market.py` reads oddly:
  `if self.prl_cooldown <= 0 == self.prl.get_current_step() % 4:`. That is a Python **chained
  comparison** meaning `cooldown <= 0 and 0 == step % 4` — it is correct as written. Leave it
  alone unless the behaviour itself is in question.

### Known inconsistencies between variants

These are real differences in the committed code. Flag them before "fixing" — they may reflect
how the thesis results were produced:

- `no_savings_env.py` is byte-for-byte `base_env.py` apart from the class name and the plot
  labels: **its `is_trade_valid` still enforces the savings check**. Only the *multi-market*
  `MultiNoSavings` actually drops it (`is_trade_valid` checks battery capacity only).
- `BaseEnv` / `NoSavingsEnv` / `TrendEnv` sell by calling `battery.charge(amount)` with a
  negative amount, and update `self.savings` directly. The multi-market envs call
  `battery.discharge(amount)` and update savings via `self.savings += profit`.
- `MultiMarket.step` calls both `self.reward_log.append(...)` and `self.log_step(reward)`,
  which appends to `reward_log` again — the multi-market reward log is double-counted.
- `main.py` and `validation.py` accept `--env reward_boosting` in `argparse`, but there is no
  `reward_boosting` entry in `env_params`, so it raises `ValueError` immediately.
  `reward_boosting.job` is therefore dead. Same for the README, which lists a smaller set of
  envs than the code supports.
- `main.py` builds the env twice (once outside the `try`, once inside); harmless but redundant.
- `multi_no_savings` (the job file) has no `.job` extension, unlike its siblings.
- The `SummaryWriterCallback` in `callbacks/` is imported by `main.py` but never passed to
  `model.learn()`.

## 6. Data pipeline

```
data/original/    Raw German market/weather downloads (SMARD, DWD, PV sim); semicolon-separated,
                  German decimal commas, '-' for missing. Gitignored in .gitignore but committed.
      |  data_preprocess/*.py   (run from inside data_preprocess/)
data/clean/       Per-source cleaned CSVs + dataset_01102018_01012023.csv
      |  preprocess_data.py     (run from repo root)
data/in-use/      unscaled_train_data.csv  <- DA training data
                  env_data.csv             <- dated intermediate (see caveat below)
      |  create_da_eval_set.py / create_prl_eval_set.py
data/in-use/eval_data/  month_{1..12}_data_{da,prl}.csv, average_{da,prl}_year.csv

data/prm/prl.csv  Raw PRL export (';'-separated, German dates)
      |  preprocess_prl.py
data/prm/         env_prl.csv (dated, for eval-set creation) and preprocessed_prl.csv (env input)
```

Column schemas:

- **DA training** (`data/in-use/unscaled_train_data.csv`, 9 cols):
  `price, consumption, prediction, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, month_sin, month_cos`
- **DA eval** (`data/in-use/eval_data/*_da.csv`, 9 cols):
  `price, consumption, prediction, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos`
  — same count, **different cyclical columns** (day-of-month instead of day-of-week).
- **PRL** (`data/prm/preprocessed_prl.csv`, `*_prl.csv`, 2 cols): `price, amount`

Conventions: prices are converted to **€/kWh** (raw €/MWh ÷ 1000); time features are
sine/cosine encoded; `price` is written as the CSV index by the preprocessing scripts but read
back as an ordinary column (`pd.read_csv` with no `index_col`), which is why `shape[1]`
includes it. Envs read observation bounds straight from `df.min()` / `df.max()`.

**Caveat: the pipeline is not runnable end-to-end as committed.** `preprocess_data.py` writes
`data/in-use/env_data.csv` and `preprocess_prl.py` writes `data/prm/env_prl.csv`; neither file
is in the repo, so `create_da_eval_set.py` / `create_prl_eval_set.py` cannot run until you
re-run the preprocessing step. (`data/clean/env_data.csv` is a *scaled* artifact and is not a
drop-in substitute.) The already-generated `eval_data/` CSVs are committed, so training and
validation work without re-running any of this.

## 7. Trade log format

Every env logs trades as **9-tuples**, and the same order is used by the dummy bots, by
`plot_engien.py` (which indexes positionally), and by the CSV headers in `validation.py`:

```
(step, type, market_price, offered_price, amount, reward, case, soc, savings)
   0     1         2             3           4       5      6     7      8
```

`type` ∈ `{'buy', 'sell', 'reserve'}`. `case` is `'accepted'` / `'prl accepted'` for valid
trades, or the rejection reason (`'battery'`, `'savings'`, `'market rejected'`) for entries in
`invalid_trades`. Holds go to a separate `self.holding` list of `(step, 'hold')`.

**If you change this tuple, you must update every consumer**: all seven envs, both
`dummy_bots/`, all positional indexing in `plot_engien.py` (`trade[3]`, `trade[4]`,
`trade[8]`, …), and the `columns=[...]` lists in `validation.py`.

## 8. Conventions to follow

- **Python style**: plain functions/classes, reStructuredText-ish docstrings
  (`:param x:` / `:return:`), 4-space indent, ~120-char lines, f-strings. No type checker,
  no linter, no formatter is configured — match the surrounding file rather than reformatting.
- **Adding an environment variant**: copy the closest existing env file wholesale, rename the
  class, change the behaviour, and update the `model_name` string in every `render()` call.
  Then register it in the `env_params` dict in **both** `main.py` and `validation.py` (and add
  it to the `choices=[...]` lists), and add a `*.job` file if it is meant for the cluster.
  Duplication between env files is the established pattern here — do not refactor the seven
  envs into a shared base class unless explicitly asked.
- **Plot output**: SVG at `dpi=1200` into `agent_data/{model_name}/`, figures `figsize=(14, 7)`,
  `fontsize=12` labels/legends. **Axis labels, legends and titles are in German**
  ("Schritte", "Kapital (€)", "Preis (€/kWh)", "Kaufen"/"Verkaufen", "Marktpreis"). Keep new
  plots German to match the thesis.
- **Language**: code, comments and docstrings are English; user-facing plot text and some
  comments in `plotting.py` are German. Follow the file you are editing.
- **Do not commit** generated `agent_data/`, `trade_logs/`, `logging/`, or `img/` output, or
  newly trained `.zip` models, unless asked. `.gitignore` does not currently cover them.

## 9. Git workflow

- Default branch: `main`. Remote: `novatc/energain`.
- Work on the feature branch you were assigned; push with `git push -u origin <branch>`.
- Commit messages here are short, lowercase, imperative-ish ("added final models",
  "latest env changes"). Match that style.
- Do not open a pull request unless explicitly asked.

"""
Build the extended datasets: day-ahead data with the solar columns kept, and PRL data with its
time features kept.

Both are currently discarded. preprocess_data.py loads solar_power_*.csv only to borrow its date
index, then selects ['price', 'consumption', 'prediction'] and drops every solar column.
preprocess_prl.py computes day_of_week / month / hour, writes them to env_prl.csv, and then drops
them again, leaving the PRL observation as just (price, amount).

Everything here is written to new *_ext files. The committed CSVs are never touched, so the
existing envs and their committed models keep reproducing their published numbers.

Run from the repository root:

    python preprocess_ext.py
"""
import numpy as np
import pandas as pd

DATASET = 'data/clean/dataset_01102018_01012023.csv'
TRAIN_IN = 'data/in-use/unscaled_train_data.csv'
TRAIN_OUT = 'data/in-use/unscaled_train_data_ext.csv'
PRL_RAW = 'data/prm/prl.csv'
PRL_OUT = 'data/prm/preprocessed_prl_ext.csv'
EVAL_DIR = 'data/in-use/eval_data'

# The solar columns, renamed from the original German headers to something an env can print.
SOLAR_COLUMNS = {
    'Einstrahlung auf die Horizontale (kWh/m²)': 'ghi',
    'Diffusstrahlung auf die Horizontale (kWh/m²)': 'dhi',
    'Außentemperatur (°C)': 'temp_air',
    'Freifläche 01-Fläche Süd: Sonnenhöhe (rad)': 'sun_elevation',
    'Freifläche 01-Fläche Süd: Einstrahlung auf die geneigte Fläche (kWh/m²)': 'poa',
    'Freifläche 01-Fläche Süd: Modultemperatur (°C)': 'temp_module',
    'Eingespeiste Energie kWh': 'pv_energy',
}
SOLAR_ORDER = ['ghi', 'dhi', 'temp_air', 'sun_elevation', 'poa', 'temp_module', 'pv_energy']

EVAL_START, EVAL_END = '2019-01-01', '2022-12-31'


def cyclical(frame, column, period):
    """
    Add sine/cosine columns for a cyclical integer column.

    :param frame: the dataframe to add to.
    :param column: the source column, e.g. 'hour'.
    :param period: the length of the cycle, e.g. 24.
    :return: None, the frame is modified in place.
    """
    frame[f'{column}_sin'] = np.sin(frame[column] * (2.0 * np.pi / period))
    frame[f'{column}_cos'] = np.cos(frame[column] * (2.0 * np.pi / period))


def load_dataset():
    """
    Load the joined clean dataset, numeric and datetime indexed.

    :return: the dataframe with the solar columns renamed.
    """
    frame = pd.read_csv(DATASET, index_col=0, parse_dates=True)
    frame = frame.rename(columns=SOLAR_COLUMNS)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.interpolate(method='linear', limit_direction='both')


def build_train(dataset):
    """
    Append the solar columns to the committed day-ahead training data.

    The two frames come from the same source with a reset index and are verified to line up row
    for row on price before joining.

    :param dataset: the joined clean dataset.
    :return: None, writes TRAIN_OUT.
    """
    train = pd.read_csv(TRAIN_IN)
    if len(train) != len(dataset):
        raise ValueError(f"row mismatch: {TRAIN_IN} has {len(train)}, {DATASET} has {len(dataset)}")
    if not np.allclose(train['price'].values, dataset['price'].values, atol=1e-9):
        raise ValueError("price columns do not line up row for row; refusing to join positionally")

    solar = dataset[SOLAR_ORDER].reset_index(drop=True)
    out = pd.concat([train, solar], axis=1)
    out.to_csv(TRAIN_OUT, index=False)
    print(f"{TRAIN_OUT}: {out.shape[0]} rows x {out.shape[1]} cols "
          f"({train.shape[1]} + {len(SOLAR_ORDER)} solar)")


def build_da_eval(dataset):
    """
    Rebuild the day-ahead evaluation sets with the solar columns kept.

    Mirrors create_da_eval_set.py: average every (month, day, hour) slot over 2019-2022, then
    cyclically encode month, day and hour. Column order matches the committed eval sets, with the
    solar block appended.

    :param dataset: the joined clean dataset.
    :return: None, writes the *_ext eval CSVs.
    """
    window = dataset.loc[EVAL_START:EVAL_END].copy()
    window['month'] = window.index.month
    window['day'] = window.index.day
    window['hour'] = window.index.hour

    aggregated = ['price', 'consumption', 'prediction'] + SOLAR_ORDER
    averaged = window.groupby(['month', 'day', 'hour'])[aggregated].mean().reset_index()

    cyclical(averaged, 'month', 12)
    cyclical(averaged, 'day', 31)
    cyclical(averaged, 'hour', 24)

    ordered = (['price', 'consumption', 'prediction', 'month_sin', 'month_cos',
                'day_sin', 'day_cos', 'hour_sin', 'hour_cos'] + SOLAR_ORDER)

    averaged = averaged.sort_values(by=['month', 'day', 'hour'])
    for month in range(1, 13):
        slice_ = averaged[averaged['month'] == month]
        slice_[ordered].to_csv(f'{EVAL_DIR}/month_{month}_data_ext_da.csv', index=False)
    averaged[ordered].to_csv(f'{EVAL_DIR}/average_da_year_ext.csv', index=False)
    print(f"{EVAL_DIR}/average_da_year_ext.csv: {len(averaged)} rows x {len(ordered)} cols "
          f"(+ 12 monthly files)")


def build_prl():
    """
    Rebuild the PRL data keeping the time features that preprocess_prl.py drops.

    Same cleaning as preprocess_prl.py (rename, numeric coercion, linear interpolation, euro per
    kWh) but without the noise injection, which is unseeded and would make the file
    irreproducible.

    :return: None, writes PRL_OUT and the *_ext PRL eval CSVs.
    """
    prl = pd.read_csv(PRL_RAW, index_col=0, sep=';')
    prl.columns = ['start', 'end', 'amount', 'price']
    prl.index = pd.to_datetime(prl.index, dayfirst=True)
    prl['price'] = pd.to_numeric(prl['price'], errors='coerce').interpolate(method='linear') / 1000
    prl['amount'] = pd.to_numeric(prl['amount'], errors='coerce').interpolate(method='linear')
    prl = prl.dropna(subset=['price', 'amount'])

    prl['hour'] = pd.to_numeric(prl['start'].str.split(':').str[0], errors='coerce')
    prl['day_of_week'] = prl.index.dayofweek
    prl['month'] = prl.index.month
    cyclical(prl, 'hour', 24)
    cyclical(prl, 'day_of_week', 7)
    cyclical(prl, 'month', 12)

    ordered = ['price', 'amount', 'hour_sin', 'hour_cos',
               'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
    prl[ordered].to_csv(PRL_OUT, index=False)
    print(f"{PRL_OUT}: {len(prl)} rows x {len(ordered)} cols")

    window = prl.loc[EVAL_START:EVAL_END].copy()
    window['day'] = window.index.day
    averaged = window.groupby(['month', 'day', 'hour'])[['price', 'amount']].mean().reset_index()
    cyclical(averaged, 'hour', 24)
    cyclical(averaged, 'month', 12)
    averaged['day_of_week_sin'] = 0.0
    averaged['day_of_week_cos'] = 1.0
    averaged = averaged.sort_values(by=['month', 'day', 'hour'])
    for month in range(1, 13):
        slice_ = averaged[averaged['month'] == month]
        slice_[ordered].to_csv(f'{EVAL_DIR}/month_{month}_data_ext_prl.csv', index=False)
    averaged[ordered].to_csv(f'{EVAL_DIR}/average_prl_year_ext.csv', index=False)
    print(f"{EVAL_DIR}/average_prl_year_ext.csv: {len(averaged)} rows x {len(ordered)} cols "
          f"(+ 12 monthly files)")


if __name__ == '__main__':
    data = load_dataset()
    build_train(data)
    build_da_eval(data)
    build_prl()

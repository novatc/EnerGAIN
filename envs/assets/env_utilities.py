import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def rescale_value_price(scaled_value: float, test_mode=False) -> float:
    # Load the scaler

    if test_mode:
        price_scaler = joblib.load('new_price_scaler.pkl')
    else:
        price_scaler = joblib.load('new_price_scaler.pkl')

    # Get the original savings values
    savings_original = price_scaler.inverse_transform(np.array(scaled_value).reshape(-1, 1))
    return float(savings_original[0][0])  # Convert numpy float to Python float


def rescale_list_price(scaled_list: list, test_mode=False) -> list:
    # Load the scaler
    if test_mode:
        price_scaler = joblib.load('new_price_scaler.pkl')
    else:
        price_scaler = joblib.load('new_price_scaler.pkl')

    # Get the original savings values
    savings_original = price_scaler.inverse_transform(np.array(scaled_list).reshape(-1, 1))
    return savings_original.flatten().tolist()  # Convert numpy array to Python list


def rescale_value_amount(scaled_value: float) -> float:
    # Load the scaler
    amount_scaler = joblib.load('new_amount_scaler.pkl')

    # Get the original charge values
    charge_original = amount_scaler.inverse_transform(np.array(scaled_value).reshape(-1, 1))
    return float(charge_original[0][0])  # Convert numpy float to Python float


def rescale_list_amount(scaled_list: list) -> list:
    # Load the scaler
    amount_scaler = joblib.load('new_amount_scaler.pkl')

    # Get the original charge values
    charge_original = amount_scaler.inverse_transform(np.array(scaled_list).reshape(-1, 1))
    return charge_original.flatten().tolist()  # Convert numpy array to Python list


def scale_list(value_list: np.array, name: str) -> np.array:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(np.array(value_list).reshape(-1, 1)).reshape(-1).tolist()
    joblib.dump(scaler, f'{name}_scaler.pkl')
    return scaled_values


def scale_value(value: float) -> float:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_value = scaler.fit_transform(np.array(value).reshape(-1, 1)).reshape(-1)
    return scaled_value[0]




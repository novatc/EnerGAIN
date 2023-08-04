import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def rescale_value_price(scaled_value: float, test_mode=False) -> float:
    """
    Rescale a price value that has been previously scaled, back to its original value.

    :param scaled_value: The scaled value to be rescaled.
    :param test_mode: If true, use the test mode scaler, otherwise use the normal scaler.
    :return: The rescaled value.
    """
    # Load the scaler
    price_scaler = joblib.load('new_price_scaler.pkl')

    # Get the original savings values
    savings_original = price_scaler.inverse_transform(np.array(scaled_value).reshape(-1, 1))
    return float(savings_original[0][0])  # Convert numpy float to Python float


def rescale_list_price(scaled_list: list, test_mode=False) -> list:
    """
    Rescale a list of price values that have been previously scaled, back to their original values.

    :param scaled_list: The list of scaled values to be rescaled.
    :param test_mode: If true, use the test mode scaler, otherwise use the normal scaler.
    :return: The list of rescaled values.
    """
    # Load the scaler
    price_scaler = joblib.load('new_price_scaler.pkl')

    # Get the original savings values
    savings_original = price_scaler.inverse_transform(np.array(scaled_list).reshape(-1, 1))
    return savings_original.flatten().tolist()  # Convert numpy array to Python list


def rescale_value_amount(scaled_value: float) -> float:
    """
    Rescale an amount value that has been previously scaled, back to its original value.

    :param scaled_value: The scaled value to be rescaled.
    :return: The rescaled value.
    """
    # Load the scaler
    amount_scaler = joblib.load('new_amount_scaler.pkl')

    # Get the original charge values
    charge_original = amount_scaler.inverse_transform(np.array(scaled_value).reshape(-1, 1))
    return float(charge_original[0][0])  # Convert numpy float to Python float


def rescale_list_amount(scaled_list: list) -> list:
    """
    Rescale a list of amount values that have been previously scaled, back to their original values.

    :param scaled_list: The list of scaled values to be rescaled.
    :return: The list of rescaled values.
    """
    # Load the scaler
    amount_scaler = joblib.load('new_amount_scaler.pkl')

    # Get the original charge values
    charge_original = amount_scaler.inverse_transform(np.array(scaled_list).reshape(-1, 1))
    return charge_original.flatten().tolist()  # Convert numpy array to Python list


def scale_list(value_list: np.array, name: str) -> np.array:
    """
    Scale a list of values using the MinMaxScaler, with a feature range of (-1, 1).

    :param value_list: The list of values to be scaled.
    :param name: The name used for saving the trained scaler model.
    :return: The list of scaled values.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(np.array(value_list).reshape(-1, 1)).reshape(-1).tolist()
    joblib.dump(scaler, f'{name}_scaler.pkl')
    return scaled_values


def scale_value(value: float) -> float:
    """
    Scale a single value using the MinMaxScaler, with a feature range of (-1, 1).

    :param value: The value to be scaled.
    :return: The scaled value.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_value = scaler.fit_transform(np.array(value).reshape(-1, 1)).reshape(-1)
    return scaled_value[0]


def get_model_names():
    """
    Iterate over the models directory and return a list of model names.

    :return: The list of model names.
    """
    # iterate over the models directory and return a list of model names
    import os
    model_names = []
    current_dir = os.getcwd()
    for file in os.listdir(f'{current_dir}/agents'):
        if file.endswith('.zip'):
            model_names.append(file)
    return model_names

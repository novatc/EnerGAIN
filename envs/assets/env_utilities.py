import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def moving_average(data, window_size):
    """
    Calculate a moving average over 1D data using a specified window size.

    :param data: The data to calculate the moving average over.
    :param window_size: The window size to use for the moving average.
    :return: The moving average of the data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

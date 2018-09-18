import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import deep4cast.models as models
import deep4cast.custom_metrics as metrics
import deep4cast.utils as utils

from deep4cast.forecasters import Forecaster


def prepare_data(data_path, lag, horizon):
    """Returns prepared data for fitting."""

    # Loading the dataset and dropping unncecessary columns
    df = pd.read_pickle(data_path)
    data = df.drop(['time', 'month', 'day', 'hour'], axis=1)
    data = data.dropna()

    # We have real-valued and categorial features and we need to make sure
    # that we do the data preparation for them correctly
    real_values = ['load', 'temperature']
    categoricals = list(set(data.columns).difference(set(real_values)))

    # Let's create shifted categorical feature to include information about
    # the future's holiday structure. This will make it easier for our model
    # to do predictions on holidays
    shifted = data[categoricals].shift(-horizon)
    shifted = shifted.rename(
        columns={column: column + '_shifted' for column in shifted.columns})
    data = pd.concat([data, shifted], axis=1)

    # Format data into numpy array
    data = np.expand_dims(data.values, 0)

    # Now we need to sequentialize the training and testing dataset
    X_train, y_train = utils.sequentialize(
        data[:, :-horizon, :], lag, horizon, targets=[0])
    X_test, y_test = utils.sequentialize(
        data[:, -horizon - lag:, :], lag, horizon, targets=[0])

    # Transform the datasets to log-scale
    X_train = utils.transform(X_train, np.log1p, targets=[0])
    X_test = utils.transform(X_test, np.log1p, targets=[0])

    # Rescale the datasets so that the neural networks can be fitted properly
    # and stably
    scaler_X = utils.VectorScaler(targets=[0, 1])
    scaler_X.fit(X_train)
    scaler_y = utils.VectorScaler()
    scaler_y.fit(y_train)
    X_train = scaler_X.transform(X_train)
    y_train = scaler_y.transform(y_train)
    X_test = scaler_X.transform(X_test)


def main():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    # Get the data
    df = prepare_data(args.data_path)

    # Store data
    df.to_pickle(args.output_path)


if __name__ == '__main__':
    main()

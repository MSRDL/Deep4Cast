# -*- coding: utf-8 -*-
"""This script provides command line access for testing forecasters on
custom data sets.

Example:
    $python deep4cast_shared_layer.py --data-path "./tutorials/timeseries_data.csv" --lookback_period 20 --test-fraction 0.1 --epochs 100

"""
import argparse

from deep4cast.forecasters import FlexibleForecaster
from pandas import read_table
from deep4cast.utils import compute_mape


def main(args):
    """Main function that handles forecasting given list of arugments."""
    # Load training at test data from file input
    print("Loading datasets.")
    df = read_table(args.data_path, sep=',')
    ts = df.values

    # Prepare train and test set. Make sure to catch the case when the user
    # did not supply a test. We use the end of the time series for testing
    # because of lookahead bias.
    if args.test_fraction:
        test_length = int(len(df) * args.test_fraction)
        train_length = len(df) - test_length
        ts_train = ts[:-test_length]
        ts_test = ts[-test_length - args.lookback_period:]
    else:
        ts_train = ts

    # Build model with default topology. This can be changed for production
    # purposes.
    topology = [({'layer': 'Conv1D', 'id': 'c1', 'parent': 'input'},
                 {'filters': 64, 'kernel_size': 5, 'activation': 'elu'}),
                ({'layer': 'MaxPooling1D', 'id': 'mp1', 'parent': 'c1'},
                 {'pool_size': 3, 'strides': 1}),
                ({'layer': 'Conv1D', 'id': 'c2', 'parent': 'mp1'},
                 {'filters': 64, 'kernel_size': 3, 'activation': 'elu'}),
                ({'layer': 'MaxPooling1D', 'id': 'mp2', 'parent': 'c2'},
                 {'pool_size': 4, 'strides': 2}),
                ({'layer': 'Conv1D', 'id': 'c3', 'parent': 'mp2'},
                 {'filters': 128, 'kernel_size': 3, 'activation': 'elu'}),
                ({'layer': 'MaxPooling1D', 'id': 'mp3', 'parent': 'c3'},
                 {'pool_size': 3, 'strides': 1}),
                ({'layer': 'Flatten', 'id': 'f1', 'parent': 'mp3'},
                 {}),
                ({'layer': 'Dense', 'id': 'd1', 'parent': 'f1'},
                 {'units': 128, 'activation': 'elu'}),
                ({'layer': 'Dense', 'id': 'output', 'parent': 'd1'},
                 {'units': 128, 'activation': 'elu'})]
    # topology = [({'layer': 'GRU', 'id': 'gru1', 'parent': 'input'},
    #              {'units': 128}),
    #             ({'layer': 'Dense', 'id': 'output', 'parent': 'gru1'},
    #              {'units': 1})]

    forecaster = FlexibleForecaster(
        topology,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    forecaster.fit(ts_train, lookback_period=args.lookback_period)

    # Print errors to screen using a specified metric function
    metric = compute_mape
    if args.test_fraction:
        print(
            'TRAIN \t Mean Absolute Percentage Error: {0:.1f}%'.format(
                metric(
                    forecaster, ts_train, ts[args.lookback_period:train_length]
                )
            )
        )
        print(
            'TEST \t Mean Absolute Percentage Error: {0:.1f}%'.format(

                metric(
                    forecaster, ts_test, ts[train_length:]
                )
            )
        )
    else:
        print(
            'TRAIN \t Mean Absolute Percentage Error: {0:.1f}%'.format(
                metric(
                    forecaster, ts_train, ts[args.lookback_period:]
                )
            )
        )


if __name__ == '__main__':
    # Collect all relevant command line arugments
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-d',
                            '--data-path',
                            metavar='|',
                            help="""Location of data set""",
                            required=True,
                            type=str)

    named_args.add_argument('-tf',
                            '--test-fraction',
                            metavar='|',
                            help="""Test fraction at end of dataset""",
                            required=False,
                            default=None,
                            type=float)

    named_args.add_argument('-lb',
                            '--lookback_period',
                            metavar='|',
                            help="""Lookback period""",
                            required=True,
                            type=int)

    named_args.add_argument('-e',
                            '--epochs',
                            metavar='|',
                            help="""Number of epochs to run""",
                            required=False,
                            default=50,
                            type=int)

    named_args.add_argument('-b',
                            '--batch-size',
                            metavar='|',
                            help="""Location of validation data""",
                            required=False,
                            default=10,
                            type=int)

    named_args.add_argument('-lr',
                            '--learning-rate',
                            metavar='|',
                            help="""Learning rate""",
                            required=False,
                            default=0.01,
                            type=float)

    args = parser.parse_args()
    main(args)

import argparse
import ast
import os
import re
import time
import threading
import numpy as np
from pandas import read_table
from deep4cast.forecasters import Forecaster
from deep4cast.metrics import adjust_for_horizon, mape

mutex = threading.Lock()

def load_topology(args):
    """
    Loads the network(s) topology from a file if file is passed by user
    otherwise just returns the default topology
    """
    if args.topology_file:
        topologies = []
        with open(args.topology_file) as f:
            content = f.readlines()
        for line in content:
            topology = []
            layers = re.findall('\(.+?\)', line)
            for layer in layers:
                layer_info = re.findall('{.*?}', layer)
                dict1 = ast.literal_eval(layer_info[0])
                dict2 = ast.literal_eval(layer_info[1])
                topology.append((dict1, dict2))
        topologies.append(topology)
        return topologies
    else:
        return None


def build_datasets(ts, lookback_period, test_fraction):
    """
    Build the train and test sets
    :param args:
    :param ts:
    :return:
    """
    # Format data to shape (n_steps, n_vars, n_series)
    while len(ts.shape) < 3:
        ts = np.expand_dims(ts, axis=-1)

    test_length = int(len(ts) * test_fraction)
    ts_train = ts[:-test_length]
    ts_test = ts[-test_length - lookback_period:]
    return ts_train, ts_test


def run_model(args, data_file, test_fraction, lag, topologies, epochs, batch_size, separator, horizon, lr, optimizers,
              results):
    """
    Runs the forecaster on a data set (with given parameters) and computes the metric on train and test sets also
    reports the training time
    :return:
    """
    global mutex
    print("Running for data " + data_file + " ...")
    df = read_table(data_file, sep=separator)
    train_set, test_set = build_datasets(df.values, lag, test_fraction)

    for i in range(0, len(topologies)):
        if args.verbose==1:
            print("Train\t\t model:" + topologies[i] + "\tdataset:" + data_file)
        start = time.time()
        try:
            forecaster = Forecaster(
                topologies[i],
                optimizer=optimizers[i],
                lag=lag,
                horizon=horizon,
                batch_size=batch_size,
                epochs=epochs,
                uncertainty=args.uncertainty,
                dropout_rate=args.dropout_rate,
                lr=lr,
                verbose=args.verbose
            )

            forecaster.fit(train_set, verbose=args.verbose)
            train_time = time.time() - start
            metric = adjust_for_horizon(mape)
            if args.verbose == 1:
                print("Predict\t\t model:" + topologies[i] + "\tdataset:" + data_file)
            train_err = metric(forecaster.predict(train_set, n_samples=args.n_samples)['mean'],
                               train_set[lag:len(train_set)])
            test_err = metric(forecaster.predict(test_set, n_samples=args.n_samples)['mean'],
                              test_set[lag:len(test_set)])
        except Exception as e:
            print(e)
            train_time = time.time() - start
            train_err = 'NA'
            test_err = 'NA'

        file_name = os.path.basename(data_file)
        mutex.acquire()
        results.append((file_name, topologies[i], train_err, test_err, train_time))
        mutex.release()
        print("Done running for data " + data_file + " ...")

    return train_err, test_err, train_time


def run_single_threaded(args, test_fractions, lags, topologies, epochs, batch_sizes, separators, horizons, lrs,
                        optimizers):
    results = []
    for i in range(0, len(args.data_files)):
        run_model(args, args.data_files[i], test_fractions[i], lags[i],
                  topologies, epochs[i], batch_sizes[i], separators[i],
                  horizons[i], lrs[i], optimizers, results)
    return results


def run_multi_threaded(args, test_fractions, lags, topologies, epochs, batch_sizes, separators, horizons, lrs,
                       optimizers):
    threads = []
    results = []
    num_threads = args.threads
    for i in range(0, len(args.data_files)):
        if len(threads) < num_threads:
            threads.append(threading.Thread(target=run_model, args=(
                args, args.data_files[i], test_fractions[i], lags[i], topologies, epochs[i], batch_sizes[i],
                separators[i],horizons[i], lrs[i], optimizers, results)))
        else:
            for i in range(0, len(threads)):
                threads[i].start()
            for i in range(0, len(threads)):
                threads[i].join()
            threads.clear()

    if len(threads) > 0:
        for i in range(0, len(threads)):
            threads[i].start()
        for i in range(0, len(threads)):
            threads[i].join()

    return results


def fill_list(list, target_size):
    """
    Creates a new list out of a given one and extends
    it with last element of the list
    :return: the extended list
    """
    new_list = []
    given_list_len = len(list)
    i = 1
    while i <= target_size:
        if i < given_list_len:
            new_list.append(list[i - 1])
        else:
            new_list.append(list[given_list_len - 1])
        i += 1
    return new_list


def main(args):

    print("\n\nRunning the benchmarks ...")
    topologies = load_topology(args)
    if topologies is None:
        topologies = args.network_type

    separators = fill_list(args.separator, len(args.data_files))
    lags = fill_list(args.lag, len(args.data_files))
    horizons = fill_list(args.horizon, len(args.data_files))
    test_fractions = fill_list(args.test_fraction, len(args.data_files))
    epochs = fill_list(args.epochs, len(topologies))
    batch_sizes = fill_list(args.batch_size, len(topologies))
    optimizers = fill_list(args.optimizer, len(topologies))
    lrs = fill_list(args.learning_rate, len(topologies))

    if args.multi_threaded == 1:
        results = run_multi_threaded(args, test_fractions, lags, topologies, epochs, batch_sizes, separators, horizons,
                                     lrs, optimizers)
    else:
        results = run_single_threaded(args, test_fractions, lags, topologies, epochs, batch_sizes, separators, horizons,
                                      lrs, optimizers)


    if args.print_results:
        print("#" * 100)
        print("Model\t\t\t\tTrain Metric\t\t\t\tTest Metric\t\t\t\tTrain Time\t\t\t\tDataset")
        print("#" * 100)
        for i in range(0, len(results)):
            print(results[i][1] + '\t\t\t' + str(results[i][2]) + '\t\t\t' + str(results[i][3]) + '\t\t\t' + str(
                results[i][4]) + '\t\t\t' + str(results[i][0]))
        print("#" * 100)
    return results


def _get_parser():
    """
    Collect all relevant command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-d', '--data-files',
                            help="List of data files",
                            required=True,
                            nargs="+")

    named_args.add_argument('-nt', '--network_type',
                            help="Network type",
                            required=False,
                            default=['rnn'],
                            type=str,
                            nargs="+")

    named_args.add_argument('-topology_file', '--topology-file',
                            help="File containing the networks topology (it overrides the --network_type parameter.",
                            required=False,
                            type=str)

    named_args.add_argument('-lg', '--lag',
                            help="Lookback period",
                            required=True,
                            nargs="+",
                            type=int)

    named_args.add_argument('-hr', '--horizon',
                            help="Forecasting horizon",
                            required=False,
                            default=[1],
                            nargs="+",
                            type=int)

    named_args.add_argument('-o', '--optimizer',
                            help="Optimizer type",
                            required=False,
                            default=['sgd'],
                            nargs="+",
                            type=str)

    named_args.add_argument('-sep', '--separator',
                            help="Location of data sets",
                            required=False,
                            default=[','],
                            nargs="+")

    named_args.add_argument('-tf', '--test-fraction',
                            help="Test fraction at end of dataset",
                            required=False,
                            default=[0.2],
                            nargs="+",
                            type=float)

    named_args.add_argument('-e', '--epochs',
                            help="Number of epochs to run",
                            required=False,
                            default=[100],
                            nargs="+",
                            type=int)

    named_args.add_argument('-b', '--batch-size',
                            help="Location of validation data",
                            required=False,
                            default=[8],
                            nargs="+",
                            type=int)

    named_args.add_argument('-lr', '--learning-rate',
                            help="Learning rate",
                            required=False,
                            default=[0.1],
                            nargs="+",
                            type=float)

    named_args.add_argument('-u', '--uncertainty',
                            help="Toggle uncertainty",
                            required=False,
                            default=False,
                            type=bool)

    named_args.add_argument('-dr', '--dropout_rate',
                            help="Dropout rate",
                            required=False,
                            default=0.1,
                            type=float)

    named_args.add_argument('-s', '--n_samples',
                            help="Number of dropout samples",
                            required=False,
                            default=10,
                            type=int)

    named_args.add_argument('-m', '--multi-threaded',
                            help="Multi-Threaded execution",
                            required=False,
                            default=0,
                            type=int)

    named_args.add_argument('-threads', '--threads',
                            help="Number of threads to parallelize the computation",
                            required=False,
                            default=3,
                            type=int)

    named_args.add_argument('-p', '--print_results',
                            help="Print results in tabular form",
                            required=False,
                            default=0,
                            type=int)

    named_args.add_argument('-v', '--verbose',
                            help="Verbose",
                            required=False,
                            default=0,
                            type=int)
    return parser


if __name__ == '__main__':
    args = _get_parser().parse_args()
    main(args)

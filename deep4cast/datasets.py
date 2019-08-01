from fastparquet import ParquetFile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from deep4cast import transforms


class TimeSeriesDataset(Dataset):
    """Provides windowed subseries for training. Each time series is split into
    lookback and horizon examples, with the number of examples in each
    calculated using ``lookback``, ``horizon`` and ``step``. Requires either
        * a list of ``numpy`` arrays or 
        * a path to parquet files and a CSV containing partition index and partition length.

    Arguments:
        * lookback (int): Number of time steps used as input for forecasting.
        * horizon (int): Number of time steps to forecast.
        * step (int): Time step size between consecutive examples.
        * transform (``transforms.Compose``): List of transformations to apply to time series examples.
        * thinning (float): Fraction of examples to include.
        * split (str): Optional, specifies ``train`` or ``test`` split.
        * time_series (list): List of time series ``numpy`` arrays.
        * path_parquet (str): File location of partitioned parquet files.
        * path_metadata (list): List of CSV file locations containing time series id and length.

    """

    def __init__(self,
                 lookback,
                 horizon,
                 step,
                 transform,
                 thinning=1.0,
                 split=None,
                 time_series=None,
                 path_parquet=None,
                 path_metadata=None):
        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        self.transform = transform
        self.split = split
        self.path_parquet = path_parquet
        self.time_series = time_series

        if path_parquet:
            self._examples_parquet(path_metadata=path_metadata)
        elif time_series:
            self._examples_array()

        # Store the number of training examples
        self._len = int(len(self.example_ids) * thinning)
    
    def _examples_parquet(self, path_metadata):
        """Takes a file location of metadata about the length of each time series
        and calculates number of examples in each.

        Arguments:
            * path_metadata (list): List of CSV files containing time series id and length.

        """
        path_file = ParquetFile(self.path_parquet)
        self.partitions = path_file.info['partitions'][0]

        # Slice each time series into examples, assigning IDs to each
        example_ids = {}
        n_dropped = 0
        for file_meta in path_metadata:
            with open(file_meta) as infile:
                for line in infile:
                    line = line.strip('\n')
                    line = line.split(',')
                    index = line[0] # Parition name
                    length = int(line[1]) # Length of time series
                    # Withhold the horizon for testing
                    if self.split is 'train':
                        length -= self.horizon
                    # At least the horizon is required for zero-padding
                    if length < self.horizon:
                        n_dropped += 1
                        continue
                    # Slice each time series into examples, assigning IDs to each
                    example_ids = self._hashmap(
                        index=index,
                        length=length,
                        example_ids=example_ids)

        # Inform user about time series that were too short
        if n_dropped > 0:
            print('Dropped {} time series due to length.'.format(n_dropped))

        self.example_ids = example_ids

    def _examples_array(self):
        """Takes a list of time series and calculates number of examples in each.

        """
        n_dropped = 0
        example_ids = {}
        for i, ts in enumerate(self.time_series):
            # Time series shorter than the forecast horizon need to be dropped.
            if ts.shape[-1] < self.horizon:
                n_dropped += 1
                continue
            # Slice each time series into examples, assigning IDs to each
            example_ids = self._hashmap(
                index=i,
                length=ts.shape[-1],
                example_ids=example_ids)

        # Inform user about time series that were too short
        if n_dropped > 0:
            print('Dropped {} time series due to length.'.format(n_dropped))
        
        self.example_ids = example_ids

    def _hashmap(self, index, length, example_ids):
        """Creates a dictionary of windowed examples indexed on the time series
        and location within the time series.

        Arguments:
            * index: Either list index or parquet parition name.
            * length (int): Length of the indexed time series.
            * example_ids (dict): Dictionary where the key is the example 
            number and the value is the tuple of 
            (time series index, start position for example slice).

        """
        last_id = len(example_ids)
        
        # only use last lookback + horizon for test case
        if self.split is 'test':
            length = length - self.lookback - self.horizon
            length = max((length, 0))
            example_ids[last_id] = (index, length)
            
            return example_ids

        num_examples = (length - self.lookback -
                        self.horizon + self.step) // self.step
        # For short time series we will zero pad the input
        num_examples = max((num_examples, 1))
        # (time series index, start position for example slice)
        for j in range(num_examples):
            example_ids[last_id + j] = (index, j * self.step)
        
        return example_ids

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        ts_id, lookback_id = self.example_ids[idx]

        if self.path_parquet:
            path_file = self.path_parquet + self.partitions + '=' + ts_id + '/part.0.parquet'

            ts = ParquetFile(path_file)
            ts = ts.to_pandas()
            ts = ts.values.T
        elif self.time_series:
            ts = self.time_series[ts_id]

        # Prepare input and target. Zero pad if necessary.
        if ts.shape[-1] < self.lookback + self.horizon:
            # If the time series is too short, we zero pad
            X = ts[:, :-self.horizon]
            X = np.pad(
                X,
                pad_width=((0, 0), (self.lookback - X.shape[-1], 0)),
                mode='constant',
                constant_values=0
            )
            y = ts[:, -self.horizon:]
        else:
            X = ts[:, lookback_id:lookback_id + self.lookback]
            y = ts[:, lookback_id + self.lookback:lookback_id +
                   self.lookback + self.horizon]

        # Create the input and output for the sample
        sample = {'X': X, 'y': y}
        sample = self.transform(sample)

        return sample

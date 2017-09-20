import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Given the window size w, pre-compute X(X^T X)^{-1}X^T - I,
# in which X = [0   1
#               1   1
#               2   1
#               ...
#             (w-1) 1]
def _compute_coef_matrix(w):
    from numpy import array, arange, ones, linalg, eye
    X = array([arange(w), ones([w])]).transpose()
    return X @ linalg.inv(X.transpose() @ X) @ X.transpose() - eye(w)


# TODO ktran: improve the algorithm
def _partition_anomalies(windows, k):
    """
    :param windows: windows, sorted by anomaly score in descending order
    :param k: number of partitions
    :return: partition positions
    """
    diffs = [windows[iw - 1][1] - windows[iw][1]
             for iw in range(1, len(windows))]
    top_jump_positions = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[0:k]
    return sorted(top_jump_positions)


def detect_anomalies(data, window_size, sensitivity=5, num_anomalies=None, visualize=True):
    if type(data) != pd.Series:
        raise ValueError('data must be of the pandas Series type')
    data = data.fillna(method='pad')  # fill NANs with 0 to make the series contiguous

    w = window_size
    coefs = _compute_coef_matrix(w)

    values = data.values.reshape(len(data), 1)
    num_windows = len(values) - w + 1
    windows = np.vstack(values[ix:ix + num_windows] for ix in range(window_size))
    residuals = np.linalg.norm(coefs @ windows, axis=0)

    windows = [(ix, residuals[ix]) for ix in range(num_windows)]
    windows.sort(key=lambda item: item[1],
                 reverse=True)  # sort the windows by their residuals in descending order

    if num_anomalies == 0 or sensitivity == 0:
        max_anomaly_score = windows[0][1]
        if visualize:
            print(
                'The maximum anomaly score in the training data is {0:2f}. '
                'Since you specified no anomaly observed in the historical data, the recommended threshold is {1:2f}'
                    .format(max_anomaly_score, max_anomaly_score * 2))
        return [max_anomaly_score * 2]

    # Filter out overlapping windows
    num_results = num_anomalies if num_anomalies else num_windows
    iw = 0
    top_iws = [iw]  # positions of the top windows, after filtering
    while len(top_iws) < num_results:
        while iw < num_windows and any(abs(windows[jw][0] - windows[iw][0]) < w for jw in top_iws):
            iw += 1
        if iw < num_windows:
            top_iws.append(iw)
            iw += 1
        else:
            break
    results = [windows[iw] for iw in top_iws]

    partition_points = _partition_anomalies(results, sensitivity)
    thresholds = [(results[jump][1] + results[jump + 1][1]) / 2 for jump in partition_points]

    timestamps = data.index
    anomalies = []
    rank = 0
    for level, limit in enumerate(partition_points):
        while rank <= limit:
            iw = results[rank][0]
            anomalies.append((sensitivity - level, str(timestamps[iw]), str(timestamps[iw + w - 1]), results[rank][1]))
            rank += 1
    anomalies = pd.DataFrame(anomalies, columns=['level', 'start', 'end', 'score'])

    if visualize:
        from IPython.display import display
        display(anomalies)

        data.plot(title='window size: {0}, sensitivity: {1}, #known incidents: {2}'
                  .format(window_size, sensitivity, num_anomalies))
        for anomaly in anomalies.values:
            plt.axvspan(anomaly[1], anomaly[2], color=plt.cm.jet(0.65 + float(anomaly[0]) / sensitivity / 3), alpha=0.5)

    return anomalies, thresholds


# @Lech: please explain this function
def anomalies_to_series(anomalies, index):
    rows = anomalies.shape[0]
    series = pd.Series(np.zeros(len(index)), dtype=np.int)
    series.index = index
    for r in range(rows):
        start = anomalies.loc[r, 'start']
        end = anomalies.loc[r, 'end']
        level = int(anomalies.loc[r, 'level'])
        series[start:end] = level
    return series


class StreamingAnomalyDetector:
    def __init__(self, window_size, thresholds):
        # This is prototype code and doesn't validate arguments
        self._w = window_size
        self._thresholds = thresholds
        self._buffer = np.array([float('nan')] * window_size)
        self._buffer.shape = (window_size, 1)  # make it vertical
        self._coef_matrix = _compute_coef_matrix(window_size)

    # Update thresholds on demand without restarting the service
    def update_thresholds(self, thresholds):
        self._thresholds = thresholds

    def score(self, value):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = value
        return np.linalg.norm(self._coef_matrix @ self._buffer)

    def classify(self, value):
        return bisect.bisect_left(self._thresholds, self.score(self, value))

import math, matplotlib, bisect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Given the window size, pre-compute the linear regression coefficients
def _ComputeCoefs(w):
    a11 = w * (w - 1) * (2 * w - 1) / 6
    a12 = w * (w - 1) / 2
    a22 = w
    invariant = 1 / (a11 * a22 - a12 * a12)
    return (a11, a12, a22, invariant)


#Given a window, compute the anomaly score as well as the trend
def _ComputeScore(coefs, values):
    b1 = b2 = 0
    for j, y in enumerate(values):
        b1 += j * y
        b2 += y
    slope = (coefs[2] * b1 - coefs[1] * b2) * coefs[3]
    intercept = (b1 - coefs[0] * slope) / coefs[1]

    residual = 0
    for j, y in enumerate(values):
        deviation = slope * j + intercept - y
        residual += deviation * deviation
    return math.sqrt(residual / len(values)), slope


def DetectAnomalies(data, window_size, num_levels=10, num_anomalies=None, visualize=True):
    if type(data) != pd.Series:
        raise ValueError('data must be of the pandas Series type')
    data = data.fillna(0) #fill NANs with 0 to make the series contiguous

    w = window_size
    coefs = _ComputeCoefs(w)

    values = data.values
    num_windows = len(values) - w + 1
    windows = [(iw, _ComputeScore(coefs, values[iw:iw + w])[0])
               for iw in range(num_windows)]
    
    [None] * (len(values) - w + 1)
    for iw, _ in enumerate(windows):
        windows[iw] = (iw, _ComputeScore(coefs, values[iw:iw + w])[0])

    #Sort windows by score in descending order
    windows.sort(key=lambda item: item[1], reverse=True)

    if num_anomalies==0 or num_levels==0:
        max_anomaly_score = windows[0][1]
        if visualize:
            print('The maximum anomaly score in the training data is {0:2f}.'.format(max_anomaly_score)
                + 'Since you specified no anomaly observed in the historical data, '
                + 'the recommended threshold is {0:2f}'.format(max_anomaly_score * 2))
        return [max_anomaly_score * 2]

    #Filter out overlapping windows
    num_results = num_anomalies if num_anomalies else num_windows
    iw = 0
    top_score = windows[iw][1]
    top_iws = [iw]  # positions of the top windows, after filtering
    diffs = [1e-6]  # differences between adjacent ranked windows
    while len(top_iws) < num_results:
        while iw < num_windows and any(abs(windows[jw][0] - windows[iw][0]) < w for jw in top_iws):
            iw += 1
        if iw < num_windows:
            top_iws.append(iw)
            score = windows[iw][1]
            diffs.append(top_score - score)
            top_score = score
            iw += 1
        else:
            break
    results = [windows[iw] for iw in top_iws]

    #Automatically compute the thresholds
    #REVIEW ktran: need better names and explanation
    top_jumps = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[0:num_levels]
    top_jumps.sort() #after figuring out the top jumps, reorder them by the anomaly index
    thresholds = [(results[jump-1][1] + results[jump][1]) / 2 for jump in top_jumps]

    timestamps = data.index
    anomalies = []
    rank = 0
    for level, rank_end in enumerate(top_jumps):
        while rank < rank_end:
            iw = results[rank][0]
            anomalies.append((num_levels - level, str(timestamps[iw]), str(timestamps[iw + w - 1]), results[rank][1]))
            rank += 1
    anomalies = pd.DataFrame(anomalies, columns = ['level', 'start', 'end', 'score'])

    if visualize:
        from IPython.display import display
        display(anomalies)

        data.plot(title='window size: {0}, #severity levels: {1}, #known incidents: {2}'
                  .format(window_size, num_levels, num_anomalies))
        for anomaly in anomalies.values:
            plt.axvspan(anomaly[1], anomaly[2], color=plt.cm.jet(0.5 + float(anomaly[0])/num_levels/2), alpha=0.5)
        
    return anomalies, thresholds


#@Lech: please explain this function
def AnomaliesToSeries(anomalies, index):
    rows = anomalies.shape[0]
    series = pd.Series(np.zeros(len(index)), dtype=np.int32)
    series.index = index
    for r in range(rows):
        start = anomalies.loc[r, 'start']
        end = anomalies.loc[r, 'end']
        level = int(anomalies.loc[r, 'level'])
        series[start:end] = level
    return series


class StreamingAnomalyDetector:
    def __init__(self, windowSize, thresholds):
        # This is prototype code and doesn't validate arguments
        self._w = windowSize;
        self._thresholds = thresholds;
        self._buffer = [float('nan')] * windowSize;
        self._coefs = _ComputeCoefs(windowSize)

    # Update thresholds on demand without restarting the service
    def UpdateThesholds(self, thresholds):
        self._thresholds = thresholds;

    def Score(self, value):
        #REVIEW: in Python, how to shift an array in place without allocating another array?
        w1 = self._w - 1;
        self._buffer[0:w1] = self._buffer[-w1:]
        self._buffer[-1] = value;
        return _ComputeScore(self._coefs, self._buffer)

    def Classify(self, value):
        (score, trend) = self.Score(self, value)
        return bisect.bisect_left(self._thresholds, score)

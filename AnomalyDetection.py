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

def DetectAnomalies(data, windowSize, levels=1, numTopResults=None, visualize=False):

    if type(data) != pd.Series:
        raise ValueError('data must be of the pandas Series type')

    # TODO, is it possible not to spread the visualize clause around?
    if visualize:
        data.plot()

    w = windowSize
    coefs = _ComputeCoefs(w)

    values = data.fillna(0).values  #fill NANs with 0 to make the series contiguous
    numWindows = len(values) - w + 1
    windows = [None] * numWindows
    for pos in range(numWindows):
        windows[pos] = (pos, _ComputeScore(coefs, values[pos:pos + w])[0])

    #Sort windows by score in descending order
    windows.sort(key=lambda item: item[1], reverse=True)

    autoThreshold = numTopResults is None
    if autoThreshold:
        numTopResults = numWindows

    #Filtering overlapping windows
    topIds = [0]  #positions of the top windows, after filtering
    diffs = [1e-6]  #differences between adjacent ranked windows
    curId = 0
    curTop = windows[0][1];
    while len(topIds) < numTopResults:
        while curId < numWindows and any(abs(windows[pos][0] - windows[curId][0]) < w for pos in topIds):
            curId += 1
        if curId < numWindows:
            topIds.append(curId)
            cur = windows[curId][1]
            diffs.append(curTop - cur)
            curTop = cur
            curId += 1
        else:
            break
    results = [windows[pos] for pos in topIds]

    if levels == 0:
        max_anom_score = results[0][1]
        if visualize:
            print('The maximum anomaly score in the training data is {0:2f}.'.format(max_anom_score)
                + 'Since you specified no anomaly in the historical data, '
                + 'the recommended threshold is {0:2f}'.format(max_anom_score * 2))
        return [max_anom_score * 2]

    #Automatically compute the thresholds
    topJumps = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[0:levels]
    topJumps.sort() #after figuring out the top jumps, reorder them by the anomaly index
    thresholds = [(results[jump-1][1] + results[jump][1]) / 2 for jump in topJumps]

    #Visualize the outputs
    timestamps = data.index
    if visualize:
        print('{0: <45}Anomaly Score'.format('Time Interval'))
    low = results[min(topJumps[-1], len(results) - 1)][1]
    hi = results[0][1]
    norm = matplotlib.colors.Normalize(2 * low - hi, hi)
    curId = 0
    levs = []
    starts = []
    ends = []
    scores = []
    for level, jump in enumerate(topJumps):
        for pos in range(curId, jump):
            idx = results[pos][0]
            start = str(timestamps[idx])
            end = str(timestamps[idx + w - 1])
            score = results[pos][1]
            levs.append(levels-level)
            starts.append(start)
            ends.append(end)
            scores.append(score)
            if visualize:
                print('{0: <45}{1:G}'.format(start + ' - ' + end, score))
                plt.axvspan(start, end, color=plt.cm.jet(norm(score)), alpha=0.5);
            curId += 1
        if visualize:
            print('--------------- Threshold level {0}: {1:G} ---------------'.format(
                levels - level, thresholds[level]))

        anomalies = pd.DataFrame(np.column_stack([levs, starts, ends, scores]))
        anomalies.columns = ['level', 'start', 'end', 'score']

    return anomalies, thresholds

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

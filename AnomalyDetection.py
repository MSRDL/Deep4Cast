import math, matplotlib
from pandas import read_csv, DataFrame, Series
from matplotlib import pyplot as plt
from bokeh import plotting

plotting.output_notebook(force=True)


def DetectAnomalies(data, windowSize, depth=1, numTopResults=None, col=0):
    dtype = type(data)
    if not (dtype is DataFrame or dtype is Series):
        print('Data must be of type Series or DataFrame')
        return

    if data.shape[1] != 1:
        print(
            "This anomaly detector only works with one-dimensional time series. If your data has multile signals, it's " \
            + "recommended that you split your data into multiple files, and apply a detector for each signal.")
        return

    matplotlib.rcParams['figure.figsize'] = [16, 3]
    data.plot()

    w = windowSize # number of data points
    a11 = w * (w - 1) * (2 * w - 1) / 6
    a12 = w * (w - 1) / 2
    a22 = w
    invariant = 1 / (a11 * a22 - a12 * a12)

    values = data.icol(col)
    numWindows = len(values) - w + 1
    windows = [None] * numWindows
    for pos in range(numWindows):
        b1 = b2 = 0
        for j, y in enumerate(values[pos:pos + w]):
            b1 += j * y
            b2 += y
        slope = (a22 * b1 - a12 * b2) * invariant
        intercept = (b1 - a11 * slope) / a12

        # Compute the residual
        residual = 0
        for j, y in enumerate(values[pos:pos + w]):
            deviation = slope * j + intercept - y
            residual += deviation * deviation

        windows[pos] = (pos, math.sqrt(residual / w))

    # Sort windows by score in descending order
    windows.sort(key=lambda item: item[1], reverse=True)

    autoThreshold = numTopResults is None
    if autoThreshold:
        numTopResults = numWindows

    # Filtering overlapping windows
    cur = 0
    tops = []
    while len(tops) < numTopResults:
        while cur < numWindows and any(abs(windows[pos][0] - windows[cur][0]) < w for pos in tops):
            cur += 1
        if cur < numWindows:
            tops.append(cur)
            cur += 1
        else:
            break
    results = [windows[pos] for pos in tops]

    cut = 0
    for d in range(depth):
        diffMax = 0
        for pos in range(cut + 1, len(tops)):
            diff = results[pos - 1][1] - results[pos][1]
            if diff > diffMax:
                cut = pos
                diffMax = diff

    timestamps = data.index
    print('{0: <45}Anomaly Score'.format('Time Interval'))
    low = results[min(cut, len(results) - 1)][1]
    hi = results[0][1]
    norm = matplotlib.colors.Normalize(2 * low - hi, hi)
    for pos in range(cut):
        idx = results[pos][0]
        start = str(timestamps[idx])
        end = str(timestamps[idx + w - 1])
        score = results[pos][1]
        print('{0: <45}{1:G}'.format(start + ' - ' + end, score))
        plt.axvspan(start, end, color=plt.cm.jet(norm(score)), alpha=0.5);

    if autoThreshold:
        almostAnom = results[cut]
        almostAnomScore = almostAnom[1]
        print('\n---------------- Auto Threshold: {0:G} ----------------\n'.format(
            (results[cut - 1][1] + almostAnomScore) / 2))
        idx = almostAnom[0]
        print('{0: <45}{1:G}'.format(str(timestamps[idx]) + ' - ' + str(timestamps[idx + w]), almostAnomScore))


def DetectAnomaliesFromFile(filename, windowSize, separator=',', depth=1, numTopResults=None):
    data = read_csv(filename, parse_dates=0, index_col=0)
    return DetectAnomalies(data, windowSize, depth=depth, numTopResults=numTopResults)

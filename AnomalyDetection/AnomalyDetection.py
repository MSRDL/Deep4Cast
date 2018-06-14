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
    top_jump_positions = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[0:k-1]
    return sorted(top_jump_positions) + [len(windows) - 1]

def label_anomaly_windows(label):
    """
    :param label: input labels 1/0
    :return: set of intervals of anomaly (start time, end time)
    """
    labels = label.values
    timestamps = label.index
    labeled_anomaly_window = []
    for i in range(1, len(label)):
        if labels[i] == 1 and labels[i-1] == 0:
            start = timestamps[i]
        elif labels[i-1] == 1 and labels[i] == 0:
            end = timestamps[i-1]
            labeled_anomaly_window.append((start, end))
  
    return labeled_anomaly_window

def calculate_IOU(anomalies, label):
    """
    :param anomalies: anomalies windows generated by the algorithm
    :param label: actual labels
    :return iou: intersection over union if the anomaly windows overlap with actual labels else 0
    :return anomaly_region: list of labels associated with each anomaly window since there could be multiple
    """
    label_window = label_anomaly_windows(label)
    from datetime import datetime
    iou = []
    anomaly_region = []
    def stt(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    def t_max(t1, t2): return t1 if t1 > t2 else t2
    def t_min(t1, t2): return t1 if t1 < t2 else t2
        
    
    for i in range(len(anomalies)):
        iou_i = 0
        region = []
        start = stt(anomalies.start[i])
        end = stt(anomalies.end[i])
        for j in range(len(label_window)):

            if start <= label_window[j][0] and end >= label_window[j][0]:
                overlap = 1 + (t_min(label_window[j][1], end) - label_window[j][0]).total_seconds()/60 
                union   = 1 + (t_max(label_window[j][1], end)- start).total_seconds()/60 
                iou_i += (float(overlap)/union)
                region.append(j)

            elif start >= label_window[j][0] and end <= label_window[j][1]:
                overlap = 1 + (end - start).total_seconds()/60 
                union   = 1 + (label_window[j][1]- label_window[j][0]).total_seconds()/60 
                iou_i += (float(overlap)/union)
                region.append(j)
                
            elif start <= label_window[j][1] and end >= label_window[j][1]:
                overlap = 1 + (label_window[j][1] - start).total_seconds()/60 
                union   = 1 + (end - label_window[j][0]).total_seconds()/60 
                iou_i += (float(overlap)/union)
                region.append(j)
                
        anomaly_region.append(region)
        iou.append(iou_i)  
    
    return iou, anomaly_region

def AP_score(anomalies, label, iou_threshold):
    """
    :param anomalies: anomaly windows generated by the algorithm
    :param label: actual labels
    :param iou_threshold: threshold above which regions are considered correct detection
    :return : AP_score (average precision)
    """
    labeled_data = label_anomaly_windows(label)
    iou, anomaly_region = calculate_IOU(anomalies, label)
    
    precision = []
    recall = []
    for i in range(1, len(iou)):
        iou_i = iou[:i]
        region = []
        tp, fp, fn = 0,0,0
        for j in range(len(iou_i)):
            if iou_i[j] > iou_threshold:
                tp += 1
                for window in anomaly_region[j]:
                    if window not in region:
                        region.append(window)
        fp = len(iou_i) - tp
                
        precision.append(float(tp)/ (tp + fp))
        recall.append(len(region)/ len(labeled_data))
    
    
    recall_interp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    recall_interp = recall_interp[::-1]
    recall = recall[::-1]
    precision = precision[::-1]
    precision_interp = []
    for val in recall_interp:
        k = 0
        p = 0
        while k < len(recall) and recall[k] >= val :
            p = precision[k] if precision[k] >= p else p
            k += 1
            
        precision_interp.append(p)
    return np.mean(precision_interp)

def mean_average_precision(data, lag, algorithm):
    """
    :param data: time series data
    :param lag: lag time
    :return: mean of average precision over various intersection over union thresholds 
    """
    label = data['Label']
    iou_thresholds = [0.0, 0.1, 0.2, 0.3]
    mean = 0
    anomalies, thresholds = detect_anomalies(data, lag, num_anomalies=len(data)/lag, visualize=False, algorithm=algorithm)
    for i in range(len(iou_thresholds)):
        mean += AP_score(anomalies, label, iou_thresholds[i])
    return float(mean)/ len(iou_thresholds)

def detect_anomalies(data, lag, num_anomalies, num_levels=5, visualize=True, algorithm='sls'):
    #separate labels
    labels = data['Label']
    data = data['Value']
    if type(data) != pd.Series:
        raise ValueError('data must be of the pandas Series type')
    if lag < 3:
        raise ValueError('lag needs to be at least 3.')
    if num_anomalies < 0:
        raise ValueError('expected number of anomalies must be positive.')
    num_levels = min(num_levels, num_anomalies) 

    data = data.fillna(method='pad')  # fill NANs with 0 to make the series contiguous

    coefs = _compute_coef_matrix(lag)
    #Update coefs if moving average is specified
    if algorithm == 'avg':
        coefs = np.ones((1, lag))/float(lag)

    values = data.values
    num_windows = len(values) - lag + 1
    windows = np.vstack(values[ix:ix + num_windows] for ix in range(lag))
    residuals = np.linalg.norm(coefs @ windows, axis=0)
    windows = [(ix, residuals[ix]) for ix in range(num_windows)]
    windows.sort(key=lambda item: item[1],
                 reverse=True)  # sort the windows by their residuals in descending order

    if num_anomalies == 0 or num_levels == 0:
        max_anomaly_score = windows[0][1]
        if visualize:
            print(
                'The maximum anomaly score in the training data is {0:2f}. '
                'Since you specified no anomaly observed in the historical data, the recommended threshold is {1:2f}'
                    .format(max_anomaly_score, max_anomaly_score * 2))
        return None, [max_anomaly_score * 2]

    # Filter out overlapping windows
    iw = 0
    top_iws = [iw]  # positions of the top windows, after filtering
    while len(top_iws) < num_anomalies:
        while iw < num_windows and any(abs(windows[jw][0] - windows[iw][0]) < lag for jw in top_iws):
            iw += 1
        if iw < num_windows:
            top_iws.append(iw)
            iw += 1
        else:
            break
    results = [windows[iw] for iw in top_iws]

    partition_points = _partition_anomalies(results, num_levels)
    thresholds = [results[iw][1] - 1e-3 for iw in partition_points]

    timestamps = data.index
    anomalies = []
    rank = 0
    for level, limit in enumerate(partition_points):
        while rank <= limit:
            iw = results[rank][0]
            anomalies.append((num_levels - level, str(timestamps[iw]), str(timestamps[iw + lag - 1]), results[rank][1]))
            rank += 1
    anomalies = pd.DataFrame(anomalies, columns=['level', 'start', 'end', 'score'])
    labeled_windows = label_anomaly_windows(labels)
    
    if visualize:
        from IPython.display import display
        display(anomalies)
        plt.subplot(211)
        data.plot(title='lag: {0}, #levels: {1}, #anomalies: {2}'
                  .format(lag, num_levels, num_anomalies))
        for anomaly in anomalies.values:
            plt.axvspan(anomaly[1], anomaly[2], color=plt.cm.jet(0.65 + float(anomaly[0]) / num_levels / 3), alpha=0.5)
        plt.subplot(212)
        data.plot(title='Labeled Data')
        for label in labeled_windows:
            plt.axvspan(label[0], label[1], color='c', alpha=0.5)
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
    def __init__(self, lag, thresholds):
        # This is prototype code and doesn't validate arguments
        self._w = lag
        self._thresholds = thresholds
        self._buffer = np.array([float('nan')] * lag)
        self._buffer.shape = (lag, 1)  # make it vertical
        self._coef_matrix = _compute_coef_matrix(lag)

    # Update thresholds on demand without restarting the service
    def update_thresholds(self, thresholds):
        self._thresholds = thresholds

    def score(self, value):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = value
        return np.linalg.norm(self._coef_matrix @ self._buffer)

    def classify(self, value):
        return bisect.bisect_left(self._thresholds, self.score(self, value))


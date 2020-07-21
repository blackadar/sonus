"""
Code to process data in a Pipeline.
"""

import sys
import pandas as pd
import numpy as np
import sklearn
import scipy
import statistics as stat
import math
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def generate_pipeline(model):
    return Pipeline([("Scaler", StandardScaler()), ("Zero Var Remover", VarianceThreshold()), ("Model", model)])


def add_stat_features(data: np.ndarray):
    """
    based on the data given, creates features such as min, max, mean, std, rms, percentiles; 0, 25, 50, and 75
    :param data: the data used to create statistics/features from
    :return: concatenation of input data and the new features
    """
    NUM_STATS = 9

    result = np.zeros(shape=(len(data), NUM_STATS))

    for i in range(len(data)):
        result[i][0] = data[i].max()
        result[i][1] = data[i].min()
        result[i][2] = data[i].mean()
        result[i][3] = stat.stdev(data[i])
        result[i][4] = math.sqrt(sum(data[i] ** 2) / len(data[i]))
        result[i][5] = np.percentile(data[i], 0)
        result[i][6] = np.percentile(data[i], 25)
        result[i][7] = np.percentile(data[i], 50)
        result[i][8] = np.percentile(data[i], 75)
    return result


def fft(data: np.ndarray, low_pass=20, high_pass=20000, return_x=False):
    """
    Adds a frequency space feature to the data. The data should be a windowed audio array.
    The output of this algorithm is appended columns at the end.
    https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
    https://www.youtube.com/watch?v=17cOaqrwXlo
    https://www.youtube.com/watch?v=aQKX3mrDFoY  << Especially helpful
    https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python
    :param return_x: Return the FFT x-axis.
    :param high_pass: Where to stop reporting, at the higher end of the FFT
    :param low_pass: Where to start reporting the low end of the FFT
    :param data: np.ndarray Windowed audio data for training
    :return: np.ndarray with appended columns, extra length of high_pass-low_pass. (or tuple x_axis, np.ndarray)
    """

    assert high_pass > low_pass, f"Invalid FFT window {low_pass} to {high_pass}"

    n_samples = data.shape[1]
    fft_ys = []

    fft_x = scipy.fft.fftfreq(n_samples, (1.0 / 16000))  # 16000 is the sample rate
    fft_x = fft_x[0:(n_samples // 2)]  # Take the first half of the x-axis too
    pass_mask = [True if low_pass <= x <= high_pass else False for x in fft_x]
    fft_x = fft_x[pass_mask]

    for i, row in enumerate(data):
        sys.stdout.write(f"\r[-] FFT: {i} of {len(data)} ({i / len(data) * 100: .2f}%)")
        sys.stdout.flush()
        n_samples = len(row)
        fft_y = scipy.fft(row)
        fft_y = np.abs(fft_y[0:(n_samples // 2)])  # Take real components of first half
        # fft_y = np.multiply(fft_y, 2) / (32767 * (n_samples // 2))  # Rescale, 16-bit PCM
        fft_y = np.divide(np.multiply(fft_y, 2), n_samples)
        fft_y = fft_y[pass_mask]  # Mask out unwanted values
        fft_ys.append(fft_y)
    sys.stdout.write(f"\r[-] FFT: Completed {len(data)} (100%)")
    sys.stdout.flush()

    fft_ys = np.array(fft_ys)
    # result = np.concatenate((data, fft_ys), axis=1)

    if not return_x:
        return fft_ys
    else:
        return fft_x, fft_ys


def column_join(*data):
    """
    Takes any number of np.ndarray to concatenate on the column dimension.
    This can combine data from multiple sources to be used with an ML model.
    :param data: np.ndarrays with equal numbers of rows.
    :return: One single np.ndarray, concatenated on columns.
    """
    assert len(data) > 1, f"Need at least 2 arrays to join. Got {len(data)}."
    assert type(data[0]) is np.ndarray
    expected_rows = data[0].shape[0]

    result = data[0]

    for item in data[1:]:
        assert type(item) is np.ndarray
        assert item.shape[0] == expected_rows, f"Rows must be equal length. Expected {expected_rows}, " \
                                               f"got {item.shape[0]}."

        result = np.concatenate((result, item), axis=1)

    return result

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
    result = np.concatenate((data, fft_ys), axis=1)

    if not return_x:
        return result
    else:
        return fft_x, result


def fft_df(data: pd.DataFrame):
    """
    Adds a frequency space feature to the data. The audio column should be in wav format, np.ndarrays.
    https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
    https://www.youtube.com/watch?v=17cOaqrwXlo
    https://www.youtube.com/watch?v=aQKX3mrDFoY  << Especially helpful
    https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python
    :param data: pd.DataFrame input DataFrame with ['audio'] column to transform
    :return: pd.DataFrame with new columns ['fft_x'], ['fft_y']
    """

    assert 'audio' in data.keys(), "'audio' must be present in keys."
    assert len(data) > 0, "data does not contain any rows."
    assert type(data.iloc[0]['audio']) is np.ndarray, "'audio' must contain numpy arrays."

    fft_xs = []
    fft_ys = []

    for i, row in data.iterrows():
        sys.stdout.write(f"\r[-] FFT: {i} of {len(data)} ({i / len(data) * 100: .2f}%)")
        sys.stdout.flush()
        n_samples = len(row['audio'])
        fft_y = scipy.fft(row['audio'])
        fft_x = scipy.fft.fftfreq(len(fft_y), (1.0 / row['samplerate']))
        fft_y = np.abs(fft_y[0:(n_samples // 2)])  # Take real components of first half
        # fft_y = np.multiply(fft_y, 2) / (32767 * (n_samples // 2))  # Rescale, 16-bit PCM
        fft_y = np.divide(np.multiply(fft_y, 2), n_samples)
        fft_x = fft_x[0:(n_samples // 2)]  # Take the first half of the x-axis too
        fft_xs.append(fft_x)
        fft_ys.append(fft_y)
    sys.stdout.write(f"\r[-] FFT: Completed {len(data)} ({i / len(data) * 100: .2f}%)")
    sys.stdout.flush()
    data['fft_y'] = pd.Series(fft_ys)
    data['fft_x'] = pd.Series(fft_xs)

    return data

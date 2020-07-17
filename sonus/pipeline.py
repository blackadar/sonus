"""
Code to process data in a Pipeline.
"""

import sys
import pandas as pd
import numpy as np
import sklearn
import scipy


def fft(data: pd.DataFrame):
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

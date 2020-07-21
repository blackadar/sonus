"""
Demonstrates functions, and helps visualize intermediates.
"""

import matplotlib.pyplot as plt


def plot_waveform(data, idx):
    """
    Plots the audio waveform of wav-style data
    :param data: np.ndarray processed by sonus.parse.window_data
    :param idx: Index to plot
    :return: None, plot displayed
    """
    import numpy as np
    fig, ax = plt.subplots()
    plt.plot(np.arange(0, data.shape[1]), data[idx])
    plt.ylabel('Amplitude')
    plt.xlabel('Sample Index')
    plt.title(f'Sample {idx} Waveform')
    fig.show()


def plot_fft(data, fft_x, idx, fft_start=8000, fft_end=11990):
    """
    Show the frequency plot of a row.
    :param fft_x: np.ndarray X-Axis of FFT
    :param fft_end: End column index of fft
    :param fft_start: Start column index of fft
    :param data: np.ndarray processed by sonus.pipeline.fft
    :param idx: Index to plot
    :return: None, plot displayed.
    """

    fig, ax = plt.subplots()
    plt.plot(fft_x, data[idx, fft_start:fft_end])
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.title(f'Sample {idx} FFT')
    fig.show()


def plot_fft_df(data, idx):
    """
    Show the frequency plot of a row.
    :param data: pd.DataFrame processed by sonus.pipeline.fft_df
    :param idx: Index to plot
    :return: None, plot displayed.
    """

    fig, ax = plt.subplots()
    plt.plot(data.loc[idx, ('fft_x')], data.loc[idx, ('fft_y')])
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    fig.show()

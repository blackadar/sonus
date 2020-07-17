"""
Demonstrates functions, and helps visualize intermediates.
"""

import matplotlib.pyplot as plt


def plot_fft(data, idx):
    """
    Show the frequency plot of a row.
    :param data: pd.DataFrame processed by sonus.pipeline.fft
    :param idx: Index to plot
    :return: None, plot displayed.
    """

    fig, ax = plt.subplots()
    plt.plot(data.loc[idx, ('fft_x')], data.loc[idx, ('fft_y')])
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    fig.show()

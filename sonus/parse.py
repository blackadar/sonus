"""
Functions to handle I/O.
"""

import sys
import pathlib
import zipfile
import audioread
import pandas as pd
import numpy as np


def unzip(file: pathlib.Path, directory: pathlib.Path, quiet_ignore: bool = False):
    """
    Unzips a zip-file into directory. Will not unzip if directory has contents.
    :param quiet_ignore: Bool, do not raise exception if directory has contents and no extraction is performed.
    :param file: pathlib.Path location of zip file
    :param directory: pathlib.Path target location for extracted files
    :return: None
    """
    if type(directory) is not pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
    if directory.is_dir() and not any(directory.iterdir()):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(directory)
    else:
        if not quiet_ignore:
            raise EnvironmentError('Target extraction directory is not empty, no files extracted.')


def enumerate_vox(directory: pathlib.Path):
    """
    Loads the VoxCeleb Audio data paths into a DataFrame.
    :param directory: Top-Level directory where the data was unzipped.
    :return: pd.DataFrame with columns file and id, for the filepath and the target.
    """
    if type(directory) is not pathlib.Path:
        directory = pathlib.Path(directory)
    assert directory.exists(), f"Directory {directory} does not exist."
    top = directory / 'aac'
    try:
        import psutil
        import os
        filesize = os.path.getsize(top.resolve())
        virt_mem = psutil.virtual_memory()
        if filesize > virt_mem.available:
            print("[!] Directory contents exceed available memory! Loading them may cause paging.")
        if filesize * 2 > virt_mem.available:
            print("[!] Available memory seems low. Load and copy may cause paging.")
    except ImportError:
        print("[-] System statistics unavailable.")
    assert top.exists(), f"Directory {directory} does not contain the aac folder."
    all_files = []
    for i in top.rglob('*.*'):
        all_files.append((i.resolve(), i.parents[1].parts[-1]))
    columns = ["file", "id"]
    df = pd.DataFrame.from_records(all_files, columns=columns)
    return df


def load_vox(data: pd.DataFrame, in_place: bool = False):
    """
    Loads audio data into memory from filepaths in a DataFrame.
    :param in_place: Modify the parameter DataFrame
    :param data: pd.DataFrame containing 'file' column to replace with data from the file.
    :return: pd.DataFrame with data from disk.
    """
    if not in_place:
        mod = data.copy()
    else:
        mod = data
    clips = []
    for i, row in mod.iterrows():
        sys.stdout.write(f"\r[-] Reading: {i} of {len(mod)} ({i / len(mod) * 100: .2f}%)")
        sys.stdout.flush()
        with audioread.audio_open(row['file']) as f:
            data = bytearray()
            for buf in f:
                data = data + buf
            clips.append(data)
    sys.stdout.write(f"\r[ ] Read {len(mod)} files into DataFrame.\r\n")
    sys.stdout.flush()
    mod['audio'] = pd.Series(clips)
    return mod

def window_data(datax, datay, window_length, hop_size, sample_rate, test_size):
    """Returns a windowed dataset in the format X_train, X_test, y_train, y_test

    Args:
        datax (pd.Series): unwindowed data in format from pickle['audio']
        datay (pd.DataFrame): data y values that corespond to datax by row pickle['id']
        window_length (float): window length in seconds
        hop_size (float): size to move window by on each iteration
        sample_rate (int): number of samples per second in the data

    Returns:
        np.ndarray: X_train
        np.ndarray: X_test
        np.ndarray: y_train
        np.ndarray: y_test
    """
    sample_window_length = int(np.floor(window_length * sample_rate))
    sample_hop_size = int(np.floor(hop_size * sample_rate))

    X_train = np.empty((0, sample_window_length))
    X_test = np.empty((0, sample_window_length))
    y_train = np.array([])
    y_test = np.array([])

    for (index, row) in datax.items():
        sys.stdout.write(f"\r[-] Reading: {index} of {len(datax)} ({index / len(datax) * 100: .2f}%)")
        sys.stdout.flush()

        windowed_row = np.empty((0, sample_window_length))
        target_row = np.array([])

        for start_pos in np.arange(0, len(row)-sample_window_length, sample_hop_size):
            window = datax.loc[index][start_pos:start_pos + sample_window_length]

            windowed_row = np.vstack((windowed_row, window))
            target_row = np.append(target_row, datay[index])

        midpoint = int(np.floor(len(windowed_row) * (1-test_size)))

        X_train = np.vstack((X_train, windowed_row[:midpoint]))
        X_test = np.vstack((X_test, windowed_row[midpoint:]))
        y_train = np.append(y_train, target_row[:midpoint])
        y_test = np.append(y_test, target_row[midpoint:])

    return X_train, X_test, y_train, y_test

"""
Functions to handle I/O.
"""

import sys
import pathlib
import zipfile
import audioread
import pandas as pd


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

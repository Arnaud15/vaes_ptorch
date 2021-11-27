import csv
import os
from typing import Optional, Tuple

import numpy as np
import requests

DATA_FILE_NAME = "old_faithful.csv"


def read_saved(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads and returns saved old faithful data stored in csv format."""
    with open(path, "r") as data_file:
        reader = csv.reader(data_file)
        eruption_times, wait_times = [], []
        for row in reader:
            assert len(row) == 2
            eruption_times.append(float(row[0]))
            wait_times.append(float(row[1]))
        return np.asarray(eruption_times), np.asarray(wait_times)


def save_data(path: str, eruptions: np.ndarray, waits: np.ndarray):
    """Save old faithful data stored in csv format."""
    assert len(eruptions) == len(waits)
    with open(path, "w") as data_file:
        writer = csv.writer(data_file)
        for ix, er in enumerate(eruptions):
            writer.writerow([er, waits[ix]])


def get_old_faithful_data(
    save: bool, save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get old faithful data and return it
    
    If a save directory is specified, attempt to read old faithful data previously stored in this directory
    If save is True, a save directory must be specified and the retrieved data will be stored in csv format in that directory.
    """
    if save_dir is not None:
        save_path = os.path.join(save_dir, DATA_FILE_NAME)
        if os.path.exists(save_path):
            return read_saved(save_path)
    raw = request_data()
    eruption_times, wait_times = extract_data(raw)
    if save:
        assert (
            save_dir is not None
        ), "asked to save the data but did not indicate a directory where to save it"
        save_data(save_path, eruption_times, wait_times)
    return eruption_times, wait_times


def request_data() -> str:
    """Request the raw old faithful data from Larry Wasserman's CMU website."""
    return requests.get(
        "https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat",
        verify=False,
    ).text


def extract_row(raw_row: str) -> Tuple[float, float]:
    """Extract a single eruption time and wait time data point"""

    useful_data = [
        elt for ix, elt in enumerate(raw_row.split(" ")) if ix > 0 and elt
    ]  # split on whitespaces, then remove the row index and empty strings
    assert len(useful_data) == 2, (raw_row, useful_data)
    eruption_time = float(useful_data[0])
    waiting_time = float(useful_data[1])
    return eruption_time, waiting_time


def extract_data(raw: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract eruption time and wait time data from the raw text of the webpage"""
    line_contains_data = False

    eruption_times, wait_times = [], []
    for line in raw.split("\n"):
        if "eruptions waiting" in line:
            line_contains_data = True
            continue
        if line_contains_data and line:
            e_time, w_time = extract_row(line)
            eruption_times.append(e_time)
            wait_times.append(w_time)
    return np.asarray(eruption_times), np.asarray(wait_times)

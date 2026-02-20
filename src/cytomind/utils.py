from typing import Sequence, Union
from datetime import datetime, timezone
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Methods to compress and decompress binary masks using run-length encoding (RLE).

def rlencode(mask: Sequence[bool] | NDArray[np.bool_]) -> NDArray[np.int_]:
    """
    Run-length encode a 1D binary mask. This can be used to store
    masks more compactly.

    Parameters
    ----------
    mask : Sequence[bool] | numpy.ndarray
        1D sequence or numpy array of bool.

    Returns
    -------
    list of int
        RLE encoded list: [length1, length2, ...]
        Values are alternating counts of 0s and 1s and 0 always starts first.

    Raises
    ------
    ValueError
        If `mask` is not 1D.
    """

    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 1:
        raise ValueError(f"rlencode expects a 1D array, got shape {mask.shape!r}")

    # Edge case: empty mask
    if mask.size == 0:
        return np.zeros(0, dtype=np.int_)

    # Encoding
    changes = np.where(np.logical_xor(mask[1:], mask[:-1]))[0] + 1
    boundaries = np.concatenate(([0], changes, [mask.size]))
    run_lengths = np.diff(boundaries)

    # Ensure the sequence always starts with count of 0s
    if mask[0] == True:
        run_lengths = np.concatenate((np.array([0], dtype=np.int_), run_lengths))

    return run_lengths


def rldecode(rle: Union[Sequence[int], NDArray[np.int_]]) -> NDArray[np.bool_]:
    """
    Decode a run-length encoded sequence back into a 1D binary mask.
    This is the inverse of `rlencode`, can be used to reconstruct masks.

    Parameters
    ----------
    rle : sequence of int or 1D numpy.ndarray[np.int_]
        RLE encoded list or NDArray: [length1, length2, ...]
        Values are alternating counts of 0s and 1s and 0 always starts first.

    Returns
    -------
    numpy.ndarray
        1D numpy array of bool representing the decoded mask.
    """
    rle_array = np.asarray(rle, dtype=np.int_)

    # Edge case: empty RLE
    if rle_array.size == 0:
        return np.zeros(0, dtype=bool)

    if np.any(rle_array < 0):
        raise ValueError("rldecode expects non-negative run lengths.")

    # Decoding
    values = np.arange(rle_array.size) % 2  # 0 for zeros, 1 for ones
    mask = np.repeat(values, rle_array)

    return mask.astype(bool)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def spillover_df_to_string(df: pd.DataFrame, fmt: str = ".6g") -> str:

    # Validate Shape
    if df.shape[1] == df.shape[0] + 1 and df.dtypes[0] == object:
        # Case 1: first column is index (detector names) and rest are spill
        df = df.set_index(df.columns[0])

    if df.shape[0] != df.shape[1]:
        raise ValueError("Spillover matrix must be square")

    if df.shape[1] < 2:
        raise ValueError("Spillover matrix must be at least 2x2")

    # index must match columns (if index exists)
    if df.index is not None and list(df.index) != list(df.columns):
        raise ValueError("DataFrame index must match columns")

    # Extract data
    markers = list(df.columns)
    n = str(len(markers))
    # flatten row-major: S11, S12, ..., S1n, S21, ..., Snn
    values = df.to_numpy(dtype=float).ravel(order="C")

    # Validate values
    if np.isnan(values).any():
        raise ValueError("Spillover matrix contains NaN values")

    if not np.isfinite(values).all():
        raise ValueError("Spillover matrix contains non-finite values")

    if values.max() > 1 or values.min() < -1:
        warnings.warn("Spillover matrix contains values outside the range [-1, 1]")

    return ",".join((n, *markers, *map(lambda x: format(x, fmt), values)))

def string_to_filename(s: str) -> str:
    """
    Convert a string to a safe filename by replacing unsafe characters.

    Parameters
    ----------
    s : str
        Input string to convert.

    Returns
    -------
    str
        Safe filename string.
    """
    # replace certain characters with underscores
    underscore_chars = [' ', ':', '/', '|', '\\', '-', ',', ';']
    for char in underscore_chars:
        s = s.replace(char, '_')
    # collapse repeated underscores
    while "__" in s:
        s = s.replace("__", "_")
    rm_chars = ['<', '>', '"', '?', '*', "'", '[', ']', '(', ')']
    for char in rm_chars:
        s = s.replace(char, '')
    return s
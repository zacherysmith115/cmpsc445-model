import json
import numpy as np
from typing import List
from typing import Tuple
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided

config = {"window": 5,
          "train_split": 0.80
         }


def read_data() -> List[Tuple[str, List[str], NDArray]]:
    """
    Read the given source of data into memory and extract releveant params to
    form a dataset 

    Returns: 
        List of tuples in the form of (Security Symbol, Dates, Closing Price)
    """

    src = './data/data.json'

    raw_data = []
    with open(src, 'r') as f:
        for datum in json.loads(f.read()): 
            raw_data.append(datum)

    data_set = []
    for datum in raw_data: 
        sym = datum["2. Symbol"]

        series: dict = datum["6. Time Series"]

        dates = [k for k in series.keys()]
        dates.reverse()

        close_prices = [float(v) for v in series.values()]
        close_prices.reverse()

        assert len(close_prices) == len(dates)

        data_set.append((sym, dates, np.array(close_prices)))

    return data_set


def normalize(x: NDArray) -> NDArray:
    """
    Function to normalize the data to have mean=0 and std_dev=1
    """
    avg = np.mean(x)
    std = np.std(x)
    return (x - avg) + std


def prep_x(x: NDArray) -> NDArray:
    """
    Function to performing 'windowing' of the input data

    LSTM uses previous values to predict the next value, so we need to have 
    features of previous values and labels of current values. 

    If given the params of: 
        window = 3
        raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Then we will want our x and y data to look like:
        x = [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6],
             [5, 6, 7],
             [6, 7, 8],
             [7, 8, 9]
            ]

        y = [4, 5, 6, 7, 8, 9, 10]
    """

    window = config["window"]
    n = len(x) - window + 1

    output_shape = (n, window)
    mem_alloc = x.strides[-1]
    strides = (mem_alloc, mem_alloc)

    x_windowed = as_strided(x, shape=output_shape, strides=strides)

    return x_windowed[0:n-1], x_windowed[-1]


def prep_y(x: NDArray) -> NDArray:
    """
    Function to determine labels for windowed x values, simply just need to offset
    by the window value 
    """

    window = config["window"]
    return x[window:]


def train_val_split(x: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Split an array for training/validation

    Returns: (Train, Validation)
    """
    split_index = int(config["train_split"] * len(x))
    return x[:split_index], x[split_index:]

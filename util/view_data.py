import numpy as np
from numpy import nan
from numpy.typing import NDArray
from typing import List
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def view_time_series(sym: str, x: List[str], y: NDArray) -> None: 
    """
    Create a matplotlib figure to display price vs timeseries
    """
    n = len(x)
    interval = n/20

    fig = figure(figsize=(20, 10), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))

    plt.plot(x, y, color="#0000FF")

    xticks = [x[i] if i % interval == 0 else None for i in range(n)]
    plt.xticks([i for i in range(0, n)], xticks, rotation='vertical')

    plt.ylabel('Price')
    plt.xlabel('Date')

    plt.title(f'Daily close price for {sym} from {x[0]} to {x[-1]}', fontsize=16)
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.show()


def view_split_time_series(sym: str, x: List[str], y_train: NDArray, y_val: NDArray) -> None:
    """
    Createa matplotlib figure to display price vs timeseries with train/validation split
    """
    n = len(x)
    interval = n/20

    y_train = np.pad(y_train, (0, n - len(y_train)), 'constant', constant_values=(nan, ))
    y_val = np.pad(y_val, (n - len(y_val), 0), 'constant', constant_values=(nan, ))

    assert len(y_train) == len(y_val)

    fig = figure(figsize=(20, 10), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))

    plt.plot(x, y_train, label="Training Data", color="#0000FF")
    plt.plot(x, y_val, label="Validation Data", color="#FF0000")

    xticks = [x[i] if i % interval == 0 else None for i in range(n)]
    plt.xticks([i for i in range(0, n)], xticks, rotation='vertical')

    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.title(f'Daily close price for {sym} from {x[0]} to {x[-1]}', fontsize=16)
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.show()
from msilib import sequence
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import date
from random import randint
from numpy.typing import ArrayLike
from pandas import DataFrame
from typing import Tuple
from typing import List
from typing import Union
from tensorflow import Tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import MeanAbsoluteError
from tensorflow.python.keras.callbacks import EarlyStopping


from sqlalchemy import create_engine
from sqlalchemy import MetaData
"""
Tensorflow Logging: 
    0 = all messages printed (default)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

config = {
    "input_width": 30,
    "label_width": 5,
    "offset": 5,
    "num_features": None,
    "label_columns": ["close"],
    "train_split": 0.7,
    "val_split": 0.8,
    "batch_size": 32,
    "epochs": 36,
    "patience": 2,
    "loss_func": MeanSquaredError(),
    "optmizer": tf.optimizers.Adam(),
    "metrics": [MeanAbsoluteError()],
    "callbacks": [EarlyStopping()]
}

class Sequence(object):
    """
    Class to hold raw data, normailizd data, and train/val/test split
    for each security 
    """

    def __init__(self, sym: str, timeseries: DataFrame) -> None:
        # Flip the dataframe so start index is oldest date
        timeseries = timeseries.iloc[::-1]

        # Record symbol and original dataframe
        self.sym = sym
        self.timeseries_df: DataFrame = timeseries
        
        # Column headers and Date before stripping them
        self.n_rows, self.n_columns = self.timeseries_df.shape
        self.dates: List[date] = [timestamp.date() for timestamp in timeseries.index.to_list()]

        self.columns: List[str] = timeseries.columns.to_list()
        self.column_indices = {col:i for i, col in enumerate(self.columns)}

        train_val_index = int(self.n_rows*config["train_split"])
        val_test_index = int(self.n_rows*config["val_split"])

        # Slices for easily indexing datasets
        self.train_slice = slice(0, train_val_index)
        self.val_slice = slice(train_val_index, val_test_index)
        self.test_slice = slice(val_test_index, None)

        config["num_features"] = self.n_columns

        # Strip the features into 2D Arrays 
        self.timeseries: ArrayLike = timeseries.to_numpy()
        self.train: ArrayLike = None
        self.val: ArrayLike = None
        self.test: ArrayLike = None

        # Save the corresponding dates in lists
        self.train_dates: List[date] = None
        self.val_dates: List[date] = None
        self.test_dates: List[date] = None

        # Split and normalize the data to feed into the model
        self.__split()
        self.__normalize()


    def __split(self) -> None:
        """ Split self.timeseries and self.dates into appopiate train, val, test sets"""
        self.train = self.timeseries[self.train_slice]
        self.train_dates = self.dates[self.train_slice]

        self.val = self.timeseries[self.val_slice]
        self.val_dates = self.dates[self.val_slice]

        self.test = self.timeseries[self.test_slice]
        self.test_dates = self.dates[self.test_slice]

        # Ensure that no datapoints were loss 
        assert len(self.train) + len(self.val) + len(self.test) == len(self.timeseries)
        assert len(self.train_dates) + len(self.val_dates) + len(self.test_dates) == len(self.dates)

    def __normalize(self) -> None:
        """Normalize the data using the training metrics"""
        self.mean = self.train.mean(axis=0)
        self.std = self.train.std(axis=0)

        self.train = (self.train - self.mean) / self.std
        self.val = (self.val - self.mean) / self.std
        self.test = (self.test - self.mean) / self.std

        # Ensure each feature has a mean value of 0
        assert any(np.around(self.train.mean(axis=0), decimals=5)) == False

    def invert(self, x: ArrayLike) -> ArrayLike:
        """Reverses the normalization for readiability"""
        return (x * self.std) + self.mean


    def __getitem__(self, i: Union[int, slice]) -> ArrayLike:
        """ Dunder to retrieve a datum """
        return self.timeseries[i]


    def __repr__(self) -> str:
        """ Dunder for represetnation """
        return (f'TimeSeries Data - Train/Val/Test: {self.train.shape}/{self.val.shape}/{self.test.shape}')


    def get_splits(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """ Get method for train, val, test arrays """
        return self.train, self.val, self.test


    def plot(self) -> None: 
        """ Plot the train/val/test splits normalized for closing price """
        close_index = self.column_indices['close']

        plt.figure(figsize=(12, 8))
        plt.plot(self.train_dates, self.train[:, close_index], c='#ff0000', label='train')
        plt.plot(self.val_dates, self.val[:, close_index], c='#00ff00', label='val')
        plt.plot(self.test_dates, self.test[:, close_index], c='#0000ff', label='test')

        plt.title(self.sym)
        plt.ylabel('Close [Normed]')
        plt.xlabel('Date')
        plt.legend()
        plt.show()



class WindowGenerator(object):
    """
    Class to handle the "windowing" of a given sequence

    LSTM uses previous values to predict the next value(s), so we need to have 
    inputs of previous values and labels of current values. 

    If given the params of: 
        input_width = 3
        label_width = 1
        offset = 1
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Then we will want our features and labels to look like:
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

    def __init__(self, seq: Sequence) -> None:
        """
        inputs_width = the window size of the input(s) 
        label_width = the window size of label(s)
        offset = the timesteps betwen input_width and the end of label_width 
        """
        
        # Record the sequence
        self.sequence = seq

        # Record indices for column lookups
        self.label_columns = config["label_columns"]
        self.label_columns_indices = {name:i for i, name in enumerate(self.label_columns)}
        self.column_indices = self.sequence.column_indices

        # Dimensions for windowing will want final shape of the window, inputs, and labels to be (batch, time, features)
        self.inputs_width = config["input_width"]
        self.labels_width = config["label_width"]
        self.offset = config["offset"]
        self.total_window_size = self.inputs_width + self.offset

        # Determine the inputs indices
        self.inputs_slice = slice(0, self.inputs_width)
        self.inputs_indices = np.arange(self.total_window_size)[self.inputs_slice]
        
        # Determine indices of the labels
        self.labels_slice = slice(self.total_window_size - self.labels_width, None)
        self.labels_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __split_window(self, features: Tensor) -> Tensor:
        """
        Apply "windowing" to a given tensor
        """

        inputs: Tensor = features[:, self.inputs_slice, :]
        labels: Tensor = features[:, self.labels_slice, :]
        labels: Tensor = tf.stack([labels[:, :, self.sequence.column_indices[name]] for name in self.label_columns], 
                        axis=-1)

        inputs.set_shape([None, self.inputs_width, None])
        labels.set_shape([None, self.labels_width, None])

        return inputs, labels


    def __make_dataset(self, data: ArrayLike) -> tf.data.Dataset:
        ds: tf.data.Dataset = tf.keras.utils.timeseries_dataset_from_array(
                                            data=data,
                                            targets=None,
                                            sequence_length=self.total_window_size,
                                            sequence_stride=1,
                                            shuffle=True,
                                            batch_size=config["batch_size"],
                                        )

        return ds.map(self.__split_window)

    
    def plot_window(self, column: str = None, n: int=3) -> None:
        inputs, labels = next(iter(self.get_train_dataset()))
        plt.figure(figsize=(12, 8))

        if not column:
            column = self.label_columns[0]

        input_column_index = self.column_indices[column]
        label_column_index = self.label_columns_indices[column]

        n = min(n, len(inputs))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.ylabel(f'{column} [normed]')
            plt.plot(self.inputs_indices, inputs[i, :, input_column_index],
                     label='Inputs', marker='.', zorder=-10)

            plt.scatter(self.labels_indices, labels[i, :, label_column_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if i == 0:
                plt.legend()

        plt.xlabel('Time Step')
        plt.show()

    def plot_predictions(self, inputs: ArrayLike, labels: ArrayLike, 
                         predictions: ArrayLike, column: str = None) -> None:

        # Default column to plot if not specified
        if not column:
            column = config["label_columns"][0]

        # Convert 3D to 2D
        inputs, labels, predictions = inputs[0], labels[0], predictions[0]

        # Inverse the normalziation
        inputs = self.sequence.invert(inputs)
        labels = self.sequence.invert(labels)
        predictions = self.sequence.invert(predictions)
    
        # Grab input column index and label column index
        input_col_index = self.column_indices[column]
        label_col_index = self.label_columns_indices[column]

        # Grab the corresponding dates
        input_dates = self.sequence.train_dates + self.sequence.val_dates
        label_dates = self.sequence.test_dates

        # Shape the dates
        input_dates = input_dates[-1*len(inputs):]
        label_dates = label_dates[:len(labels)]

        # Ensure the correct shapes
        assert len(input_dates) == len(inputs)
        assert len(label_dates) == len(labels)

        # Plot hte results
        plt.figure(figsize=(12, 8))
        plt.plot(input_dates, inputs[:, input_col_index],
                    label='Inputs', marker='.', zorder=-10)
        plt.scatter(label_dates, labels[:, label_col_index], edgecolors='k', 
                    label='Labels', c='#2ca02c', s=64)
        plt.scatter(label_dates, predictions[:, label_col_index], edgecolors='k', 
                    marker='X', label='Predictions', c='#ff7f0e', s=64)

        plt.title(self.sequence.sym)
        plt.ylabel(f'{column}')
        plt.xlabel('Dates')  
        plt.legend()
        plt.show()
        return
            
    def get_train_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.sequence.train)


    def get_val_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.sequence.val)


    def get_test_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.sequence.test)
    

    def __repr__(self) -> str:
        return (f'\nTotal window size: {self.total_window_size}\n'
                f'Input indices: {self.inputs_indices}\n'
                f'Label indices: {self.labels_indices}\n'
                f'Label column names: {self.label_columns}\n')



def load_data() -> List[Sequence]:
    """Function to load local stored data"""

    engine = create_engine('sqlite:///../data/test.db')

    metadata = MetaData(engine)
    metadata.reflect(engine)
    syms = list(metadata.tables.keys())
    data = []
    for sym in syms:
        df = pd.read_sql_table(sym, engine.connect(), index_col='date', parse_dates=['date'])
        data.append(Sequence(sym, df))

    assert len(data) == len(syms)
    return data

if __name__ == "__main__":
    data = load_data()
    seq = data[randint(0, len(data) - 1)]
    print(f'Random sequnce selected for ticker={seq.sym} \n')
    print(f'{seq} \n')
    seq.plot()

    window = WindowGenerator(seq)
    print(f'{window} \n')

    train_ds = window.get_test_dataset()
    for inputs, labels in train_ds.take(1):
        print("Dataset Shape: ")
        print(f'\tInputs shape (batch, steps, features): {inputs.shape}')
        print(f'\tLabels shape (batch, steps, features): {labels.shape}\n')

    window.plot_window()
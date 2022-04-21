import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import Series
from typing import Tuple
from typing import List
from typing import Union
from tensorflow import Tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.metrics import MeanAbsoluteError
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import History

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
    "label_width": 10,
    "offset": 10,
    "num_features": None,
    "columns": ["close"],
    "train_split": 0.7,
    "val_split": 0.8,
    "batch_size": 16,
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

    train_split = config["train_split"]
    val_split = config["val_split"]

    def __init__(self, sym: str, timeseries: DataFrame) -> None:
        timeseries = timeseries.iloc[::-1]

        self.sym = sym
        self.ts_df: DataFrame = timeseries
        config["num_features"] = self.ts_df.shape[1]

        self.ts_df_norm: pd.DataFrame = None
        self.train_df: pd.DataFrame = None
        self.val_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None

        self.n_rows, self.n_features = self.ts_df.shape

        self.__split()
        self.__normalize()

    def __split(self) -> None:
        self.train_df = self.ts_df[0:int(self.n_rows*self.train_split)]
        self.val_df = self.ts_df[int(self.n_rows*self.train_split):int(self.n_rows*self.val_split)]
        self.test_df = self.ts_df[int(self.n_rows*self.val_split):]


    def __normalize(self) -> None:
        self.mean = self.train_df.mean()
        self.std = self.train_df.std()

        self.train_df = (self.train_df - self.mean) / self.std
        self.val_df = (self.val_df - self.mean) / self.std
        self.test_df = (self.test_df - self.mean) / self.std


    def invert(self, df: DataFrame) -> DataFrame:
        """Reverses the normalization"""
        return df * self.std + self.mean


    def __getitem__(self, i: Union[int, slice]) -> Union[DataFrame, Series]:
        return self.ts_df.iloc[i]


    def __repr__(self) -> str:
        return (f'Sequence Shape\nTrain/Val/Test: {self.train_df.shape}/{self.val_df.shape}/{self.test_df.shape}')


    def get_splits(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        return self.train_df, self.val_df, self.test_df


    def plot(self) -> None: 
        plt.figure(figsize=(12, 8))

        indices = [i for i in range(len(self.ts_df))]
        train_indices = indices[0:int(self.n_rows*self.train_split)]
        val_indices = indices[int(self.n_rows*self.train_split):int(self.n_rows*self.val_split)]
        test_indices = indices[int(self.n_rows*self.val_split):]

        plt.plot(train_indices, self.train_df['close'], c='#ff0000', label='train')
        plt.plot(val_indices, self.val_df['close'], c='#00ff00', label='val')
        plt.plot(test_indices, self.test_df['close'], c='#0000ff', label='test')
        
        plt.ylabel('Close [normed]')
        plt.xlabel('Time Step')
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

    def __init__(self, inputs_width: int, labels_width: int, offset: int, 
                 seq: Sequence, label_columns: List[str]) -> None:
        """
        inputs_width = the window size of the input(s) 
        label_width = the window size of label(s)
        offset = the timesteps betwen input_width and the end of label_width 
        """
        
        self.train_df, self.val_df, self.test_df = seq.get_splits()
        self.label_columns = label_columns
        self.label_columns_indices = {name:i for i, name in enumerate(label_columns)}
        self.column_indices = {name:i for i, name in enumerate(self.train_df.columns)}

        self.inputs_width = inputs_width
        self.labels_width = labels_width
        self.offset = offset

        self.total_window_size = inputs_width + offset

        # Determine the inputs indices
        self.inputs_slice = slice(0, inputs_width)
        self.inputs_indices = np.arange(self.total_window_size)[self.inputs_slice]
        
        # Determine indices of the labels
        self.labels_start = self.total_window_size - self.labels_width
        self.labels_slice = slice(self.labels_start, None)
        self.labels_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __split_window(self, features: Tensor) -> Tensor:
        """
        Apply "windowing" to a given tensor
        """

        inputs: Tensor = features[:, self.inputs_slice, :]
        labels = features[:, self.labels_slice, :]
        labels: Tensor = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], 
                        axis=-1)

        inputs.set_shape([None, self.inputs_width, None])
        labels.set_shape([None, self.labels_width, None])

        return inputs, labels


    def __make_dataset(self, data: DataFrame) -> tf.data.Dataset:
        data = np.array(data, dtype=float)
        ds: tf.data.Dataset = tf.keras.utils.timeseries_dataset_from_array(
                                            data=data,
                                            targets=None,
                                            sequence_length=self.total_window_size,
                                            sequence_stride=1,
                                            shuffle=True,
                                            batch_size=config["batch_size"],
                                        )

        return ds.map(self.__split_window)

    
    def plot(self, model: Model=None, column: str = config["columns"][0], n: int=3) -> None:
        inputs, labels = next(iter(self.get_train_dataset()))
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[column]

        n = min(n, len(inputs))
        for i in range(n):
            plt.subplot(n, 1, i+1)
            plt.ylabel(f'{column} [normed]')
            plt.plot(self.inputs_indices, inputs[i, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            label_col_index = self.label_columns_indices.get(column, None)

            plt.scatter(self.labels_indices, labels[i, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.labels_indices, predictions[i, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if i == 0:
                plt.legend()

        plt.xlabel('Day')
        plt.show()

    def plot_predictions(self, sym: str, inputs: np.ndarray, labels: np.ndarray, 
                         predictions: np.ndarray, column: str = config["columns"][0]) -> None:

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[column]

        plt.ylabel(f'{column} [normed]')

        
        input_indicies = range(len(inputs))
        plt.plot(input_indicies, inputs[:, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

        label_indicies = range(len(inputs), len(inputs) + len(labels[0]))
        label_col_index = self.label_columns_indices.get(column, None)

        plt.scatter(label_indicies, labels[0, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
        plt.scatter(label_indicies, predictions[0, :, label_col_index], edgecolors='k', marker='X', label='Predictions', c='#ff7f0e', s=64)
        plt.title(sym)
        plt.ylabel('Close')
        plt.xlabel('Date')
        plt.legend()
        plt.show()
            
    def get_train_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.train_df)


    def get_val_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.val_df)


    def get_test_dataset(self) -> tf.data.Dataset:
        return self.__make_dataset(self.test_df)
    

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

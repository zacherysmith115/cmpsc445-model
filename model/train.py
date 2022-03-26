
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from pandas import Series
from typing import Tuple
from typing import List
from typing import Union
from tensorflow import Tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v1 import Adam
from tensorflow.python.keras.metrics import MeanAbsoluteError
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import History

"""
Tensorflow Logging: 
    0 = all messages printed (default)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

config = {
    "input_width": 5,
    "label_width": 5,
    "offset": 1,
    "columns": ["4. close"],
    "train_split": 0.7,
    "val_split": 0.8,
    "test_split": 0.9,
    "batch_size": 32,
    "epochs": 24,
    "patience": 2,
    "loss_func": MeanSquaredError(),
    "optmizer": Adam(),
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
    test_split = config["test_split"]

    def __init__(self, sym: str, timeseries: DataFrame) -> None:
        self.sym = sym
        self.ts_df: DataFrame = timeseries.iloc[::-1]

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
        self.test_df = self.ts_df[int(self.n_rows*self.test_split):]


    def __normalize(self) -> None:
        mean = self.train_df.mean()
        std = self.train_df.std()

        self.train_df = (self.train_df - mean) / std
        self.val_df = (self.val_df - mean) / std
        self.test_df = (self.test_df - mean) / std

    def __getitem__(self, i: Union[int, slice]) -> Union[DataFrame, Series]:
        return self.ts_df.iloc[i]

    def __repr__(self) -> str:
        return f'\n{str(self.train_df)}\n'

    def get_splits(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        return self.train_df, self.val_df, self.test_df


class WindowGenerator(object):
    """
    Class to handle the "windowing" of a given sequence

    LSTM uses previous values to predict the next value(s), so we need to have 
    features of previous values and labels of current values. 

    If given the params of: 
        feature_width = 3
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
                 batch: Sequence, label_columns: List[str]) -> None:
        """
        feature_width =the window size of the feature(s) 
        label_width = the window size of label(s)
        offset = the timesteps betwen feature_width and label width 
        """
        
        self.train_df, self.val_df, self.test_df = batch.get_splits()
        self.label_columns = label_columns
        self.label_columns_indices = {name:i for i, name in enumerate(label_columns)}
        self.column_indices = {name:i for i, name in enumerate(self.train_df.columns)}

        self.inputs_width = inputs_width
        self.labels_width = labels_width
        self.offset = offset

        self.total_window_size = inputs_width + offset

        # Detemine the inputs indices
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
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], 
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


def read_data(src: str) -> List[Sequence]:
    """Function to load local stored data"""

    data = []
    with open(src, 'r') as f:
        for r in json.loads(f.read()): 
            
            sym = r["2. Symbol"]
            df = DataFrame.from_dict(r["6. Time Series"], dtype=float)
            data.append(Sequence(sym, df))

    return data

def fit(model: Model, window: WindowGenerator) -> History:
    model.compile(loss=config["loss_func"],
                  optimizer=config["optmizer"],
                  metrics=config["metrics"])

    history = model.fit(window.get_train_dataset(),
                           epochs=config["epochs"],
                           validation_data=window.get_val_dataset(),
                           callbacks=config["callbacks"])

    return history


data = read_data('../data/data.json')
data = data[0]
print(data)

w = WindowGenerator(config["input_width"],
                    config["label_width"],
                    config["offset"],
                    data,
                    config["columns"])

print(w)

train_ds = w.get_train_dataset()
val_ds = w.get_val_dataset()
test_ds = w.get_test_dataset()

for inputs, labels in train_ds.take(1):
    print(f'Inputs shape (batch, steps, features): {inputs.shape}')
    print(f'Labels shape (batch, steps, features): {labels.shape}')


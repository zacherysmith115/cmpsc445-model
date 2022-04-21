from random import randint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.callbacks import History


from preprocess import WindowGenerator
from preprocess import load_data
from preprocess import config


def fit(model: Model, window: WindowGenerator) -> History:
    model.compile(loss=config["loss_func"],
                  optimizer=config["optmizer"])

    history = model.fit(window.get_train_dataset(),
                        epochs=config["epochs"],
                        validation_data=window.get_val_dataset(),
                        callbacks=config["callbacks"])

    return history


if __name__ == "__main__":
    #Load the data from the database into a list of Sequence classes
    timeseries_sequences = load_data()

    train = False
    early_break = True
    if train:

        lstm: Sequential = tf.keras.models.Sequential([
                        LSTM(16, return_sequences=False),
                        Dense(config["label_width"]*config["num_features"],
                            kernel_initializer=tf.initializers.zeros()),
                        Reshape([config["label_width"], config["num_features"]])
                        ])


        n = len(timeseries_sequences)

        # Loop through each time series sequence and train on it
        for i, sequence in enumerate(timeseries_sequences):
            print(f'[{i}/{n}] Training on timeseries for ticker {sequence.sym}...')

            window = WindowGenerator(config["input_width"], config["label_width"], config["offset"],
                                     sequence, config["columns"])
            

            train_ds = window.get_train_dataset()
            val_ds = window.get_val_dataset()
            test_ds = window.get_test_dataset()

            for inputs, labels in train_ds.take(1):
                print(f'\tInputs shape (batch, steps, features): {inputs.shape}')
                print(f'\tLabels shape (batch, steps, features): {labels.shape}\n')

            history = fit(lstm, window)

            # Limited training flag for testing
            if early_break and i == 10:
                break

        lstm.save('./weights/lstm')
    else:
        lstm: Sequential = load_model('./weights/lstm')


    # Randomly select a sequence of data to evaluate against
    rand_index = randint(0, len(timeseries_sequences))
    sequence = timeseries_sequences[rand_index]
    # sequence.plot()
    sym = sequence.sym
    window = WindowGenerator(config["input_width"], config["label_width"], config["offset"],
                                     sequence, config["columns"])


    # Get and shape the inputs 
    inputs: pd.DataFrame = pd.concat([window.train_df, window.val_df])
    inputs =  inputs.tail(30)
    inputs = np.array(inputs, dtype=float)
    inputs: Tensor = tf.convert_to_tensor(inputs[None, :, :], dtype=tf.float32)


    # Get and shape the labels
    labels = np.array(window.test_df.head(5))
    labels = labels[None, :, :]
    predictions: np.ndarray = lstm(inputs)

    print(f'Inputs: {inputs.shape}')
    print(f'Labels: {labels.shape}')
    print(f'Predictions: {predictions.shape}')

    # Plot predicted values vs labels
    window.plot_predictions(inputs, labels, predictions)



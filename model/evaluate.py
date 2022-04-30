import numpy as np
import tensorflow as tf
from random import randint
from numpy.typing import ArrayLike
from tensorflow import Tensor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model

from preprocess import config
from preprocess import WindowGenerator
from preprocess import load_data

if __name__ == "__main__":
    timeseries_sequences = load_data()
    lstm: Sequential = load_model('./weights/lstm')

    # Randomly select a sequence of data to evaluate against
    rand_index = randint(0, len(timeseries_sequences) - 1)
    sequence = timeseries_sequences[rand_index]
    window = WindowGenerator(sequence)

    # Grab the end of val train and val data to use for input
    inputs = np.concatenate((sequence.train, sequence.val), axis=0)
    assert len(inputs) == len(sequence.train) + len(sequence.val)

    # Shape the inputs
    inputs_width = config["input_width"]
    inputs = inputs[-1*inputs_width:]
    assert len(inputs) == inputs_width
    inputs: Tensor = tf.convert_to_tensor(inputs[None, :, :], dtype=tf.float32)

    # Grab the unseen test data 
    labels = sequence.test
    label_width = config["label_width"]
    labels = labels[:label_width]
    assert len(labels) == label_width
    labels = labels[None, :, :]
    
    # Make predictions
    predictions: ArrayLike = lstm(inputs)

    print(f'Inputs: {inputs.shape}')
    print(f'Labels: {labels.shape}')
    print(f'Predictions: {predictions.shape}')

    # Plot predicted values vs labels
    window.plot_predictions(inputs, labels, predictions)

    x_test = []
    y_test = []

    for sequence in timeseries_sequences:

        window = WindowGenerator(sequence)

        # Grab the end of val train and val data to use for input
        inputs = np.concatenate((sequence.train, sequence.val), axis=0)
        assert len(inputs) == len(sequence.train) + len(sequence.val)

        # Shape the inputs
        inputs_width = config["input_width"]
        inputs = inputs[-1*inputs_width:]
        assert len(inputs) == inputs_width
        inputs: Tensor = tf.convert_to_tensor(inputs[None, :, :], dtype=tf.float32)

        # Grab the unseen test data 
        labels = sequence.test
        label_width = config["label_width"]
        labels = labels[:label_width]
        assert len(labels) == label_width
        labels = labels[None, :, :]

        x_test.append(inputs)
        y_test.append(labels)

    eval = lstm.evaluate(x_test, y_test, return_dict=True)
    for metric in eval:
        print(metric + ":", eval[metric])
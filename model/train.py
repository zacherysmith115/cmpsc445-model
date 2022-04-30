import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.callbacks import History

from preprocess import WindowGenerator
from preprocess import load_data
from preprocess import config


def fit(model: Model, window: WindowGenerator) -> History:
    model.compile(loss=config["loss_func"],
                  optimizer=config["optmizer"], metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

    history = model.fit(window.get_train_dataset(),
                        epochs=config["epochs"],
                        validation_data=window.get_val_dataset(),
                        callbacks=config["callbacks"])

    return history


if __name__ == "__main__":
    #Load the data from the database into a list of Sequence classes
    timeseries_sequences = load_data()

    early_break = True
    break_point = 10

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

        window = WindowGenerator(sequence)
        history = fit(lstm, window)

        # Limited training flag for testing
        if early_break and break_point == 10:
            break

    lstm.save('./weights/lstm')



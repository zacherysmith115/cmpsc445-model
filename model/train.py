from model.data_prep import *
import tensorflow as tf 


class DataSet(object):

    def __init__(self) -> None:
        raw_data = read_data()

        self.sym, self.dates, data = raw_data[0] 

        normalized_data = normalize(data)
        x, _ = prep_x(normalized_data)
        y = prep_y(normalized_data)

        self.train_x, self.val_x = train_val_split(x)
        self.train_y, self.val_y = train_val_split(y)

        assert len(self.train_x) == len(self.train_y)
        assert len(self.val_x) == len(self.val_y)

def train() -> None:
    pass
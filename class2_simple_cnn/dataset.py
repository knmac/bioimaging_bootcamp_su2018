from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import urllib.request
import pickle
import os
import gzip
import numpy as np
import random


class Dataset(object):
    def __init__(self):
        """
        """
        # initialize the batch indices
        self.train_idx = 0
        self.val_idx = 0
        self.test_idx = 0
        pass

    def load_dataset(self):
        """Load MNIST dataset (download if necessary).

        The data are recorded in this class object, including training,
        validation, and testing data. Each type of data is stored in two
        variable: image X (ndaray of shape (N, 28, 28)) and label y (ndaray of
        shape (N,)).
        """
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            print("Downloading MNIST dataset...")
            urllib.request.urlretrieve(url, filename)

        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        X_train, y_train = data[0]
        X_val, y_val = data[1]
        X_test, y_test = data[2]

        self.X_train = X_train.reshape((-1, 28, 28))
        self.X_val = X_val.reshape((-1, 28, 28))
        self.X_test = X_test.reshape((-1, 28, 28))

        self.y_train = y_train.astype(np.uint8)
        self.y_val = y_val.astype(np.uint8)
        self.y_test = y_test.astype(np.uint8)
        pass

    def shuffle_training_data(self):
        """Shuffle training data
        """
        full_data = zip(self.X_train, self.y_train)
        random.shuffle(full_data)
        self.X_train, self.y_train = zip(*full_data)
        pass

    def _produce_batch(self, x, y, idx, batch_size):
        """Produce a general batch (train, val, or test). 

        Allow smaller batch at the end of the data. Should be used via
        produce_train_batch, produce_val_batch, or produce_test_batch.

        Args:
            x: full images
            y: full labels
            idx: current batch index
            batch_size: size of the batch

        Returns:
            x_batch: batch of images
            y_batch: batch of labels
            new_idx: new batch index
        """
        # produce batch
        x_batch = x[idx:idx+batch_size]
        y_batch = y[idx:idx+batch_size]

        # update current batch index
        new_idx = idx + batch_size

        # reset batch index if out of bound
        if new_idx >= len(y):
            new_idx = 0
        return x_batch, y_batch, new_idx

    def produce_train_batch(self, batch_size):
        """Produce training batch
        """
        (x_batch,
         y_batch,
         self.train_idx) = self._produce_batch(self.X_train, self.y_train,
                                               self.train_idx, batch_size)
        return x_batch, y_batch

    def produce_val_batch(self, batch_size):
        """Produce validation batch
        """
        (x_batch,
         y_batch,
         self.val_idx) = self._produce_batch(self.X_val, self.y_val,
                                             self.val_idx, batch_size)
        return x_batch, y_batch

    def produce_test_batch(self, batch_size):
        """Produce testing batch
        """
        (x_batch,
         y_batch,
         self.test_idx) = self._produce_batch(self.X_test, self.y_test,
                                              self.test_idx, batch_size)
        return x_batch, y_batch

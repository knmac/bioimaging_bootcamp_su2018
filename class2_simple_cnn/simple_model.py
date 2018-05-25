from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class SimpleModel(object):
    def __init__(self, height, width, learning_rate):
        """
        """
        self.sess = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, height, width])
        self.y_placeholder = tf.placeholder(tf.float32, [None])
        self.learning_rate = learning_rate

        # build graph
        self.inference(self.x_placeholder)
        pass

    def inference(self, x):
        """Forwarding function. This contains the main network architecture

        Args:
            x: input tensor, shape of (?, height, width)
        """
        pass

    def update(self):
        """Update model weights in training
        """
        pass

    def predict(self, x):
        """Predict the input
        """
        return

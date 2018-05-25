from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import matplotlib.pyplot as plt


from simple_model import SimpleModel
from dataset import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_steps', 20000, 'Number of update steps to run.')


def train_model(data):
    """Train model
    """
    for epoch in range(FLAGS.num_steps):
        # shuffle data
        data.shuffle_training_data()

        # train model with multiple batches
        # check performance frequently
    pass


def eval_model():
    """Evaluate model
    """
    pass


def main(_):
    # load MNIST data
    data = Dataset()
    data.load_dataset()

    # show an example image
    plt.imshow(data.X_train[0], cmap='gray')
    plt.show()
    pass


if __name__ == '__main__':
    tf.app.run()

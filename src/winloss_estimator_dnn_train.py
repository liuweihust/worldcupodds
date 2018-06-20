from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from datasets import SoccerDb

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=60000, type=int,
                    help='number of training steps')
parser.add_argument('--model_dir', default='/tmp/train/', type=str, help='path to save train model')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y) = SoccerDb.load_traindata()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[128, 128,128],
        # The model must choose between 3 classes.
        model_dir=args.model_dir,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:SoccerDb.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

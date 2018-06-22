from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--dataset', default='310', type=str, help='dataset:310,Pts')
parser.add_argument('--datadir', default='../data/', type=str, help='which path csv data located')
parser.add_argument('--model', default='dnn', type=str, help='model:dnn,handmade,default:dnn')
parser.add_argument('--train_steps', default=20000, type=int,
                    help='number of training steps')
parser.add_argument('--model_dir', default='/tmp/train/', type=str, help='path to save train model')

def main(argv):
    args = parser.parse_args(argv[1:])

    dataset = dataset_factory.get_dataset(args.dataset)
    (train_x, train_y) = dataset.get_split('train',args.datadir)

    net_fn = nets_factory.get_network(args.model,)
    classifier = net_fn(features=train_x.keys(),model_dir=args.model_dir)
    #classifier = net_fn(features=train_x.keys())

    # Train the Model.
    classifier.train(
        input_fn=lambda:dataset.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
